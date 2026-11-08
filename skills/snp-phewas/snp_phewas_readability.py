"""Is a GWAS Catalog study readable as a harmonized file by position, and if not, why.

Used to build the Million Veteran Program catalog (`snp_phewas_mvp_catalog`). Readability is
decided by a DIRECT PROBE of the study's `harmonised/` directory, and the reason for an
unreadable study comes from EBI's harmonization TRACKER (`Harm_sumstats_status.db`). The
manifest `harmonised_list.txt` decides nothing: the 2026-10-19 MVP catalog was built from it
and called 1,580 studies "not harmonized", and on 2026-10-31 1,575 of them were range-readable
(files dated 2025-05 to 2025-07) while none was in that day's manifest.

Three layers, so each is testable without the others:

  `probe_study`     network: list `harmonised/`, pick the data file, see its `.tbi`, and
                    range-read the first bytes to prove BGZF; when no data file is served,
                    list the study's parent directory too. Raw observations only.
  `classify`        pure: (tracker row or none, probe) -> one `Readability` with a `Reason`.
  `classify_batch`  the offline classification of probes already taken (the vendored
                    snapshots rebuild the catalog byte-identically from it); `assess`
                    probes a universe politely first.

Every reason carries a `Verdict`, kept apart on purpose:

  * `readable`      the probe read a BGZF block of a data file that has its `.tbi`.
  * `cannot_ever`   the tracker says it will not harmonize the study (`cannot_harm` with
                    no failed step) and the probe finds no file. Re-derived on every snapshot.
  * `not_yet`       unreadable today for a reason that can change (queued, a failed step EBI
                    retries, a file the tracker records that the server does not serve, no
                    index yet, refused). Re-checked on every refresh.
  * `undetermined`  the probe got no answer (no connection, timeout, 429, 5xx). Never
                    recorded as unreadable: the catalog build refuses instead.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

#: Semver of the classification rule. Bumps when any (tracker, probe) input moves to a
#: different reason, so a pinned snapshot says which rule it was classified with.
READABILITY_VERSION = "1.1.0"

DEFAULT_FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

GCST_ID = re.compile(r"\AGCST\d+\Z")

#: The tracker's token for "no value".
NA = "NA"

#: `Harm_status`'s domain, measured over the 209,730 rows of the 2026-10-30 tracker:
#: harmonised 164,546 / cannot_harm 32,930 / harmonising 11,019 / backlog 1,235. A status
#: outside it stops the classification rather than land in an existing bucket.
KNOWN_HARM_STATUS: frozenset[str] = frozenset(
    {"harmonised", "cannot_harm", "harmonising", "backlog"})

#: HTTP answers that are not an answer about the study: no connection (0), timeout,
#: throttling, and server errors.
_TRANSPORT_CODES = frozenset({0, 408, 425, 429})

#: The BGZF header: gzip magic, deflate, FEXTRA set, and the `BC` extra subfield at byte 12.
_BGZF_MAGIC = b"\x1f\x8b\x08\x04"
_RANGE_BYTES = 18

_LISTING_TITLE = re.compile(r"<title>\s*Index of /", re.IGNORECASE)


def is_transport(code: int | None) -> bool:
    """True when an HTTP status (0 = no response at all) says nothing about the study."""
    return code is not None and (code in _TRANSPORT_CODES or code >= 500)


class Verdict(str, Enum):
    READABLE = "readable"
    CANNOT_EVER = "cannot_ever"
    NOT_YET = "not_yet"
    UNDETERMINED = "undetermined"


class Reason(Enum):
    """Why a study is or is not readable. Each member carries its verdict and a plain label,
    so a new reason cannot silently join another's bucket."""

    READABLE = ("readable", Verdict.READABLE,
                "readable by position (harmonized file with its index)")
    TRACKER_CANNOT_HARM = ("tracker_cannot_harm", Verdict.CANNOT_EVER,
                           "the GWAS Catalog records that it cannot harmonize this study")
    TRACKER_FAILED_STEP = ("tracker_failed_step", Verdict.NOT_YET,
                           "the GWAS Catalog's harmonization of this study failed at a step")
    TRACKER_QUEUED = ("tracker_queued", Verdict.NOT_YET,
                      "queued for harmonization by the GWAS Catalog, no file yet")
    TRACKER_HARMONISED_FILE_MISSING = (
        "tracker_harmonised_file_missing", Verdict.NOT_YET,
        "recorded as harmonized by the GWAS Catalog, but no harmonized file is on its server")
    UNTRACKED_UNHARMONISED = (
        "untracked_unharmonised", Verdict.NOT_YET,
        "held by the GWAS Catalog only as an unharmonized upload, which cannot be read by position")
    UNTRACKED_NO_DEPOSIT = ("untracked_no_deposit", Verdict.NOT_YET,
                            "no summary statistics deposited at the GWAS Catalog")
    FILE_WITHOUT_INDEX = ("file_without_index", Verdict.NOT_YET,
                          "a harmonized file without the position index needed to read it")
    RANGE_READ_REFUSED = ("range_read_refused", Verdict.NOT_YET,
                          "the server refused a partial read of the harmonized file")
    NOT_BGZF = ("not_bgzf", Verdict.NOT_YET,
                "the harmonized file is not block-compressed, so it cannot be read by position")
    ACCESS_DENIED = ("access_denied", Verdict.NOT_YET,
                     "the GWAS Catalog server refused access to the study's files")
    TRANSPORT = ("transport", Verdict.UNDETERMINED,
                 "the GWAS Catalog server could not be reached; readability not determined")

    def __init__(self, code: str, verdict: Verdict, label: str) -> None:
        self.code = code
        self.verdict = verdict
        self.label = label


REASON_LABELS: dict[str, str] = {r.code: r.label for r in Reason}


class ReadabilityError(ValueError):
    """The inputs cannot be classified: a status outside the measured domain, an incomplete
    probe, a probe or tracker row for another study, or a universe the rows do not cover."""


class UndeterminedStudies(RuntimeError):
    """A study could not be probed (transport); it must never be written as unreadable."""


# --- the tracker -------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackerRow:
    """One study's row of the harmonization tracker, read faithfully."""

    study: str
    harm_status: str
    harm_failstep: str
    harm_exitcode: str
    harm_failed_reason: str
    latest_harm_start_date: str
    harm_account: str
    raw_file_type: str
    raw_genome_build: str
    raw_coordinate_system: str
    effect_size_type: str
    harm_drop_rate: str
    liftover_drop_rate: str

    @property
    def failstep_set(self) -> bool:
        return self.harm_failstep != NA


TRACKER_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(TrackerRow))
#: The tracker's own column for each `TrackerRow` field.
_DB_COLUMNS = {"study": "Study", "harm_status": "Harm_status", "harm_failstep": "Harm_failstep",
               "harm_exitcode": "Harm_exitcode", "harm_failed_reason": "Harm_failed_reason",
               "latest_harm_start_date": "Latest_harm_start_date",
               "harm_account": "Harm_account", "raw_file_type": "Raw_file_type",
               "raw_genome_build": "Raw_genome_build",
               "raw_coordinate_system": "Raw_coordinate_system",
               "effect_size_type": "Effect_size_type", "harm_drop_rate": "Harm_drop_rate",
               "liftover_drop_rate": "Liftover_drop_rate"}


@dataclass(frozen=True)
class Tracker:
    rows_by_study: dict[str, TrackerRow]
    last_run: str | None
    sha256: str


def read_tracker_db(db_path: str | Path) -> Tracker:  # pragma: no cover - needs the db
    """The tracker database, read-only, refusing one the rule must not read from: SQLite
    `integrity_check` must say ok, and no study may carry two rows that disagree."""
    db_path = str(db_path)
    if not Path(db_path).is_file():
        raise ReadabilityError(f"{db_path}: no such file")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ReadabilityError(f"tracker {db_path}: integrity check says {integrity!r}")
        cols = {row[1] for row in con.execute("PRAGMA table_info('studies')").fetchall()}
        missing = sorted(set(_DB_COLUMNS.values()) - cols)
        if missing:
            raise ReadabilityError(f"tracker {db_path}: `studies` lacks columns {missing}")
        last = con.execute("SELECT timestamp FROM last_run").fetchall() \
            if con.execute("SELECT 1 FROM sqlite_master WHERE name = 'last_run'").fetchone() \
            else []
        select = ", ".join(f'"{_DB_COLUMNS[f]}"' for f in TRACKER_COLUMNS)
        rows: dict[str, TrackerRow] = {}
        conflicting: list[str] = []
        for values in con.execute(f"SELECT {select} FROM studies"):
            rec = TrackerRow(*(str(v) if v not in (None, "") else NA for v in values))
            if rec.study in rows:
                if rows[rec.study] != rec:
                    conflicting.append(rec.study)
                continue
            rows[rec.study] = rec
    finally:
        con.close()
    if conflicting:
        raise ReadabilityError(f"tracker {db_path}: studies with conflicting rows "
                               f"{conflicting[:5]}")
    sha = hashlib.sha256(Path(db_path).read_bytes()).hexdigest()
    return Tracker(rows, str(last[0][0]) if last else None, sha)


# --- the probe ---------------------------------------------------------------------------

def relative_harmonised_dir(study: str) -> str:
    """`GCST<lo>-GCST<hi>/<study>/harmonised`, relative to the FTP base (buckets of 1,000)."""
    if not GCST_ID.match(study):
        raise ReadabilityError(f"not a GCST accession: {study!r}")
    n = int(study[4:])
    lo = ((n - 1) // 1000) * 1000 + 1
    return f"GCST{lo:06d}-GCST{lo + 999:06d}/{study}/harmonised"


@dataclass(frozen=True)
class HarmonisedDirProbe:
    """What the FTP served for one study, raw. `*_http` 0 means no response at all."""

    study: str
    probed_utc: str
    harm_dir_http: int
    data_file: str = ""
    data_files: tuple[str, ...] = ()
    tbi_present: bool = False
    range_http: int | None = None
    bgzf: bool | None = None
    last_modified: str = ""
    study_dir_http: int | None = None
    study_dir_files: tuple[str, ...] = ()
    detail: str = ""


PROBE_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(HarmonisedDirProbe))


def _probe_to_row(p: HarmonisedDirProbe) -> dict[str, str]:
    def s(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, tuple):
            return "|".join(v)
        return str(v).replace("\t", " ").replace("\n", " ")
    return {c: s(getattr(p, c)) for c in PROBE_COLUMNS}


def _probe_from_row(r: Mapping[str, str]) -> HarmonisedDirProbe:
    def opt_int(v: str) -> int | None:
        return int(v) if v != "" else None

    def opt_bool(v: str) -> bool | None:
        if v == "":
            return None
        if v not in ("true", "false"):
            raise ReadabilityError(f"not a boolean: {v!r}")
        return v == "true"

    def tup(v: str) -> tuple[str, ...]:
        return tuple(v.split("|")) if v else ()

    return HarmonisedDirProbe(
        study=r["study"], probed_utc=r["probed_utc"], harm_dir_http=int(r["harm_dir_http"]),
        data_file=r["data_file"], data_files=tup(r["data_files"]),
        tbi_present=opt_bool(r["tbi_present"]) or False, range_http=opt_int(r["range_http"]),
        bgzf=opt_bool(r["bgzf"]), last_modified=r["last_modified"],
        study_dir_http=opt_int(r["study_dir_http"]), study_dir_files=tup(r["study_dir_files"]),
        detail=r["detail"])


def choose_data_file(study: str, names: Iterable[str]) -> str:
    """The harmonized data file a reader should use: `<study>.h.tsv.gz` when served, else a
    file without a `_build` suffix (the GRCh37 copies some studies carry), else the first."""
    data = sorted(n for n in names if n.endswith(".h.tsv.gz"))
    if not data:
        return ""
    if f"{study}.h.tsv.gz" in data:
        return f"{study}.h.tsv.gz"
    plain = [n for n in data if "_build" not in n]
    return (plain or data)[0]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _list_dir(url: str, session, timeout_s: float) -> tuple[int, list[str]]:  # pragma: no cover - network
    """(status, file names) of an EBI FTP-over-HTTP directory. 0 on a transport error AND on a
    200 whose body is not a directory listing (a maintenance page), so neither is read as
    "no data"."""
    import requests

    try:
        r = session.get(url, timeout=timeout_s)
    except requests.RequestException:
        return 0, []
    if r.status_code != 200:
        return r.status_code, []
    if not _LISTING_TITLE.search(r.text):
        return 0, []
    return 200, [href.rstrip("/") for href in re.findall(r'href="([^"]+)"', r.text)
                 if not href.startswith("?") and not href.startswith("/")]


def _range_read(session, url: str, timeout_s: float) -> tuple[int, bool | None, str]:  # pragma: no cover - network
    """(HTTP status, is BGZF, Last-Modified) from a Range read of the first bytes. Any
    failure to get an answer, including a connection dropped while the body streams, is
    code 0 (transport)."""
    import requests
    import urllib3

    no_answer = (requests.RequestException, urllib3.exceptions.HTTPError, OSError)
    try:
        r = session.get(url, headers={"Range": f"bytes=0-{_RANGE_BYTES - 1}"},
                        timeout=timeout_s, stream=True)
    except no_answer:
        return 0, None, ""
    try:
        if r.status_code != 206:
            return r.status_code, None, r.headers.get("Last-Modified", "")
        head = r.raw.read(_RANGE_BYTES) if getattr(r, "raw", None) is not None else r.content
        head = head[:_RANGE_BYTES]
        return 206, head[:4] == _BGZF_MAGIC and head[12:14] == b"BC", \
            r.headers.get("Last-Modified", "")
    except no_answer:
        return 0, None, ""
    finally:
        r.close()


def probe_study(study: str, *, session, ftp_base: str = DEFAULT_FTP_BASE,
                timeout_s: float = 60.0, clock: Callable[[], str] = _utc_now
                ) -> HarmonisedDirProbe:  # pragma: no cover - network
    """Probe one study; a transport failure is data (code 0), never a raise. Requests, only
    as needed: GET `harmonised/`; a Range GET of the chosen data file when its `.tbi` is
    listed; GET the parent study directory when `harmonised/` holds no data file, which is
    what separates an unharmonized upload from no deposit at all."""
    base = f"{ftp_base.rstrip('/')}/{relative_harmonised_dir(study)}"
    now = clock()
    hcode, names = _list_dir(base + "/", session, timeout_s)
    data = tuple(sorted(n for n in names if n.endswith(".h.tsv.gz")))
    # A data file that has its index wins over one otherwise preferred that does not.
    chosen = (choose_data_file(study, [n for n in data if n + ".tbi" in names])
              or choose_data_file(study, names))
    tbi = bool(chosen) and (chosen + ".tbi") in names
    rcode: int | None = None
    bgzf: bool | None = None
    lastmod = ""
    if chosen and tbi:
        rcode, bgzf, lastmod = _range_read(session, f"{base}/{chosen}", timeout_s)
    scode: int | None = None
    sfiles: tuple[str, ...] = ()
    if not chosen and hcode in (200, 404, 410):
        scode, snames = _list_dir(base.rsplit("/harmonised", 1)[0] + "/", session, timeout_s)
        sfiles = tuple(sorted(n for n in snames if n != "harmonised"))
    return HarmonisedDirProbe(
        study=study, probed_utc=now, harm_dir_http=hcode, data_file=chosen, data_files=data,
        tbi_present=tbi, range_http=rcode, bgzf=bgzf, last_modified=lastmod,
        study_dir_http=scode, study_dir_files=sfiles)


# --- the classification (pure) -----------------------------------------------------------

@dataclass(frozen=True)
class Readability:
    """One study's answer: the reason, the tracker row it rests on, the probe it read."""

    study: str
    reason: Reason
    probe: HarmonisedDirProbe
    tracker: TrackerRow | None
    detail: str = ""

    @property
    def readable(self) -> bool:
        return self.reason is Reason.READABLE

    @property
    def verdict(self) -> Verdict:
        return self.reason.verdict

    @property
    def data_path(self) -> str:
        """The data file relative to the FTP base, or "" when none is served."""
        if not self.probe.data_file:
            return ""
        return f"{relative_harmonised_dir(self.study)}/{self.probe.data_file}"


def _no_file_reason(tracker: TrackerRow | None, probe: HarmonisedDirProbe) -> tuple[Reason, str]:
    if tracker is None:
        if is_transport(probe.study_dir_http):
            return Reason.TRANSPORT, f"study directory: HTTP {probe.study_dir_http}"
        if probe.study_dir_http is not None and probe.study_dir_http not in (200, 404, 410):
            return Reason.ACCESS_DENIED, f"study directory: HTTP {probe.study_dir_http}"
        if probe.study_dir_http == 200 and probe.study_dir_files:
            return Reason.UNTRACKED_UNHARMONISED, "not in the tracker; upload: " + ", ".join(
                probe.study_dir_files[:4])
        if probe.study_dir_http in (200, 404, 410):
            return Reason.UNTRACKED_NO_DEPOSIT, (
                f"not in the tracker; study directory HTTP {probe.study_dir_http}")
        raise ReadabilityError(f"{probe.study}: no data file and no study-directory listing; "
                               f"the probe is incomplete")
    status = tracker.harm_status
    if status not in KNOWN_HARM_STATUS:
        raise ReadabilityError(
            f"{tracker.study}: Harm_status {status!r} is outside the measured domain "
            f"{sorted(KNOWN_HARM_STATUS)}; the rule cannot place it")
    # A failed step outranks the status: `cannot_harm` WITH a failed step is a step that
    # failed (exit 137 is a kill, typically out of memory), so it is re-checked every run.
    # Only `cannot_harm` with no failed step is the tracker's own "will not".
    if tracker.failstep_set:
        return Reason.TRACKER_FAILED_STEP, (f"tracker: {status} (failed step "
                                            f"{tracker.harm_failstep}, exit "
                                            f"{tracker.harm_exitcode})")
    if status == "cannot_harm":
        return Reason.TRACKER_CANNOT_HARM, "tracker: cannot_harm, no failed step"
    if status in ("backlog", "harmonising"):
        return Reason.TRACKER_QUEUED, f"tracker: {status}"
    return Reason.TRACKER_HARMONISED_FILE_MISSING, (
        f"tracker: harmonised, no failed step; harmonised/ HTTP {probe.harm_dir_http}"
        + (" with no data file" if probe.harm_dir_http == 200 else ""))


def classify(study: str, tracker: TrackerRow | None, probe: HarmonisedDirProbe) -> Readability:
    """The decision, pure. The probe decides READABLE; the tracker supplies the reason when
    no file is served."""
    if probe.study != study:
        raise ReadabilityError(f"probe for {probe.study} offered for {study}")
    if tracker is not None and tracker.study != study:
        raise ReadabilityError(f"tracker row for {tracker.study} offered for {study}")

    def out(reason: Reason, detail: str) -> Readability:
        return Readability(study=study, reason=reason, probe=probe, tracker=tracker,
                           detail=detail)

    if is_transport(probe.harm_dir_http):
        return out(Reason.TRANSPORT, f"harmonised/: HTTP {probe.harm_dir_http}")
    if probe.harm_dir_http == 403:
        return out(Reason.ACCESS_DENIED, "harmonised/: HTTP 403")
    if probe.data_file:
        if not probe.tbi_present:
            return out(Reason.FILE_WITHOUT_INDEX, f"{probe.data_file} served without .tbi")
        if probe.range_http is None:
            raise ReadabilityError(f"{study}: data file and index listed but no range read "
                                   f"recorded")
        if is_transport(probe.range_http):
            return out(Reason.TRANSPORT, f"range read: HTTP {probe.range_http}")
        if probe.range_http != 206:
            return out(Reason.RANGE_READ_REFUSED, f"range read: HTTP {probe.range_http}")
        if not probe.bgzf:
            return out(Reason.NOT_BGZF, f"{probe.data_file}: first block is not BGZF")
        return out(Reason.READABLE, probe.data_file)
    if probe.harm_dir_http not in (200, 404, 410):
        return out(Reason.ACCESS_DENIED, f"harmonised/: HTTP {probe.harm_dir_http}")
    reason, detail = _no_file_reason(tracker, probe)
    return out(reason, detail)


@dataclass(frozen=True)
class ReadabilityBatch:
    rows: tuple[Readability, ...]
    universe: tuple[str, ...] = field(repr=False, default=())

    def by_study(self) -> dict[str, Readability]:
        return {r.study: r for r in self.rows}

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.reason.code] = out.get(r.reason.code, 0) + 1
        return dict(sorted(out.items()))

    def require_determined(self) -> None:
        """Raise unless every row is determined: a transport failure must never become an
        unreadable row."""
        und = [r.study for r in self.rows if r.verdict is Verdict.UNDETERMINED]
        if und:
            raise UndeterminedStudies(
                f"{len(und)} studies could not be probed (transport), e.g. {und[:5]}; "
                f"re-probe them before writing anything")


def classify_batch(universe: Sequence[str], *, tracker: Mapping[str, TrackerRow],
                   probes: Mapping[str, HarmonisedDirProbe]) -> ReadabilityBatch:
    """Classify a universe of GCST accessions from probes already taken. An id that is not a
    GCST accession, a study with no probe, and a probe for a study outside the universe each
    fail loud; the rows must equal the universe as SETS."""
    ids = list(dict.fromkeys(universe))
    bad = [s for s in ids if not GCST_ID.match(s)]
    if bad:
        raise ReadabilityError(f"not GCST accessions: {bad[:5]}")
    missing = sorted(set(ids) - set(probes))
    if missing:
        raise ReadabilityError(f"{len(missing)} studies have no probe, e.g. {missing[:5]}")
    stray = sorted(set(probes) - set(ids))
    if stray:
        raise ReadabilityError(f"{len(stray)} probes name studies outside the universe, "
                               f"e.g. {stray[:5]}")
    rows = tuple(classify(s, tracker.get(s), probes[s]) for s in ids)
    if {r.study for r in rows} != set(ids) or len(rows) != len(ids):
        raise ReadabilityError("the readability rows do not partition the universe")
    return ReadabilityBatch(rows=rows, universe=tuple(ids))


def assess(universe: Sequence[str], *, tracker: Mapping[str, TrackerRow], workers: int = 4,
           ftp_base: str = DEFAULT_FTP_BASE, timeout_s: float = 60.0,
           transport_retries: int = 2, backoff_s: float = 30.0,
           sleep: Callable[[float], None] = time.sleep,
           log=sys.stderr) -> ReadabilityBatch:  # pragma: no cover - network
    """Probe and classify a universe. Polite: `workers` concurrent sessions (default 4), and
    the studies whose probe hit a transport failure are re-probed up to `transport_retries`
    more times after a growing back-off."""
    import requests

    tl = threading.local()
    lock = threading.Lock()
    t0 = time.monotonic()

    def run(studies: list[str], label: str) -> dict[str, HarmonisedDirProbe]:
        done = [0]
        step = max(1, min(200, len(studies) // 20 or 1))

        def one(s: str) -> HarmonisedDirProbe:
            if not hasattr(tl, "s"):
                tl.s = requests.Session()
            p = probe_study(s, session=tl.s, ftp_base=ftp_base, timeout_s=timeout_s)
            with lock:
                done[0] += 1
                if done[0] % step == 0 or done[0] == len(studies):
                    print(f"  [{time.monotonic() - t0:6.0f}s] {label}: probed "
                          f"{done[0]}/{len(studies)}", file=log, flush=True)
            return p

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return {p.study: p for p in ex.map(one, studies)}

    ids = list(dict.fromkeys(universe))
    probes = run(ids, "readability")
    for attempt in range(1, transport_retries + 1):
        retry = [s for s, p in probes.items()
                 if classify(s, tracker.get(s), p).verdict is Verdict.UNDETERMINED]
        if not retry:
            break
        sleep(backoff_s * attempt)
        probes.update(run(retry, f"transport retry {attempt}"))
    return classify_batch(ids, tracker=tracker, probes=probes)


# --- the pinned snapshots (read offline by the catalog build) ----------------------------

def tracker_rows_tsv(rows: Iterable[TrackerRow]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(TRACKER_COLUMNS)
    for r in sorted(rows, key=lambda x: x.study):
        w.writerow([getattr(r, c).replace("\t", " ").replace("\n", " ")
                    for c in TRACKER_COLUMNS])
    return buf.getvalue()


def read_tracker_rows_tsv(text: str) -> dict[str, TrackerRow]:
    rd = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(rd.fieldnames or ()) != TRACKER_COLUMNS:
        raise ReadabilityError(f"tracker extract columns {rd.fieldnames} != {TRACKER_COLUMNS}")
    out: dict[str, TrackerRow] = {}
    for r in rd:
        if r["study"] in out:
            raise ReadabilityError(f"{r['study']} appears twice in the tracker extract")
        out[r["study"]] = TrackerRow(**{c: r[c] for c in TRACKER_COLUMNS})
    return out


def probes_tsv(probes: Iterable[HarmonisedDirProbe]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(PROBE_COLUMNS), delimiter="\t", lineterminator="\n")
    w.writeheader()
    for p in sorted(probes, key=lambda x: x.study):
        w.writerow(_probe_to_row(p))
    return buf.getvalue()


def read_probes_tsv(text: str) -> dict[str, HarmonisedDirProbe]:
    rd = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(rd.fieldnames or ()) != PROBE_COLUMNS:
        raise ReadabilityError(f"probe columns {rd.fieldnames} != {PROBE_COLUMNS}")
    out: dict[str, HarmonisedDirProbe] = {}
    for r in rd:
        p = _probe_from_row(r)
        if p.study in out:
            raise ReadabilityError(f"{p.study} appears twice in the probe snapshot")
        out[p.study] = p
    return out
