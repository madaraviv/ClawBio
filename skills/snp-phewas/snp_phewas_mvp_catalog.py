"""The Million Veteran Program outcome catalog, built from the GWAS Catalog's own records.

MVP's genome-wide PheWAS (Verma A et al., Science 2024;385:eadj1182,
doi:10.1126/science.adj1182, PMID 39024449) is served by the NHGRI-EBI GWAS Catalog as
one study per (trait, population): 7,537 studies on the 2026-10-19 snapshot. The Catalog
record carries no MVP trait id, so everything here is read from the record itself:

  * `disease_trait` -> the trait kind and code. A phecode study ends in `(PheCode 447)`;
    a quantitative study ends in `(..., inv-norm transformed)` and names its summary
    statistic (mean / maximum / minimum of the EHR values, or a single survey value); any
    other binary study is a survey item.
  * `discovery_ancestry` -> the population (EUR / AFR / AMR / EAS), or META for the
    Catalog's multi-population meta-analyses.
  * `initial_sample_size` -> cases and controls (binary) or individuals (quantitative).

Which studies can be read by position is not in the record either. It is decided by the
readability rule in `snp_phewas_readability`: a DIRECT PROBE of the study's `harmonised/`
directory (a data file, its `.tbi`, and a range read that returns a BGZF block) says whether
it is readable, and EBI's harmonization tracker (`Harm_sumstats_status.db`) says why when it
is not. Membership in `harmonised_list.txt` decides nothing: the 2026-10-19 build read that
manifest and called 1,580 studies "not harmonized", and 1,575 of them were range-readable on
2026-10-31 with files dated 2025-05 to 2025-07. Every unreadable study is still a catalog row, its `access_status` the
readability reason code (`tracker_harmonised_file_missing`, `untracked_no_deposit`, ...),
and a lookup returns it as a `source_file_not_readable` row with that reason, never as an
absence.

Two halves, so the committed catalog is reproducible offline:

  `refresh_snapshot(out_dir, tracker_db=...)`  (network, run by hand) writes the source
        snapshots into `catalogs/mvp_source/`: the study records (REST v2; the pinned snapshot is
        kept unless `fetch_studies=True`), the tracker rows for those studies, the raw
        readability probes, and each readable file's metadata (`-meta.yaml`: md5, date,
        assembly).
  `build_rows(...)`  (offline) classifies every study with the readability rule from the pinned
        tracker rows and probes and turns the result into catalog rows, with the write-time
        assertions (readable + not readable == the study list, as SETS).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
from collections import Counter
from typing import Any, Iterable

import snp_phewas_readability as RD
from snp_phewas_crosswalk import normalize_phecode, quant_key  # noqa: F401

PUBMED_ID = 39024449
STUDIES_URL = ("https://www.ebi.ac.uk/gwas/rest/api/v2/studies?pubmed_id=39024449"
               "&page={page}&size=200")
DATA_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
#: The committed catalog a refresh is compared against (readable must stay readable).
PRIOR_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalogs",
                                  "mvp_outcomes.tsv.gz")

# The fields of a REST v2 study record the snapshot keeps (the rest is presentation).
STUDY_FIELDS = ("accession_id", "disease_trait", "initial_sample_size", "discovery_ancestry",
                "efo_traits", "full_summary_stats_available", "full_summary_stats", "cohort",
                "terms_of_license", "pubmed_id")

META_COLUMNS = ("accession", "data_file_name", "data_file_md5sum", "date_metadata_last_modified",
                "genome_assembly", "is_sorted", "is_harmonised", "file_type")

#: The readable status keeps its historical code (`harmonised`): every reader keys on it.
ACCESS_HARMONISED = "harmonised"
#: An unreadable study's `access_status` is the readability reason code. An
#: undetermined (`transport`) study is never written: the build refuses instead.
ACCESS_UNREADABLE: tuple[str, ...] = tuple(
    r.code for r in RD.Reason if r.verdict in (RD.Verdict.CANNOT_EVER, RD.Verdict.NOT_YET))
ACCESS_STATUSES = (ACCESS_HARMONISED, *ACCESS_UNREADABLE)
ACCESS_STATUS_LABELS = {
    ACCESS_HARMONISED: "harmonized by the GWAS Catalog (sorted, position-indexed file)",
    **{code: RD.REASON_LABELS[code] for code in ACCESS_UNREADABLE},
}

KIND_PHECODE = "phecode"
KIND_QUANT = "quantitative"
KIND_SURVEY = "survey_binary"
TRAIT_KIND_LABELS = {
    KIND_PHECODE: "disease (phecode, from the health record)",
    KIND_QUANT: "measurement (lab, vital sign or survey number)",
    KIND_SURVEY: "survey answer",
}

POPULATIONS = ("EUR", "AFR", "AMR", "EAS")
META = "META"
# The Catalog's ancestry wording for MVP's genetically inferred groups.
_ANCESTRY = {
    "European (U.S.)": "EUR",
    "African American or Afro-Caribbean (U.S.)": "AFR",
    "Hispanic or Latin American (U.S.)": "AMR",
    "East Asian (U.S.)": "EAS",
}

_PHECODE_RE = re.compile(r"^(?P<label>.*?)\s*\(PheCode\s+(?P<code>[0-9.]+)\)\s*$")
_QUANT_RE = re.compile(r"^(?P<base>.*?)\s*\((?P<inner>[^()]*transformed)\)\s*$")
_STATS = {"mean": "mean", "maximum": "maximum", "minimum": "minimum",
          "minimm": "minimum"}  # `minimm`: a typo in the Catalog's HDL-C minimum record
QUANT_STAT_WORDS = {"mean": "mean of EHR values", "maximum": "maximum of EHR values",
                    "minimum": "minimum of EHR values", "single": ""}
_N_RE = re.compile(r"^(?P<n>\d[\d,]*)\s+.*?\b(?P<what>cases|controls|individuals)\s*$")

CATALOG_COLUMNS: tuple[str, ...] = (
    "outcome_id", "population", "meta_populations", "trait_text", "trait_kind", "phecode",
    "phecode_display", "quant_base", "quant_qualifier", "quant_stat", "outcome_label",
    "category_native", "efo_ids", "n", "n_cases", "n_controls", "access_status", "filename",
    "filename_tabix", "md5_hex", "harmonised_date",
)


class MvpCatalogError(ValueError):
    """A study record that the catalog rules cannot place; never silently skipped."""


# --- parsing one record -------------------------------------------------------------------

def parse_trait(text: str) -> dict[str, str]:
    """{trait_kind?, phecode, phecode_display, quant_base, quant_qualifier, quant_stat,
    label} from the Catalog's `disease_trait`. The kind of a non-phecode, non-quantitative
    trait is decided by the sample size (binary = survey item), not here."""
    t = " ".join(text.split())
    m = _PHECODE_RE.match(t)
    if m:
        code = m.group("code")
        return {"trait_kind": KIND_PHECODE, "phecode": normalize_phecode(code),
                "phecode_display": code, "quant_base": "", "quant_qualifier": "",
                "quant_stat": "", "label": m.group("label")}
    m = _QUANT_RE.match(t)
    if m:
        toks = [s.strip() for s in m.group("inner").split(",")]
        stat, quals = "single", []
        for tk in toks[:-1]:  # the last token is the transformation
            if tk in _STATS:
                stat = _STATS[tk]
            else:
                quals.append(tk)
        base = m.group("base")
        words = [*quals, QUANT_STAT_WORDS[stat]]
        label = f"{base} ({', '.join(w for w in words if w)}{', ' if any(words) else ''}inverse-normal transformed)"
        return {"trait_kind": KIND_QUANT, "phecode": "", "phecode_display": "",
                "quant_base": base, "quant_qualifier": ", ".join(quals), "quant_stat": stat,
                "label": label}
    return {"trait_kind": "", "phecode": "", "phecode_display": "", "quant_base": "",
            "quant_qualifier": "", "quant_stat": "", "label": t}


def parse_population(discovery_ancestry: list[str]) -> tuple[str, str]:
    """(population, meta_populations) from the Catalog's discovery ancestry entries."""
    pops = []
    for a in discovery_ancestry:
        name = a.split(" ", 1)[1] if a[:1].isdigit() else a
        if name not in _ANCESTRY:
            raise MvpCatalogError(f"unknown MVP discovery ancestry {a!r}")
        pops.append(_ANCESTRY[name])
    if not pops:
        raise MvpCatalogError("a study with no discovery ancestry")
    if len(pops) == 1:
        return pops[0], ""
    return META, "+".join(p for p in POPULATIONS if p in pops)


def parse_sample_size(text: str) -> tuple[int | None, int | None, int]:
    """(n_cases, n_controls, n) summed over the populations of `initial_sample_size`."""
    cases = controls = indiv = 0
    seen = Counter()
    for seg in text.split(", "):  # thousands separators carry no space
        m = _N_RE.match(seg.strip())
        if not m:
            raise MvpCatalogError(f"unparsable sample size segment {seg!r} in {text!r}")
        n = int(m.group("n").replace(",", ""))
        seen[m.group("what")] += 1
        if m.group("what") == "cases":
            cases += n
        elif m.group("what") == "controls":
            controls += n
        else:
            indiv += n
    if seen["individuals"] and (seen["cases"] or seen["controls"]):
        raise MvpCatalogError(f"mixed binary and quantitative sample size {text!r}")
    if seen["individuals"]:
        return None, None, indiv
    if not (seen["cases"] and seen["controls"]):
        raise MvpCatalogError(f"a binary sample size without cases and controls {text!r}")
    return cases, controls, cases + controls


# --- the offline build --------------------------------------------------------------------

def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def read_studies(jsonl_gz: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in gzip.decompress(jsonl_gz).decode("utf-8").splitlines()
            if line.strip()]


def read_tracker_rows(tsv_gz: bytes) -> dict[str, Any]:
    """The pinned harmonization-tracker rows (`Harm_sumstats_status.db`, MVP studies only)."""
    return RD.read_tracker_rows_tsv(gzip.decompress(tsv_gz).decode("utf-8"))


def read_probes(tsv_gz: bytes) -> dict[str, Any]:
    """The pinned readability probes of each study's `harmonised/` directory."""
    return RD.read_probes_tsv(gzip.decompress(tsv_gz).decode("utf-8"))


def read_meta(meta_gz: bytes) -> dict[str, dict[str, str]]:
    rows = csv.DictReader(io.StringIO(gzip.decompress(meta_gz).decode("utf-8")), delimiter="\t")
    return {r["accession"]: r for r in rows}


def classify_studies(studies: list[dict[str, Any]], tracker: dict[str, Any],
                     probes: dict[str, Any]) -> "RD.ReadabilityBatch":
    """The readability rule over the study list. Refuses an undetermined (transport)
    study and an id the rule cannot key on: neither may become a catalog row."""
    accs = [s["accession_id"] for s in studies]
    if len(set(accs)) != len(accs) or not all(accs):
        raise MvpCatalogError("the study list repeats or omits an accession")
    stray = set(tracker) - set(accs)
    if stray:
        raise MvpCatalogError(f"tracker rows for studies not in the list: {sorted(stray)[:5]}")
    try:
        batch = RD.classify_batch(accs, tracker=tracker, probes=probes)
        batch.require_determined()
    except (RD.UndeterminedStudies, RD.ReadabilityError) as exc:
        raise MvpCatalogError(str(exc)) from exc
    return batch


def build_rows(studies: list[dict[str, Any]], tracker: dict[str, Any], probes: dict[str, Any],
               meta: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """One catalog row per study, with the write-time assertions:

    * `outcome_id` (the GCST accession) unique and non-null on every row, and the rows equal
      the study list as SETS; readable + not readable == the study list (the readability
      rule's own partition assertion runs first, and refuses any undetermined study);
    * `filename`, `filename_tabix` and `md5_hex` set exactly when the study is readable,
      the readable file is the one its metadata names, and it is GRCh38 (else the build
      refuses rather than read it at the wrong position);
    * every metadata row names a readable study (a stray row fails loudly).
    """
    batch = classify_studies(studies, tracker, probes)
    by = batch.by_study()
    readable = {acc for acc, r in by.items() if r.readable}
    stray = set(meta) - readable
    if stray:
        raise MvpCatalogError(f"harmonized metadata for studies that are not readable: "
                              f"{sorted(stray)[:5]}")
    rows = []
    for s in sorted(studies, key=lambda x: x["accession_id"]):
        acc = s["accession_id"]
        pop, meta_pops = parse_population(s["discovery_ancestry"])
        n_cases, n_controls, n = parse_sample_size(s["initial_sample_size"])
        tr = parse_trait(s["disease_trait"])
        binary = n_controls is not None
        if tr["trait_kind"] == KIND_QUANT and binary:
            raise MvpCatalogError(f"{acc}: a transformed trait with cases and controls")
        if tr["trait_kind"] == KIND_PHECODE and not binary:
            raise MvpCatalogError(f"{acc}: a phecode trait without cases and controls")
        if not tr["trait_kind"]:
            if not binary:
                raise MvpCatalogError(f"{acc}: an untyped quantitative trait {s['disease_trait']!r}")
            tr["trait_kind"] = KIND_SURVEY
            tr["label"] = f"{tr['label']} (survey answer)"
        if acc in readable:
            access = ACCESS_HARMONISED
            m = meta.get(acc)
            if m is None or not m.get("data_file_md5sum"):
                raise MvpCatalogError(f"{acc}: harmonized but no metadata md5 in the snapshot")
            # `is_sorted` is NOT required: on 2026-10-19, 12 of the 4,442 harmonized MVP
            # files' metadata said false while all 12 carried a .tbi, and tabix refuses to
            # index a file whose positions are out of order, so the flag was stale (on the
            # 2026-10-31 re-fetch all 6,017 say true). The probe's range read is the check.
            if m["genome_assembly"] != "GRCh38":
                raise MvpCatalogError(f"{acc}: harmonized file is {m['genome_assembly']}, "
                                      f"not GRCh38; not readable at a GRCh38 position")
            filename = by[acc].data_path
            if filename.rsplit("/", 1)[-1] != m["data_file_name"]:
                raise MvpCatalogError(f"{acc}: metadata names {m['data_file_name']}, the "
                                      f"directory serves {filename}")
            md5, date = m["data_file_md5sum"], m["date_metadata_last_modified"]
        else:
            access = by[acc].reason.code
            filename, md5, date = "", "", ""
        efo = s.get("efo_traits") or []
        rows.append({
            "outcome_id": acc, "population": pop, "meta_populations": meta_pops,
            "trait_text": " ".join(s["disease_trait"].split()),
            "trait_kind": tr["trait_kind"], "phecode": tr["phecode"],
            "phecode_display": tr["phecode_display"], "quant_base": tr["quant_base"],
            "quant_qualifier": tr["quant_qualifier"], "quant_stat": tr["quant_stat"],
            "outcome_label": tr["label"],
            "category_native": (f"PheCode {tr['phecode']}" if tr["trait_kind"] == KIND_PHECODE
                                else TRAIT_KIND_LABELS[tr["trait_kind"]]),
            "efo_ids": ";".join(e["efo_id"] for e in efo),
            "n": str(n), "n_cases": "" if n_cases is None else str(n_cases),
            "n_controls": "" if n_controls is None else str(n_controls),
            "access_status": access, "filename": filename,
            "filename_tabix": f"{filename}.tbi" if filename else "",
            "md5_hex": md5, "harmonised_date": date,
        })
    assert_catalog(rows, set(by))
    n_read = sum(1 for r in rows if r["access_status"] == ACCESS_HARMONISED)
    if n_read != len(readable):
        raise MvpCatalogError(f"{n_read} readable rows written for {len(readable)} readable "
                              f"studies")
    return rows


def assert_catalog(rows: list[dict[str, str]], study_ids: set[str]) -> None:
    """Raised (not `assert`ed) write-time invariants of the MVP catalog."""
    ids = [r["outcome_id"] for r in rows]
    if not all(ids) or len(set(ids)) != len(ids) or set(ids) != study_ids:
        raise MvpCatalogError(
            f"catalog rows do not partition the {len(study_ids)} studies "
            f"({len(set(ids))} distinct of {len(ids)}; {len(study_ids - set(ids))} missing)")
    for r in rows:
        if r["access_status"] not in ACCESS_STATUSES:
            raise MvpCatalogError(f"{r['outcome_id']}: unknown access status {r['access_status']!r}")
        readable = r["access_status"] == ACCESS_HARMONISED
        for col in ("filename", "filename_tabix", "md5_hex"):
            if bool(r[col]) != readable:
                raise MvpCatalogError(f"{r['outcome_id']}: {col} must be set exactly when "
                                      f"harmonised (status {r['access_status']})")
        if r["trait_kind"] not in TRAIT_KIND_LABELS:
            raise MvpCatalogError(f"{r['outcome_id']}: unknown trait kind {r['trait_kind']!r}")


def readable_in_catalog(path: str) -> set[str]:
    """The studies a written MVP catalog marks readable (`access_status` harmonised)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {r["outcome_id"] for r in csv.DictReader(fh, delimiter="\t")
                if r["access_status"] == ACCESS_HARMONISED}


def assert_no_readable_lost(prior_readable: Iterable[str], now_readable: Iterable[str], *,
                            allow_lost: Iterable[str] = ()) -> None:
    """A study readable in the prior catalog must stay readable, unless named in
    `allow_lost`. An outage that answers 404 for served files (EBI, 2026-10-14) reads as
    unreadable through the probe just as it did through the manifest; this is the check
    that stops it becoming the catalog. `allow_lost` names the studies a human confirmed
    gone; a name in it that is still readable, or was never readable, is an error too, so
    the list cannot go stale silently."""
    prior, now, allowed = set(prior_readable), set(now_readable), set(allow_lost)
    stale = sorted(allowed - (prior - now))
    if stale:
        raise MvpCatalogError(f"{len(stale)} studies in allow_lost were not lost "
                              f"(still readable, or never readable), e.g. {stale[:5]}")
    lost = sorted(prior - now - allowed)
    if lost:
        raise MvpCatalogError(
            f"{len(lost)} studies readable in the prior catalog are unreadable now, e.g. "
            f"{lost[:5]}; re-probe (an outage looks like this), or name them in allow_lost "
            f"once confirmed gone at the source")


def write_catalog(rows: Iterable[dict[str, str]], path: str) -> int:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CATALOG_COLUMNS), delimiter="\t",
                       lineterminator="\n")
    w.writeheader()
    n = 0
    for r in rows:
        w.writerow(r)
        n += 1
    with open(path, "wb") as fh:
        # filename="" and mtime=0: the gzip header carries neither the output path nor the
        # time, so a rebuild from the same snapshots is byte-identical wherever it is written
        with gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(buf.getvalue().encode("utf-8"))
    return n


# --- the network snapshot (run by hand) ---------------------------------------------------

def _gz(text: str) -> bytes:
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    return bio.getvalue()


SNAPSHOT_NAMES = {
    "studies": "gwas_catalog_studies_pmid39024449.jsonl.gz",
    "tracker": "harmonisation_tracker_mvp.tsv.gz",
    "probes": "harmonised_probe_mvp.tsv.gz",
    "meta": "harmonised_meta_mvp.tsv.gz",
}


def fetch_study_list(sess, *, log=print) -> list[dict[str, Any]]:  # pragma: no cover - network
    """The MVP study records from the GWAS Catalog REST v2, sorted by accession."""
    from concurrent.futures import ThreadPoolExecutor

    from snp_phewas_tabix import get_with_retry

    def page_of(page: int) -> dict[str, Any]:
        r = get_with_retry(sess, STUDIES_URL.format(page=page), attempts=5, backoff_s=5.0,
                           timeout=300)
        r.raise_for_status()
        return r.json()

    first = page_of(0)
    total_pages, total = first["page"]["totalPages"], first["page"]["totalElements"]
    pages = [first]
    with ThreadPoolExecutor(max_workers=4) as ex:  # one page of 200 takes ~20 s to serve
        for i, d in enumerate(ex.map(page_of, range(1, total_pages)), 2):
            pages.append(d)
            log(f"study list page {i}/{total_pages}")
    studies = [{k: s.get(k) for k in STUDY_FIELDS}
               for d in pages for s in d["_embedded"]["studies"]]
    if len(studies) != total or len({s["accession_id"] for s in studies}) != total:
        raise MvpCatalogError(f"read {len(studies)} study records of {total} "
                              f"({len({s['accession_id'] for s in studies})} distinct)")
    studies.sort(key=lambda s: s["accession_id"])
    return studies


def refresh_snapshot(out_dir: str, *, tracker_db: str, probes_tsv: str | None = None,
                     fetch_studies: bool = False, workers: int = 4,
                     prior_catalog: str | None = None, allow_lost: Iterable[str] = (),
                     log=print) -> dict[str, Any]:  # pragma: no cover - network
    """Write the source snapshots into `out_dir`; returns their md5s and provenance for the
    pin. `tracker_db` is a local copy of `Harm_sumstats_status.db`. The study list is the
    pinned one in `out_dir` unless `fetch_studies`. `probes_tsv` reuses probes already taken
    over the same studies (`snp_phewas_readability.probes_tsv`); otherwise the studies are
    probed here (polite: `workers` sessions, transport retried). An undetermined
    study stops the refresh: it must never be written as unreadable."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests
    import yaml

    sess = requests.Session()
    os.makedirs(out_dir, exist_ok=True)
    studies_path = os.path.join(out_dir, SNAPSHOT_NAMES["studies"])
    if fetch_studies:
        studies = fetch_study_list(sess, log=log)
    else:
        with open(studies_path, "rb") as fh:
            studies = read_studies(fh.read())
    accs = [s["accession_id"] for s in studies]
    db = RD.read_tracker_db(tracker_db)
    db_sha = db.sha256
    tracker = {a: db.rows_by_study[a] for a in accs if a in db.rows_by_study}
    if probes_tsv:
        with open(probes_tsv, encoding="utf-8") as fh:
            taken = RD.read_probes_tsv(fh.read())
        probes = {a: taken[a] for a in accs if a in taken}
        batch = RD.classify_batch(accs, tracker=tracker, probes=probes)
    else:
        batch = RD.assess(accs, tracker=tracker, workers=workers)
    batch.require_determined()
    by = batch.by_study()
    paths = {a: r.data_path for a, r in by.items() if r.readable}
    assert_no_readable_lost(readable_in_catalog(prior_catalog or PRIOR_CATALOG_PATH), paths,
                            allow_lost=allow_lost)

    def meta_of(acc: str) -> dict[str, str]:
        url = DATA_BASE_URL + paths[acc] + "-meta.yaml"
        for attempt in range(4):
            try:
                resp = sess.get(url, timeout=60)
                if resp.status_code == 200:
                    m = yaml.safe_load(resp.text)
                    return {"accession": acc, "data_file_name": m.get("data_file_name", ""),
                            "data_file_md5sum": m.get("data_file_md5sum", ""),
                            "date_metadata_last_modified": str(m.get("date_metadata_last_modified", "")),
                            "genome_assembly": str(m.get("genome_assembly", "")),
                            "is_sorted": str(m.get("is_sorted", "")).lower(),
                            "is_harmonised": str(m.get("is_harmonised", "")).lower(),
                            "file_type": str(m.get("file_type", ""))}
            except requests.RequestException:
                pass  # a transient error: retry, then raise below
            time.sleep(5.0 * (attempt + 1))
        raise MvpCatalogError(f"{acc}: harmonized metadata could not be read from {url}")

    metas: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, m in enumerate(ex.map(meta_of, sorted(paths)), 1):
            metas.append(m)
            if i % 250 == 0 or i == len(paths):
                log(f"harmonized metadata {i}/{len(paths)}")
    blobs = {
        "studies": _gz("".join(json.dumps(s, sort_keys=True) + "\n" for s in studies)),
        "tracker": _gz(RD.tracker_rows_tsv(tracker.values())),
        "probes": _gz(RD.probes_tsv(r.probe for r in batch.rows)),
        "meta": _gz("".join(["\t".join(META_COLUMNS) + "\n"]
                            + ["\t".join(m[c] for c in META_COLUMNS) + "\n" for m in metas])),
    }
    probed = sorted(r.probe.probed_utc for r in batch.rows)
    out: dict[str, Any] = {"tracker_sha256": db_sha, "tracker_last_run": db.last_run,
                           "probed_utc_first": probed[0], "probed_utc_last": probed[-1],
                           "readability_version": RD.READABILITY_VERSION,
                           "n_studies": len(studies), "n_readable": len(paths),
                           "reason_counts": batch.counts()}
    for k, blob in blobs.items():
        with open(os.path.join(out_dir, SNAPSHOT_NAMES[k]), "wb") as fh:
            fh.write(blob)
        out[f"{SNAPSHOT_NAMES[k]}_md5"] = _md5(blob)
    return out
