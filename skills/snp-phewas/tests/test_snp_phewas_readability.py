"""The MVP readability rule: the probe decides, the tracker explains, the manifest never
decides. Every reason is constructed here as the input that produces it, a transport failure
is shown never to read as unreadable, and the batch refuses a dropped or an invented study."""

from __future__ import annotations

import dataclasses

import pytest

import snp_phewas_readability as R
from snp_phewas_readability import TrackerRow

S = "GCST90475089"
BGZF_HEAD = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00\x1b\x00"


def trow(study=S, status="harmonised", failstep="NA", exitcode="NA", reason="NA"):
    return TrackerRow(study=study, harm_status=status, harm_failstep=failstep,
                      harm_exitcode=exitcode, harm_failed_reason=reason,
                      latest_harm_start_date="2025-05-02", harm_account="ot_gwas",
                      raw_file_type="gwas-ssf", raw_genome_build="GRCh38",
                      raw_coordinate_system="1-based", effect_size_type="beta",
                      harm_drop_rate="0.01", liftover_drop_rate="NA")


def probe(study=S, **kw):
    base = dict(study=study, probed_utc="2026-10-31T16:00:00Z", harm_dir_http=200,
                data_file=f"{study}.h.tsv.gz", data_files=(f"{study}.h.tsv.gz",),
                tbi_present=True, range_http=206, bgzf=True)
    base.update(kw)
    return R.HarmonisedDirProbe(**base)


NO_FILE = dict(harm_dir_http=404, data_file="", data_files=(), tbi_present=False,
               range_http=None, bgzf=None, study_dir_http=200,
               study_dir_files=(f"{S}.tsv.gz",))


# --- every reason, constructed ----------------------------------------------------------

CASES = [
    ("readable", trow(), probe(), R.Reason.READABLE),
    ("readable while the tracker lags", trow(status="backlog"), probe(), R.Reason.READABLE),
    ("cannot_harm, no file", trow(status="cannot_harm"), probe(**NO_FILE),
     R.Reason.TRACKER_CANNOT_HARM),
    ("cannot_harm with a failed step is a failed step, re-checked every run",
     trow(status="cannot_harm", failstep="majordirectionmaptobuild", exitcode="137"),
     probe(**NO_FILE), R.Reason.TRACKER_FAILED_STEP),
    ("failed step, retained harmonised", trow(failstep="majordirectiontenpercentcounts",
                                              exitcode="2"), probe(**NO_FILE),
     R.Reason.TRACKER_FAILED_STEP),
    ("failed step, harmonising", trow(status="harmonising", failstep="majordirectionmaptobuild",
                                      exitcode="137"), probe(**NO_FILE),
     R.Reason.TRACKER_FAILED_STEP),
    ("backlog, no file", trow(status="backlog"), probe(**NO_FILE), R.Reason.TRACKER_QUEUED),
    ("harmonising, no file", trow(status="harmonising"), probe(**NO_FILE),
     R.Reason.TRACKER_QUEUED),
    ("tracker harmonised, dir 404", trow(), probe(**NO_FILE),
     R.Reason.TRACKER_HARMONISED_FILE_MISSING),
    ("tracker harmonised, dir lists metadata only", trow(),
     probe(**{**NO_FILE, "harm_dir_http": 200}), R.Reason.TRACKER_HARMONISED_FILE_MISSING),
    ("untracked, raw upload", None, probe(**NO_FILE), R.Reason.UNTRACKED_UNHARMONISED),
    ("untracked, no study dir", None,
     probe(**{**NO_FILE, "study_dir_http": 404, "study_dir_files": ()}),
     R.Reason.UNTRACKED_NO_DEPOSIT),
    ("file, no index", trow(), probe(tbi_present=False, range_http=None, bgzf=None),
     R.Reason.FILE_WITHOUT_INDEX),
    ("range refused", trow(), probe(range_http=200, bgzf=None), R.Reason.RANGE_READ_REFUSED),
    ("not bgzf", trow(), probe(bgzf=False), R.Reason.NOT_BGZF),
    ("dir 403", trow(), probe(**{**NO_FILE, "harm_dir_http": 403, "study_dir_http": None,
                                 "study_dir_files": ()}), R.Reason.ACCESS_DENIED),
    ("untracked, study dir 403", None,
     probe(**{**NO_FILE, "study_dir_http": 403, "study_dir_files": ()}),
     R.Reason.ACCESS_DENIED),
    ("dir unreachable", trow(), probe(**{**NO_FILE, "harm_dir_http": 0, "study_dir_http": None,
                                         "study_dir_files": ()}), R.Reason.TRANSPORT),
    ("dir 503", trow(status="cannot_harm"),
     probe(**{**NO_FILE, "harm_dir_http": 503, "study_dir_http": None, "study_dir_files": ()}),
     R.Reason.TRANSPORT),
    ("range read timed out", trow(), probe(range_http=0, bgzf=None), R.Reason.TRANSPORT),
    ("range read throttled", trow(), probe(range_http=429, bgzf=None), R.Reason.TRANSPORT),
    ("untracked, study dir unreachable", None,
     probe(**{**NO_FILE, "study_dir_http": 0, "study_dir_files": ()}), R.Reason.TRANSPORT),
]


@pytest.mark.parametrize("name,tracker,pr,expected", CASES, ids=[c[0] for c in CASES])
def test_each_reason_is_produced_by_its_input(name, tracker, pr, expected):
    got = R.classify(S, tracker, pr)
    assert got.reason is expected, (name, got.detail)


def test_every_reason_is_constructed_by_some_case():
    assert {c[3] for c in CASES} == set(R.Reason)


def test_only_cannot_harm_is_permanent():
    perm = {r for r in R.Reason if r.verdict is R.Verdict.CANNOT_EVER}
    assert perm == {R.Reason.TRACKER_CANNOT_HARM}


# --- transport is never unavailable -----------------------------------------------------

@pytest.mark.parametrize("code", [0, 408, 429, 500, 502, 503, 504])
def test_a_transport_answer_is_undetermined_whatever_the_tracker_says(code):
    for tracker in (None, trow(), trow(status="cannot_harm"), trow(status="backlog")):
        got = R.classify(S, tracker, probe(**{**NO_FILE, "harm_dir_http": code,
                                              "study_dir_http": None, "study_dir_files": ()}))
        assert got.reason is R.Reason.TRANSPORT
        assert got.verdict is R.Verdict.UNDETERMINED
        assert not got.readable


def test_require_determined_refuses_a_batch_with_a_transport_row():
    probes = {S: probe(**{**NO_FILE, "harm_dir_http": 0, "study_dir_http": None,
                          "study_dir_files": ()})}
    batch = R.classify_batch([S], tracker={S: trow()}, probes=probes)
    with pytest.raises(R.UndeterminedStudies):
        batch.require_determined()


# --- loud failures ----------------------------------------------------------------------

def test_a_fifth_tracker_status_stops_the_rule():
    with pytest.raises(R.ReadabilityError):
        R.classify(S, trow(status="failed"), probe(**NO_FILE))


def test_a_listed_file_with_index_but_no_range_read_is_an_incomplete_probe():
    with pytest.raises(ValueError, match="no range read"):
        R.classify(S, trow(), probe(range_http=None, bgzf=None))


def test_a_probe_for_another_study_is_refused():
    with pytest.raises(ValueError):
        R.classify(S, trow(), probe(study="GCST1"))


# --- the partition ----------------------------------------------------------------------

def test_a_batch_refuses_a_dropped_and_an_invented_study():
    other = "GCST90475092"
    with pytest.raises(R.ReadabilityError, match="have no probe"):
        R.classify_batch([S, other], tracker={}, probes={S: probe()})
    with pytest.raises(R.ReadabilityError, match="outside the universe"):
        R.classify_batch([S], tracker={}, probes={S: probe(), other: probe(study=other)})
    with pytest.raises(R.ReadabilityError, match="not GCST"):
        R.classify_batch(["rs123"], tracker={}, probes={})


def test_the_bucket_directory_holds_the_study():
    assert R.relative_harmonised_dir("GCST90475089") == \
        "GCST90475001-GCST90476000/GCST90475089/harmonised"
    assert R.relative_harmonised_dir("GCST90476000") == \
        "GCST90475001-GCST90476000/GCST90476000/harmonised"


# --- persisted snapshots round-trip ------------------------------------------------------

def test_probe_and_tracker_snapshots_round_trip():
    ps = [c[2] for c in CASES]
    uniq = {p.study + str(i): dataclasses.replace(p, study=f"GCST{9000 + i}")
            for i, p in enumerate(ps)}
    back = R.read_probes_tsv(R.probes_tsv(uniq.values()))
    assert set(back.values()) == set(uniq.values())
    rows = [trow(), trow(study="GCST2", status="cannot_harm", reason="x\ty")]
    tb = R.read_tracker_rows_tsv(R.tracker_rows_tsv(rows))
    assert tb[S] == rows[0] and tb["GCST2"].harm_failed_reason == "x y"


def test_a_duplicated_probe_row_is_refused():
    text = R.probes_tsv([probe()])
    text += text.splitlines()[1] + "\n"
    with pytest.raises(ValueError, match="twice"):
        R.read_probes_tsv(text)


def test_every_reason_has_a_plain_label_and_no_internal_jargon():
    for r in R.Reason:
        assert r.label and "_" not in r.label
    assert set(R.REASON_LABELS) == {r.code for r in R.Reason}
