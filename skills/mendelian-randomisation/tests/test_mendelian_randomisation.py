"""Tests for mendelian-randomisation skill.

Validates MR estimators (IVW, Egger, weighted median/mode), sensitivity
tests, instrument diagnostics, and end-to-end demo mode.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR))

from mendelian_randomisation import (
    Instrument,
    MREstimate,
    cochran_q,
    egger_weighted_dispersion,
    compute_i_squared_gx,
    generate_report,
    ivw,
    leave_one_out,
    load_demo_instruments,
    mr_egger,
    run_pipeline,
    run_sensitivity,
    steiger_test,
    weighted_median,
    weighted_mode,
)


@pytest.fixture
def demo_instruments():
    instruments, _, _ = load_demo_instruments()
    return instruments


@pytest.fixture
def known_causal():
    """Create instruments with a known causal effect of 0.5."""
    rng = np.random.default_rng(42)
    instruments = []
    for i in range(20):
        bx = rng.uniform(0.03, 0.07) * rng.choice([1, -1])
        se_x = abs(bx) / rng.uniform(5, 10)
        by = bx * 0.5 + rng.normal(0, se_x * 0.2)
        se_y = se_x * rng.uniform(1.5, 2.0)
        instruments.append(Instrument(
            snp=f"rs{i}", effect_allele="A", other_allele="G",
            eaf=rng.uniform(0.2, 0.8),
            beta_exposure=float(bx), se_exposure=float(se_x),
            pval_exposure=1e-10,
            beta_outcome=float(by), se_outcome=float(se_y),
            pval_outcome=0.01,
            f_statistic=float((bx / se_x) ** 2),
        ))
    return instruments


# ---------------------------------------------------------------------------
# Unit tests — estimators
# ---------------------------------------------------------------------------
class TestIVW:
    def test_returns_mr_estimate(self, demo_instruments):
        est = ivw(demo_instruments)
        assert isinstance(est, MREstimate)
        assert est.method == "IVW"
        assert est.n_snps == 30

    def test_recovers_true_effect(self, demo_instruments):
        est = ivw(demo_instruments)
        assert 0.3 < est.estimate < 0.9

    def test_significant_pvalue(self, demo_instruments):
        est = ivw(demo_instruments)
        assert est.pvalue < 0.05

    def test_ci_contains_estimate(self, demo_instruments):
        est = ivw(demo_instruments)
        assert est.ci_lower < est.estimate < est.ci_upper

    def test_known_causal_effect(self, known_causal):
        est = ivw(known_causal)
        assert abs(est.estimate - 0.5) < 0.3


class TestMREgger:
    def test_returns_estimate_and_intercept(self, demo_instruments):
        est, intercept, se_int, p_int = mr_egger(demo_instruments)
        assert est.method == "MR-Egger"
        assert isinstance(intercept, float)
        assert isinstance(p_int, float)

    def test_intercept_near_zero_when_no_pleiotropy(self, demo_instruments):
        _, intercept, _, p_int = mr_egger(demo_instruments)
        assert abs(intercept) < 0.1
        assert p_int > 0.05


class TestWeightedMedian:
    def test_returns_estimate(self, demo_instruments):
        est = weighted_median(demo_instruments)
        assert est.method == "Weighted Median"
        assert est.n_snps == 30

    def test_consistent_with_ivw(self, demo_instruments):
        ivw_est = ivw(demo_instruments)
        wm_est = weighted_median(demo_instruments)
        assert abs(ivw_est.estimate - wm_est.estimate) < 0.3


class TestWeightedMode:
    def test_returns_estimate(self, demo_instruments):
        est = weighted_mode(demo_instruments)
        assert est.method == "Weighted Mode"


# ---------------------------------------------------------------------------
# Unit tests — sensitivity
# ---------------------------------------------------------------------------
class TestSensitivity:
    def test_cochran_q_no_heterogeneity(self, demo_instruments):
        ivw_est = ivw(demo_instruments)
        q, p, df = cochran_q(demo_instruments, ivw_est)
        assert df == 29
        assert p > 0.05

    def test_steiger_correct_direction(self, demo_instruments):
        correct, p = steiger_test(demo_instruments)
        assert correct is True

    def test_i_squared_gx(self, demo_instruments):
        i2 = compute_i_squared_gx(demo_instruments)
        assert 0 <= i2 <= 1

    def test_leave_one_out_length(self, demo_instruments):
        loo = leave_one_out(demo_instruments)
        assert len(loo) == 30

    def test_run_sensitivity_returns_all_fields(self, demo_instruments):
        ivw_est = ivw(demo_instruments)
        sens = run_sensitivity(demo_instruments, ivw_est)
        assert sens.mean_f_statistic > 0
        assert sens.cochran_q >= 0


# ---------------------------------------------------------------------------
# Unit tests — instrument properties
# ---------------------------------------------------------------------------
class TestInstrumentProperties:
    def test_palindromic_AT(self):
        i = Instrument("rs1", "A", "T", 0.5, 0.05, 0.01, 1e-10, 0.03, 0.02, 0.1, 25.0)
        assert i.is_palindromic is True
        assert i.palindromic_ambiguous is True

    def test_palindromic_not_ambiguous(self):
        i = Instrument("rs1", "A", "T", 0.2, 0.05, 0.01, 1e-10, 0.03, 0.02, 0.1, 25.0)
        assert i.is_palindromic is True
        assert i.palindromic_ambiguous is False

    def test_not_palindromic(self):
        i = Instrument("rs1", "A", "C", 0.5, 0.05, 0.01, 1e-10, 0.03, 0.02, 0.1, 25.0)
        assert i.is_palindromic is False

    def test_weak_instrument(self):
        i = Instrument("rs1", "A", "G", 0.3, 0.05, 0.02, 1e-10, 0.03, 0.02, 0.1, 6.0)
        assert i.weak_instrument is True

    def test_strong_instrument(self):
        i = Instrument("rs1", "A", "G", 0.3, 0.05, 0.005, 1e-10, 0.03, 0.02, 0.1, 100.0)
        assert i.weak_instrument is False


# ---------------------------------------------------------------------------
# Integration tests — demo pipeline
# ---------------------------------------------------------------------------
class TestDemoPipeline:
    def test_demo_end_to_end(self, tmp_path):
        instruments, exposure, outcome = load_demo_instruments()
        summary = run_pipeline(instruments, exposure, outcome, tmp_path, demo=True)

        assert summary["n_instruments"] == 30
        assert 0.3 < summary["ivw_estimate"] < 0.9
        assert summary["ivw_pvalue"] < 0.05
        assert summary["n_weak"] == 0

        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "tables" / "mr_results.tsv").exists()
        assert (tmp_path / "tables" / "sensitivity.tsv").exists()
        assert (tmp_path / "tables" / "harmonised_instruments.tsv").exists()
        assert (tmp_path / "figures" / "scatter.png").exists()
        assert (tmp_path / "figures" / "forest.png").exists()
        assert (tmp_path / "figures" / "funnel.png").exists()
        assert (tmp_path / "figures" / "leave_one_out.png").exists()

    def test_demo_report_content(self, tmp_path):
        instruments, exposure, outcome = load_demo_instruments()
        run_pipeline(instruments, exposure, outcome, tmp_path, demo=True)

        report = (tmp_path / "report.md").read_text()
        assert "Mendelian Randomisation" in report
        assert "BMI" in report
        assert "T2D" in report
        assert "ClawBio is a research" in report
        assert "IVW" in report

        result = json.loads((tmp_path / "result.json").read_text())
        assert result["exposure"] == "Body mass index (BMI)"
        assert len(result["estimates"]) == 4
        assert result["sensitivity"]["n_weak"] == 0

    def test_all_methods_consistent(self, tmp_path):
        instruments, exposure, outcome = load_demo_instruments()
        run_pipeline(instruments, exposure, outcome, tmp_path, demo=True)

        result = json.loads((tmp_path / "result.json").read_text())
        estimates = {e["method"]: e["estimate"] for e in result["estimates"]}
        ivw_est = estimates["IVW"]
        for method, est in estimates.items():
            assert abs(est - ivw_est) < 0.2, f"{method} ({est:.3f}) diverges from IVW ({ivw_est:.3f})"


# ---------------------------------------------------------------------------
# MR-Egger applicability: two independent conditions, and a trap in the obvious
# test suite for them
# ---------------------------------------------------------------------------
#
# MR-Egger fits a slope AND an intercept. Two things have to hold, and neither implies
# the other:
#
#   STATISTICAL  n >= 3, or there is no residual degree of freedom at all.
#   NUMERICAL    at least two distinct beta_exposure values, which is not an n < 3
#                problem -- it is reachable at any n from a merge that duplicates one
#                SNP across rows, tissues or proxies.
#
# THE TRAP, and why the cases below are shaped the way they are: the n < 3 return comes
# FIRST, so every n=1 and n=2 input short-circuits before it ever reaches the
# conditioning test. A suite made only of n=1 and n=2 cases therefore passes whether or
# not the centered denominator and the relative threshold work at all -- which is the
# "a gate never observed failing is not known to work" failure, committed by the very
# tests written to prevent it. The n>=3 near-collinear case is the only input that
# reaches the second guard, so it is the load-bearing one here.


def _egger_input(bxs, by_scale=0.5, se_y=0.02):
    """Instruments with the given exposure effects and a clean linear outcome."""
    return [
        Instrument(snp=f"rs{k}", effect_allele="A", other_allele="G", eaf=0.3,
                   beta_exposure=float(bx), se_exposure=abs(float(bx)) / 8 or 0.01,
                   pval_exposure=1e-10,
                   beta_outcome=float(bx) * by_scale, se_outcome=se_y,
                   pval_outcome=0.01, f_statistic=64.0)
        for k, bx in enumerate(bxs)
    ]


@pytest.mark.parametrize("n", [1, 2])
def test_egger_below_three_instruments_is_not_applicable(n):
    """The statistical condition. Previously n=1 was a three-way coin flip -- guard,
    crash, or a finite slope with a median SE of 2.1e+07 -- and n=2 exited 0 while
    writing an infinite standard error."""
    est, intercept, int_se, int_p = mr_egger(_egger_input([0.2, 0.5][:n]))

    assert est.applicable is False
    assert est.n_snps == n
    assert "at least 3" in est.reason
    assert math.isnan(est.estimate) and math.isnan(est.se)
    # the intercept (the pleiotropy test) is undefined on exactly the same inputs
    assert math.isnan(intercept) and math.isnan(int_se) and math.isnan(int_p)


def test_egger_with_near_identical_exposure_effects_is_not_applicable():
    """The NUMERICAL condition, and the only input in this file that reaches it.

    n=5, so the instrument-count return does not fire; the exposure effects agree to
    ~1e-11 relative, so the slope is unidentified. The old code returned a number here:
    the same shape gave slope=3960 with se=2.64e+06 and no warning of any kind.
    """
    base = 0.4
    est, intercept, _, _ = mr_egger(
        _egger_input([base + k * 1e-11 for k in range(5)]))

    assert est.applicable is False
    assert est.n_snps == 5
    assert "too close to identical" in est.reason
    assert math.isnan(est.estimate)
    assert math.isnan(intercept)


def test_the_centered_dispersion_is_non_negative_where_the_expanded_form_is_not():
    """The sign, which is the root cause rather than the threshold.

    Tested on the denominator DIRECTLY rather than through `mr_egger`, and the reason
    is worth stating: once the conditioning check is in place it refuses these inputs
    anyway, so the public path behaves identically with either form. Centering is a
    STRUCTURAL property, not a behavioural one, and a test routed through the public
    API would pass without it -- which is the failure this whole file is about.

    The construction matters. Equally-spaced exposure effects with equal weights do NOT
    reproduce the defect: the cancellation stays non-negative by symmetry. It needs
    effects that are near-identical but irregularly spaced, with the varying weights
    real standard errors produce. Over the 200 draws below the expanded form goes
    negative for roughly a third of them, matching the ~35% measured at n=5 across
    4,000 draws; the centered form was non-negative in all 64,000 draws of that sweep.
    """
    rng = random.Random(20260829)
    saw_expanded_go_negative = 0
    for _ in range(200):
        base = rng.uniform(0.05, 0.9)
        bx = np.array([base * (1 + rng.uniform(-1e-12, 1e-12)) for _ in range(5)])
        se = np.array([rng.uniform(0.005, 0.05) for _ in range(5)])
        w = 1.0 / se ** 2

        centered = egger_weighted_dispersion(w, bx)
        assert centered >= 0.0, f"centered dispersion went negative: {centered!r}"
        assert math.isfinite(centered)

        expanded = np.sum(w) * np.sum(w * bx ** 2) - np.sum(w * bx) ** 2
        if expanded < 0:
            saw_expanded_go_negative += 1

    # exactly zero when the exposure effects are identical, which is the algebraic
    # statement that the slope is unidentified
    assert egger_weighted_dispersion(np.full(4, 2500.0), np.full(4, 0.4)) == 0.0

    # and the failure the centering removes is observed on this very sweep, not
    # asserted from a story about it
    assert saw_expanded_go_negative > 20, (
        f"the expanded form went negative only {saw_expanded_go_negative} times in 200 "
        "draws, so this test is not observing the defect it exists for")


@pytest.mark.parametrize("n", [3, 5])
def test_a_well_conditioned_fit_still_produces_an_estimate(n):
    """The false-positive direction: neither guard may fire on ordinary data."""
    est, intercept, int_se, int_p = mr_egger(
        _egger_input([0.1 * (k + 1) for k in range(n)]))

    assert est.applicable is True
    assert est.reason == ""
    assert math.isfinite(est.estimate) and math.isfinite(est.se)
    assert est.se < 1.0                      # not the inflated-SE branch
    assert est.estimate == pytest.approx(0.5, abs=0.05)
    assert all(math.isfinite(v) for v in (intercept, int_se, int_p))


def test_a_two_instrument_run_writes_parseable_json_and_a_report_that_says_why(tmp_path):
    """End to end, because every symptom of this bug was in the ARTIFACTS.

    A two-instrument run used to exit 0 having written `"se": Infinity` into
    result.json -- which RFC 8259 has no token for, so a strict parser rejects the whole
    document -- and a report.md asserting the result was "robust", because the
    consistency check compared point estimates and never looked at the standard errors.
    """
    insts = _egger_input([0.2, 0.5])
    out = tmp_path / "run"
    out.mkdir()
    run_pipeline(insts, exposure="X", outcome="Y", output_dir=out, demo=True)

    # json.load rejects bare Infinity/NaN by default, so this parses only if nothing
    # non-finite was written.
    result = json.loads((out / "result.json").read_text())
    egger = next(e for e in result["estimates"] if e["method"] == "MR-Egger")
    assert egger["applicable"] is False
    assert "at least 3" in egger["reason"]
    assert "estimate" not in egger and "se" not in egger
    assert result["sensitivity"]["egger_intercept"] is None

    report = (out / "report.md").read_text()
    assert "MR-Egger was not computed" in report
    # The POSITIVE assertion, not merely the absence of the old sentence: the robustness
    # claim must NAME the estimators it actually compared. Asserting only that
    # "consistent estimates across IVW, MR-Egger" is gone passes just as well when a
    # non-applicable Egger is still fed into the comparison and drags it to "Caution" --
    # right conclusion, wrong reason, and the next real disagreement would be invisible.
    assert ("Sensitivity analyses show consistent estimates across IVW, "
            "Weighted Median, Weighted Mode") in report
    assert "MR-Egger, weighted median" not in report
    assert "| Egger intercept | not computed |" in report

    table = (out / "tables" / "mr_results.tsv").read_text()
    egger_row = next(l for l in table.splitlines() if l.startswith("MR-Egger"))
    assert "not_applicable" in egger_row
    assert "inf" not in egger_row.lower()


def test_the_json_writer_refuses_any_non_finite_value(tmp_path):
    """`allow_nan=False` is a BACKSTOP, and the fix removed every input that trips it.

    That is the point of it -- the pipeline no longer produces a non-finite -- but it
    also means no end-to-end test can observe it working. So drive the writer directly
    with a value it must refuse. Without the flag, Python writes a bare `Infinity`,
    which RFC 8259 has no token for, so a strict parser rejects the WHOLE document
    rather than one field. That is what a two-instrument run used to ship, exit code 0.
    """
    from mendelian_randomisation import SensitivityResults, _write_result_json

    broken = MREstimate("IVW", float("inf"), 1.0, 0.0, 2.0, 0.5, 4)
    with pytest.raises(ValueError):
        _write_result_json([broken], SensitivityResults(), 0.1, 0.5,
                           "X", "Y", tmp_path, "2026-01-01T00:00:00Z", True)


def test_the_scatter_plot_does_not_advertise_an_estimator_that_did_not_run(tmp_path):
    """A NaN slope draws no visible line and still claims a legend entry.

    matplotlib silently omits a line whose y values are NaN, so the plot LOOKS right --
    but the legend still gains an "MR-Egger (nan)" row, and a reader takes a legend
    entry as a statement that the method was fitted. The absence has to be an absence
    everywhere, not only in the tables.

    Recorded at the point the label is REQUESTED rather than read off the finished
    figure: `scatter_plot` closes its figure, so inspecting the current one afterwards
    reads an empty canvas and the test passes for the wrong reason.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.axes
    from mendelian_randomisation import scatter_plot

    requested = []
    real_plot = matplotlib.axes.Axes.plot

    def spy(self, *a, **kw):
        if "label" in kw:
            requested.append(kw["label"])
        return real_plot(self, *a, **kw)

    insts = _egger_input([0.2, 0.5])
    estimates = [
        ivw(insts),
        MREstimate.not_applicable("MR-Egger", 2, "requires at least 3 instruments"),
        weighted_median(insts),
    ]
    matplotlib.axes.Axes.plot = spy
    try:
        scatter_plot(insts, estimates, tmp_path / "scatter.png")
    finally:
        matplotlib.axes.Axes.plot = real_plot

    assert (tmp_path / "scatter.png").exists()
    assert requested, "no labelled line was drawn at all; the spy is not wired"
    assert not any("MR-Egger" in lab for lab in requested), requested
    assert not any("nan" in lab.lower() for lab in requested), requested
    assert any("IVW" in lab for lab in requested), requested


def test_a_zero_outcome_standard_error_is_refused_rather_than_passed_through():
    """The conditioning check is written `not (rel > t)`, and this is why.

    An instrument with `se_outcome = 0` gives it infinite weight, which makes both the
    dispersion and its scale infinite and their ratio NaN. Every comparison against NaN
    is False, so the natural-reading `rel < threshold` would be False and the estimator
    would proceed on a quantity that is not a number. The negated form turns that same
    False into a refusal.

    Reachable from a malformed input file rather than only in principle, which is why it
    is tested instead of merely commented.
    """
    insts = [
        Instrument(snp=f"rs{k}", effect_allele="A", other_allele="G", eaf=0.3,
                   beta_exposure=0.1 * (k + 1), se_exposure=0.01, pval_exposure=1e-10,
                   beta_outcome=0.05 * (k + 1), se_outcome=se, pval_outcome=0.01,
                   f_statistic=64.0)
        for k, se in enumerate([0.02, 0.0, 0.02])
    ]

    est, intercept, _, _ = mr_egger(insts)

    assert est.applicable is False
    assert "not identified" in est.reason
    assert math.isnan(est.estimate) and math.isnan(intercept)
