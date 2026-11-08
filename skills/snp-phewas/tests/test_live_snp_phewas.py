"""Live smoke (network): the response SHAPES the offline fixtures assume, the golden values
on the live sources, and the two source behaviors this skill exists to handle.

Run with: RUN_LIVE_TESTS=1 python -m pytest skills/snp-phewas/tests/test_live_snp_phewas.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1"),
]

PAN = "https://pan-ukb-us-east-1.s3.amazonaws.com/"
LDL = "biomarkers-30780-both_sexes-irnt.tsv.bgz"
FG = "https://storage.googleapis.com/finngen-public-data-r12/summary_stats/release/finngen_R12_I9_CHD.gz"


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp("tbi"))


def test_finngen_api_shape_golden_and_p_threshold():
    from snp_phewas_pheweb import FinnGenPhewebClient

    payload = FinnGenPhewebClient().fetch_variant("1", 109274968, "G", "T")
    res = {r["phenocode"]: r for r in payload["results"]}
    assert len(res) >= 2469
    chd = res["I9_CHD"]
    assert {"beta", "sebeta", "pval", "mlogp", "maf", "n_case", "n_control"} <= set(chd)
    assert chd["beta"] == pytest.approx(-0.0801145) and chd["sebeta"] == pytest.approx(0.00864016)
    assert chd["pval"] == pytest.approx(1.82e-20, rel=1e-2)
    # The API fills values only below p = 0.05; the rest come back null (snp_phewas_pheweb).
    valued = [r for r in payload["results"] if r.get("pval") is not None]
    assert valued and all(r["pval"] < 0.05 for r in valued)
    assert any(r.get("pval") is None for r in payload["results"])
    assert FinnGenPhewebClient().fetch_variant("1", 109274968, "G", "A") is None


def test_finngen_api_maf_is_the_alt_allele_frequency_live():
    """The decisive class for "folded or not" is a variant whose alt allele is the major
    one: a folded minor-allele frequency can never exceed 0.5. rs593651 (1:109266488-G-A)
    has alt A at 0.751 in FinnGen R12 (file af_alt; the API's own AC/AN agrees). If this
    fails, the API started folding `maf` and FinnGenReader._api_record must unfold it."""
    from snp_phewas_pheweb import FinnGenPhewebClient

    payload = FinnGenPhewebClient().fetch_variant("1", 109266488, "G", "A")
    assert payload is not None
    annot = payload["variant"]["annotation"]["annot"]
    alt_af = int(annot["AC"]) / int(annot["AN"])
    assert alt_af > 0.7  # the premise: alt is the major allele here
    mafs = [r["maf"] for r in payload["results"] if r.get("maf") is not None]
    assert len(mafs) >= 100
    assert all(m > 0.5 for m in mafs), (min(mafs), len(mafs))
    assert all(abs(m - alt_af) < 0.01 for m in mafs), (min(mafs), max(mafs), alt_af)
    # `maf` is censored together with beta: a result has both or neither
    assert all((r.get("beta") is None) == (r.get("maf") is None) for r in payload["results"])


def test_finngen_tabix_equals_api_live(cache):
    from snp_phewas_tabix import ensure_index, fetch_position, read_header

    hdr = read_header(FG)
    (row,) = fetch_position(FG, ensure_index(FG + ".tbi", cache), "1", 109274968)
    rec = dict(zip(hdr, row))
    assert (rec["ref"], rec["alt"]) == ("G", "T")
    assert float(rec["beta"]) == pytest.approx(-0.0801145)
    assert float(rec["sebeta"]) == pytest.approx(0.00864016)
    assert float(rec["af_alt"]) == pytest.approx(0.214503)


def test_pan_ukbb_ldl_golden_and_the_index_defect(cache):
    from snp_phewas_tabix import ensure_index, fetch_position, read_header

    idx = ensure_index(PAN + "sumstats_flat_files_tabix/" + LDL + ".tbi", cache)
    hdr = read_header(PAN + "sumstats_flat_files/" + LDL)
    assert {"chr", "pos", "ref", "alt", "beta_EUR", "se_EUR", "neglog10_pval_EUR", "af_EUR",
            "low_confidence_EUR"} <= set(hdr)
    (row,) = fetch_position(PAN + "sumstats_flat_files/" + LDL, idx, "1", 109817590)
    rec = dict(zip(hdr, row))
    assert float(rec["beta_EUR"]) == pytest.approx(-0.1345)
    assert float(rec["se_EUR"]) == pytest.approx(0.002913)
    assert float(rec["neglog10_pval_EUR"]) == pytest.approx(464.8)
    # Positions an exact-position htslib query on this file missed (measured 2026-10-17); the
    # reader must find every one of them.
    for chrom, pos in (("1", 109817590), ("5", 51097763), ("5", 51401766), ("5", 50327893)):
        assert fetch_position(PAN + "sumstats_flat_files/" + LDL, idx, chrom, pos), (chrom, pos)


def test_ensembl_grch37_mapping_live():
    from snp_phewas_variant import Placement, map_placement

    p37, note = map_placement(Placement("GRCh38", "1", 109274968, "G", "T"), "GRCh37",
                              map_fn=None)
    assert (p37.pos, p37.ref, p37.alt, p37.strand_vs_grch38) == (109817590, "G", "T", 1)


def test_rsid_resolution_live():
    from snp_phewas_variant import resolve_variant

    v = resolve_variant("rs12740374")
    assert v.variant_id_grch38 == "1_109274968_G_T" and v.rs_id == "rs12740374"


def test_open_targets_resolver_returns_ranked_ids_live():
    from snp_phewas_ot_resolver import search_variant_ids

    assert search_variant_ids("rs12740374")[0] == "1_109274968_G_T"
    assert search_variant_ids("rs0") == []


def test_quick_mode_end_to_end_live(tmp_path):
    import snp_phewas

    assert snp_phewas.main(["--variant", "rs12740374", "--quick", "--output", str(tmp_path)]) == 0
    report = (tmp_path / "report.md").read_text()
    assert "Quick mode: FinnGen web API only" in report
    rows = (tmp_path / "tables" / "snp_phewas_rows.tsv").read_text().splitlines()
    assert len(rows) == 1 + 2469
    assert any("\tI9_CHD\t" in r and "\tok\t" in r for r in rows)


def test_mvp_ldl_mean_reads_live_with_the_p_underflow_derived(cache):
    """Million Veteran Program LDL cholesterol mean, European ancestry (GCST90475416), read
    from the GWAS Catalog's harmonized file: CRLF lines, p = 0 in the file (measured
    2026-10-19: beta -0.1561, SE 0.003009)."""
    from snp_phewas_catalog import get_dataset, load_outcome_catalog
    from snp_phewas_sources import HM_CODES, P_UNDERFLOW_DETAIL, MvpReader
    from snp_phewas_tabix import ensure_index, fetch_position, read_header
    from snp_phewas_variant import Placement

    o = {x.outcome_id: x for x in load_outcome_catalog("mvp_eur")}["GCST90475416"]
    reader = MvpReader(get_dataset("mvp_eur"), index_cache=cache)
    rec = reader.read_outcome(o, Placement("GRCh38", "1", 109274968, "G", "T"))
    assert rec.status == "ok" and rec.variant_id_native == "1_109274968_G_T"
    assert (rec.beta, rec.se) == (-0.1561, 0.003009)
    assert rec.detail == P_UNDERFLOW_DETAIL and rec.neglog10_p > 500
    # the live record's harmonization code is one the code table has a rule for
    data, idx = reader.paths(o)
    header = read_header(data)
    hits = fetch_position(data, ensure_index(idx, reader.index_cache), "1", 109274968)
    codes = {dict(zip(header, f))["hm_code"].strip() for f in hits}
    assert codes and codes <= set(HM_CODES)


def test_rs7903146_resolves_to_the_common_allele_with_gwas_signals_live():
    """Open Targets' search lists the rare C>G first; the lookup must read C>T."""
    from snp_phewas_variant import resolve_variant

    v = resolve_variant("rs7903146")
    assert v.grch38.variant_id == "10_112998590_C_T"
    assert "most GWAS credible sets" in v.resolution_note and "10_112998590_C_G" in v.resolution_note


@pytest.mark.parametrize("label, right, left", [
    ("CFTR F508del", "7_117559591_TCTT_T", "7_117559590_ATCT_A"),
    ("BRCA1 c.5266dup", "17_43057063_G_GG", "17_43057062_T_TG"),
])
def test_live_a_right_aligned_typed_id_resolves_to_the_left_aligned_one(label, right, left):
    """Through Ensembl's reference sequence, a right-aligned typed indel is read under the
    left-aligned id gnomAD and Open Targets hold."""
    from snp_phewas_variant import resolve_variant

    v = resolve_variant(right)
    assert v.variant_id_grch38 == left, label
    assert v.query == right and v.anchor_variant_id == left
    kind = "deletion" if len(left.split("_")[2]) > len(left.split("_")[3]) else "insertion"
    assert f"is the same {kind} as {left}" in v.alignment_note
