"""Left-align and trim a typed allele against the GRCh38 reference (vt normalize).

Tan A, Abecasis GR, Kang HM. Unified representation of genetic variants. Bioinformatics.
2015;31(13):2202-2204. doi:10.1093/bioinformatics/btv112. PMID: 25701572. The paper defines a
VCF entry as normalized "if and only if it is left aligned and parsimonious" and gives the
algorithm implemented here (its Algorithm 1): while the alleles end with the same base, drop
it; when an allele is empty, extend both one base to the left; then drop a shared first base
while both alleles keep at least two.

This is the form gnomAD writes an indel in, and the one Open Targets holds (CFTR F508del
`7-117559590-ATCT-A`, BRCA1 c.5266dup `17-43057062-T-TG`), not the right-aligned form
Ensembl's variant recoder answers (`7-117559591-TCTT-T`, `17-43057063-G-GG`). It is not
NCBI's SPDI contextual allele, which widens a variant to the whole repeat it sits in.

The reference comes from Ensembl REST: `GET /sequence/region/human/{chrom}:{start}..{end}:1`
with `Accept: text/plain` answers the forward-strand bases and nothing else. One window per
allele on a 1,000-base grid with at least 2,000 bases to the left, so nearby alleles share
one; windows are kept for the life of the process. A one-base substitution is never sent.
Every failure to read a window is `ReferenceUnreachable` (an outage), never a finding about
the variant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from snp_phewas_chromosome import normalize_chromosome
from snp_phewas_tabix import get_with_retry

ENSEMBL_REST = "https://rest.ensembl.org"
SOURCE = "Ensembl's reference sequence"
#: Measured 2026-11-06 from one laptop: 10 window reads of 31 to 2,004 bases answered in
#: 0.48 to 11.13 s, median 2.8 s, all HTTP 200; latency did not follow length. A slower
#: answer is treated as an outage.
DEFAULT_TIMEOUT_S = 30.0
#: Reference bases read to the left of an allele. Latency does not depend on window length
#: (same 10 reads), so this is a choice with a stated failure (`RepeatExceedsWindow`), not a
#: measured repeat length.
LEFT_BP = 2000
#: Windows are aligned to this grid so nearby alleles share one (cached) window.
GRID = 1000
#: Bases read to the right of the reference allele.
RIGHT_PAD = 10
_BASES = frozenset("ACGTN")

#: GRCh38 primary-assembly chromosome lengths, read from Ensembl's `/info/assembly/homo_sapiens`
#: (GRCh38.p14) on 2026-11-06. A window is clamped to them: Ensembl clips a slice past the end
#: or refuses one that starts past it (HTTP 400), and either read as an outage would turn
#: every indel near a chromosome end into a false "could not be reached". Keyed by the one
#: chromosome name (`normalize_chromosome`), so `M` and `chrM` read MT's length.
CHROMOSOME_LENGTHS: dict[str, int] = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259,
    "6": 170805979, "7": 159345973, "8": 145138636, "9": 138394717, "10": 133797422,
    "11": 135086622, "12": 133275309, "13": 114364328, "14": 107043718, "15": 101991189,
    "16": 90338345, "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415, "MT": 16569,
}


class ReferenceUnreachable(ConnectionError):
    """Ensembl's reference sequence could not be READ (transport error, timeout, a status
    other than 200, or a body that is not the window's bases). An outage, never a finding
    about the variant."""


class ReferenceMismatch(ValueError):
    """The allele's reference bases are not GRCh38's at its position. `expected` holds the
    window's bases there (as many as the allele's reference)."""

    def __init__(self, chrom: str, pos: int, ref: str, expected: str) -> None:
        super().__init__(f"GRCh38 has {expected} at {chrom}:{pos}, not {ref}")
        self.chrom, self.pos, self.ref, self.expected = chrom, pos, ref, expected


class PositionPastChromosomeEnd(ReferenceMismatch):
    """The allele runs past the end of its GRCh38 chromosome: a placement fact (often a
    GRCh37 position or a typo), never an outage."""

    def __init__(self, chrom: str, pos: int, ref: str, length: int, last: int) -> None:
        ValueError.__init__(self, f"GRCh38 chromosome {chrom} ends at position {length:,}, and "
                                  f"this allele reaches position {last:,}")
        self.chrom, self.pos, self.ref, self.expected = chrom, pos, ref, ""
        self.length = length


class RepeatExceedsWindow(ValueError):
    """Left-alignment needed a base before the window's first one: the allele sits in a
    repeat longer than `window_bp`, the bases read to its left. Never guessed past."""

    def __init__(self, chrom: str, pos: int, window_bp: int) -> None:
        super().__init__(f"{chrom}:{pos} left-aligns past the {window_bp} bases read to its left")
        self.chrom, self.pos, self.window_bp = chrom, pos, window_bp


@dataclass(frozen=True)
class ReferenceWindow:
    """GRCh38 bases `seq` on `chrom` from `start` (1-based, inclusive)."""

    chrom: str
    start: int
    seq: str

    @property
    def end(self) -> int:
        return self.start + len(self.seq) - 1

    def bases(self, pos: int, n: int) -> str:
        """`n` bases from `pos`; `RepeatExceedsWindow` when any falls outside the window."""
        if pos < self.start or pos + n - 1 > self.end:
            raise RepeatExceedsWindow(self.chrom, pos, self.end - self.start + 1)
        return self.seq[pos - self.start:pos - self.start + n]

    def base(self, pos: int) -> str:
        return self.bases(pos, 1)


def window_bounds(pos: int, ref_len: int, *, chrom: str | None = None, ref: str = ""
                  ) -> tuple[int, int]:
    """`(start, end)`, 1-based inclusive, on the grid: at least `LEFT_BP` bases left of
    `pos` (fewer only at the chromosome's start) and the reference allele plus `RIGHT_PAD`
    to the right, never past the chromosome's end when `chrom` is a GRCh38 chromosome. An
    allele that itself runs past the end raises `PositionPastChromosomeEnd`."""
    length = CHROMOSOME_LENGTHS.get(normalize_chromosome(chrom)) if chrom is not None else None
    last = pos + max(ref_len, 1) - 1
    if length is not None and last > length:
        raise PositionPastChromosomeEnd(str(chrom), pos, ref, length, last)
    start = max(1, ((pos - LEFT_BP) // GRID) * GRID + 1)
    end = -(-(pos + max(ref_len, 1) + RIGHT_PAD) // GRID) * GRID
    if length is not None:
        end = min(end, length)
    return start, end


def _allele(a: str) -> str:
    a = (a or "").upper()
    return "" if a == "-" else a


def is_single_base_substitution(ref: str, alt: str) -> bool:
    ref, alt = _allele(ref), _allele(alt)
    return len(ref) == len(alt) == 1


def left_align(chrom: str, pos: int, ref: str, alt: str, window: ReferenceWindow
               ) -> tuple[str, int, str, str]:
    """vt normalize (Tan 2015, Algorithm 1). A one-base substitution, or an allele whose two
    sides are equal, is returned as is without reading `window`. Raises `ReferenceMismatch`
    when `ref` is not the window's bases at `pos` (checked before anything moves) and
    `RepeatExceedsWindow` when extending left needs a base before `window.start`."""
    ref, alt = _allele(ref), _allele(alt)
    if len(ref) == len(alt) == 1 or ref == alt:
        return chrom, pos, ref, alt
    if ref:
        expected = window.bases(pos, len(ref))
        if expected != ref:
            raise ReferenceMismatch(chrom, pos, ref, expected)
    start_pos = pos
    changed = True
    while changed:
        changed = False
        if ref and alt and ref[-1] == alt[-1]:
            ref, alt = ref[:-1], alt[:-1]
            changed = True
        if not ref or not alt:
            if pos - 1 < window.start:
                raise RepeatExceedsWindow(chrom, start_pos, start_pos - window.start)
            pos -= 1
            b = window.base(pos)
            ref, alt = b + ref, b + alt
            changed = True
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        pos += 1
    return chrom, pos, ref, alt


#: Windows read in this process, by (chrom, start, end). Failures are never kept.
_WINDOWS: dict[tuple[str, int, int], ReferenceWindow] = {}


def reference_window(chrom: str, pos: int, ref_len: int, *,
                     session: requests.Session | None = None,
                     timeout_s: float = DEFAULT_TIMEOUT_S) -> ReferenceWindow:  # pragma: no cover - network
    """The GRCh38 window `window_bounds(pos, ref_len)` on `chrom`: one GET, retried on a
    transport error or a 429 / 5xx, kept for the process. `ReferenceUnreachable` for
    anything but a 200 whose body is ACGTN of the requested length."""
    start, end = window_bounds(pos, ref_len, chrom=chrom)
    key = (chrom, start, end)
    if key in _WINDOWS:
        return _WINDOWS[key]
    url = f"{ENSEMBL_REST}/sequence/region/human/{chrom}:{start}..{end}:1"
    sess = session or requests.Session()
    try:
        resp = get_with_retry(sess, url, params={"coord_system_version": "GRCh38"},
                              headers={"Accept": "text/plain"}, timeout=timeout_s)
    except requests.RequestException as e:
        raise ReferenceUnreachable(f"{SOURCE}: {type(e).__name__}") from e
    finally:
        if session is None:
            sess.close()
    if resp.status_code != 200:
        raise ReferenceUnreachable(f"{SOURCE} answered HTTP {resp.status_code}")
    seq = (getattr(resp, "text", "") or "").strip().upper()
    if len(seq) != end - start + 1:
        raise ReferenceUnreachable(
            f"{SOURCE} answered {len(seq)} bases for {chrom}:{start}..{end}, not {end - start + 1}")
    if not set(seq) <= _BASES:
        raise ReferenceUnreachable(f"{SOURCE} answered something other than bases")
    window = ReferenceWindow(chrom, start, seq)
    _WINDOWS[key] = window
    return window


def normalize_allele(chrom: str, pos: int, ref: str, alt: str, *,
                     reference: Callable[..., Any] | None = None
                     ) -> tuple[str, int, str, str]:
    """`(chrom, pos, ref, alt)` left-aligned and trimmed against GRCh38. A one-base
    substitution is returned as is with no read. `reference(chrom, pos, ref_len)` gives the
    window; by default `reference_window`, looked up at call time. Raises
    `ReferenceUnreachable`, `ReferenceMismatch` (including `PositionPastChromosomeEnd`) or
    `RepeatExceedsWindow`."""
    if is_single_base_substitution(ref, alt):
        return chrom, pos, ref.upper(), alt.upper()
    ref_len = len(ref) if ref not in ("-", "") else 0
    # A position past the chromosome end is a placement fact: refused before any read.
    window_bounds(pos, ref_len, chrom=chrom, ref=ref)
    read = reference or reference_window
    return left_align(chrom, pos, ref, alt, read(chrom, pos, ref_len))
