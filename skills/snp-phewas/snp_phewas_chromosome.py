"""The one chromosome-name rule a typed variant id is read by, and the format a refusal asks for.

A chromosome is named 1-22, X, Y or MT. One `chr` prefix is stripped in any case, the rest is
uppercased, and `M` folds to `MT` (`chr7` -> `7`, `chrx` -> `X`, `chrM` -> `MT`). Everything
else is refused, not guessed: 0, 23 and above (numeric codes some formats use for X, Y or MT
are a source's own convention, not a typed name), zero-padded numbers (`01`), `XY`, and patch
or scaffold names. A refused id is answered with the format the skill requires, on the build
the caller named.
"""

from __future__ import annotations

import re

CHROMOSOMES: frozenset[str] = frozenset([*(str(i) for i in range(1, 23)), "X", "Y", "MT"])

#: The rule in words: the clause every refusal of a chromosome name ends with.
CHROMOSOME_NAME_RULE = ("chromosomes are named 1-22, X, Y or MT; a chr prefix and lowercase "
                        "are accepted, and M is read as MT")

#: One example id per build: rs12740374, G>T, at chr1:109,274,968 on GRCh38 and
#: chr1:109,817,590 on GRCh37.
_FORMAT_EXAMPLES = {"GRCh38": "1_109274968_G_T", "GRCh37": "1_109817590_G_T"}

# The shape of a typed id with any of the separators one is written with (`_`, `:`, `-`, a
# space before the alleles, `>` between them, as in `chr23:100 A>G`): used only to say WHY an
# id was refused, never to accept one.
_TYPED_ID_SHAPE = re.compile(
    r"^([0-9A-Za-z]+)[_:\-](\d+)(?:[_:\-]|\s+)([ACGTN]+)[_:\->]([ACGTN]+)$", re.IGNORECASE)


def normalize_chromosome(name: str | int | None) -> str | None:
    """The one name of a chromosome (`CHROMOSOMES`), or None when `name` is not one."""
    if name is None:
        return None
    s = str(name)
    if s[:3].lower() == "chr":
        s = s[3:]
    s = s.upper()
    if s == "M":
        s = "MT"
    return s if s in CHROMOSOMES else None


def variant_id_format(build: str = "GRCh38") -> str:
    """The variant-id format a refusal asks for, on the build the caller named (GRCh38 or
    GRCh37; anything else is said as GRCh38, the default)."""
    build = build if build in _FORMAT_EXAMPLES else "GRCh38"
    return (f"chromosome_position_ref_alt on {build}, for example {_FORMAT_EXAMPLES[build]}, "
            f"where {CHROMOSOME_NAME_RULE}")


def refused_chromosome(variant_id: object) -> str | None:
    """The chromosome as written when `variant_id` has a variant id's shape and only its
    chromosome is refused (`23_1_A_G` -> `23`, `chr23:1:A>G` -> `chr23`); else None."""
    m = _TYPED_ID_SHAPE.match(str(variant_id or "").strip())
    if not m or normalize_chromosome(m.group(1)) is not None:
        return None
    return m.group(1)


def variant_id_refusal(variant_id: object, *, build: str = "GRCh38") -> str:
    """The sentence said when a variant id is refused: what was wrong when only the chromosome
    was, and always the format required, on the build the caller named."""
    said = str(variant_id or "").strip()
    bad = refused_chromosome(said)
    if bad is not None:
        return (f"{said!r} names chromosome {bad!r}, which is not one we read. Write the "
                f"variant as {variant_id_format(build)}")
    return f"{said!r} is not a variant id we read. Write it as {variant_id_format(build)}"
