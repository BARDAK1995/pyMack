"""Drift tests for the hard-cell truth corpus (spec slice 01).

Skips cleanly when ``data/`` is absent (the NPZs are regenerable and
gitignored).  When data is present, every stored verdict is re-derived from
the stored spectrum through the EXISTING production filter functions
(``discrete_mode._decaying_candidates``, ``pymack.dense.candidate_indices`` +
``_select_seed``, and the transcribed band filters that were cross-checked
verbatim against the production drivers at build time) and asserted to match.
This guards against filter-code drift: if any production filter changes
behavior, the re-derived verdicts diverge from the frozen corpus.

Additional integrity gates: sha256 checksums of every NPZ's arrays against
the manifest, and assertions that the frozen transcribed filter constants
still equal the production module values.  None of these tests need the
gitignored selection CSVs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_truth_set as bts  # noqa: E402  (shared derivation code)


def _corpus_or_skip():
    if not bts.MANIFEST_PATH.exists() or not any(bts.DATA_DIR.glob("*.npz")):
        pytest.skip(
            "hard-cell corpus data absent; run build_truth_set.py to "
            "generate it",
            allow_module_level=True,
        )
    return json.loads(bts.MANIFEST_PATH.read_text(encoding="utf-8"))


MANIFEST = _corpus_or_skip()
CENSUS = json.loads(bts.CENSUS_PATH.read_text(encoding="utf-8"))
_CELL_IDS = [c["id"] for c in MANIFEST["cells"]]
_BY_ID = {c["id"]: c for c in MANIFEST["cells"]}
_CENSUS_BY_ID = {c["id"]: c for c in CENSUS["cells"]}


def _close(a, b, tol=1.0e-12):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def _same_selected(a, b, tol=1.0e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(complex(a[0], a[1]) - complex(b[0], b[1])) <= tol


@pytest.mark.parametrize("cell_id", _CELL_IDS)
def test_stored_verdict_rederives_from_stored_spectrum(cell_id):
    entry = _BY_ID[cell_id]
    npz_path = HERE / entry["npz"]
    assert npz_path.exists(), f"missing corpus NPZ: {npz_path}"
    cell, spectra, stored, _kappa, crosscheck = bts.load_npz(npz_path)

    assert cell["id"] == cell_id
    assert cell["params"] == entry["params"], \
        "manifest params drifted from the NPZ cell definition"
    assert crosscheck.get("match") is True, \
        f"{cell_id}: build-time cross-check against the production driver " \
        f"was not clean: {crosscheck}"

    rederived = bts.derive_verdict(cell, spectra)

    assert rederived["status"] == stored["status"], \
        f"{cell_id}: status drift {rederived['status']} != {stored['status']}"
    assert _same_selected(rederived.get("selected"), stored.get("selected")), \
        f"{cell_id}: selected drift {rederived.get('selected')} != " \
        f"{stored.get('selected')}"
    for key in ("n_match", "n_band", "n_filtered"):
        if key in stored or key in rederived:
            assert rederived.get(key) == stored.get(key), \
                f"{cell_id}: {key} drift"
    for key in ("selected_fs", "omega_i", "growth", "phase_speed",
                "min_match_distance"):
        if key in stored or key in rederived:
            assert _close(rederived.get(key), stored.get(key)), \
                f"{cell_id}: {key} drift"

    # The manifest carries the same verdict as the NPZ.
    assert stored["status"] == entry["verdict"]["status"]
    assert _same_selected(stored.get("selected"),
                          entry["verdict"].get("selected"))


@pytest.mark.parametrize("cell_id", _CELL_IDS)
def test_census_in_box_counts_rederive(cell_id):
    entry = _BY_ID[cell_id]
    cell, spectra, _stored, kappa, _cc = bts.load_npz(HERE / entry["npz"])
    row = bts.census_row_for(cell, spectra, kappa)
    ref = _CENSUS_BY_ID[cell_id]
    got = [(ps["spectrum"], ps["in_box_count"]) for ps in row["per_spectrum"]]
    want = [(ps["spectrum"], ps["in_box_count"])
            for ps in ref["per_spectrum"]]
    assert got == want, f"{cell_id}: census in-box counts drifted"


@pytest.mark.parametrize("cell_id", _CELL_IDS)
def test_npz_array_checksums(cell_id):
    """Catches verdict-preserving NPZ corruption (reviewer finding 3)."""
    entry = _BY_ID[cell_id]
    assert "npz_sha256" in entry, f"{cell_id}: manifest missing npz_sha256"
    got = bts.npz_arrays_sha256(HERE / entry["npz"])
    assert got == entry["npz_sha256"], \
        f"{cell_id}: NPZ array content diverged from the manifest checksum"


def test_frozen_constants_match_production_modules():
    """The transcribed filter constants must still equal the production
    values they were copied from (reviewer finding 3)."""
    import inspect

    # Mack fig10.4 band filter constants.
    assert bts.MACK_M10["cr_band"] == [float(bts.mk.CR_LO),
                                       float(bts.mk.CR_HI)]
    assert bts.MACK_M10["ci_cap"] == float(bts.mk.CI_CAP)
    assert bts.MACK_M10["N"] == int(bts.mk.N_BY_MACH[10.0])
    assert bts.MACK_M10["y_max"] == float(bts.mk.Y_MAX_BY_MACH[10.0])

    # Ma&Zhong: case constants + the 0.06 alpha_i cap inside growth().
    assert bts.MAZHONG["N"] == int(bts.mzc.N)
    assert bts.MAZHONG["y_max"] == float(bts.mzc.Y_MAX)
    assert bts.MAZHONG["wall_bc"] == str(bts.mzc.WALL_BC)
    assert bts.MAZHONG["lambda_mu_ratio"] == float(bts.mzc.LAMBDA_MU)
    growth_src = inspect.getsource(bts.mztr.growth)
    assert "0.06" in growth_src, \
        "trace_mazhong_curves.growth alpha_i cap changed; " \
        "MAZHONG['ai_cap'] transcription is stale"
    assert bts.MAZHONG["ai_cap"] == 0.06

    # Ozgen discrete-mode extractor defaults + grid-builder constants.
    sig = inspect.signature(bts.dm.discrete_mode)
    defaults = {k: v.default for k, v in sig.parameters.items()}
    assert bts.OZGEN_FIRST["ci_abs_max"] == defaults["ci_abs_max"]
    assert tuple(bts.OZGEN_FIRST["cr_band"]) == defaults["cr_band"]
    assert bts.OZGEN_FIRST["fs_thresh"] == defaults["fs_thresh"]
    assert bts.OZGEN_FIRST["match_tol"] == defaults["match_tol"]
    import build_firstmode_grid as fmg
    import build_secondmode_grid as smg
    assert bts.OZGEN_FIRST["N"] == fmg.N
    assert {k: tuple(v) for k, v in bts.OZGEN_YMF_FIRST.items()} == \
        {k: tuple(v) for k, v in fmg.YMF_BY_MACH.items()}
    assert bts.OZGEN_SECOND["N"] == smg.N
    assert tuple(bts.OZGEN_YMF_SECOND) == tuple(smg.YMF)
    assert tuple(bts.OZGEN_SECOND["cr_band"]) == tuple(smg.CR_BAND)

    # Ma&Zhong band definitions used by the corpus params.
    for cell in MANIFEST["cells"]:
        if cell["kind"] != "mazhong_spatial":
            continue
        band = bts.mztr.MODES[cell["params"]["mode"]]
        assert cell["params"]["c_lo"] == float(band["c_lo"])
        assert cell["params"]["c_hi"] == float(band["c_hi"])


def test_verdict_basis_and_reviewer_annotations():
    for cell in MANIFEST["cells"]:
        assert cell.get("verdict_basis") == bts.VERDICT_BASIS[cell["kind"]]
        if cell["kind"] == "mach6_dense":
            assert "seed_vs_tracked_alpha_distance" in cell
    contradicted = [c["id"] for c in MANIFEST["cells"]
                    if c.get("box_content_contradicts_verdict")]
    assert contradicted == ["hc_043"], \
        f"expected exactly hc_043 flagged, got {contradicted}"
    mack = CENSUS["families"]["mack_fig10_4_m10_3d"]
    assert mack["n_no_discrete_mode_cells"] == 0
    assert "empty_box_note" in mack


def test_manifest_census_consistency():
    assert MANIFEST["cell_count"] == len(MANIFEST["cells"]) == len(_CELL_IDS)
    assert CENSUS["cell_count"] == len(CENSUS["cells"])
    assert set(_CENSUS_BY_ID) == set(_BY_ID)
    l_from_families = max(item["L"] for item in CENSUS["families"].values())
    assert CENSUS["l_max"] == l_from_families
    # every family in the census is populated and carries a recommendation
    for fam, item in CENSUS["families"].items():
        assert item["cell_count"] >= 1, fam
        for key in ("N_q", "L", "margin", "collar"):
            assert key in item, (fam, key)


def test_no_discrete_mode_is_a_first_class_outcome():
    statuses = {c["verdict"]["status"] for c in MANIFEST["cells"]}
    assert "no_discrete_mode" in statuses, \
        "corpus must contain honest empty-box cells"
    assert "discrete_mode" in statuses
