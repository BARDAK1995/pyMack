"""Bitwise certification of the 2-D temporal assembly extraction (slice 04).

The inline assemblies of ``pymack.solver.solve_temporal_compressible`` (Mack
enthalpy form) and ``pymack.temporal_solver.solve_temporal_2d`` (Ozgen form)
were extracted into module-level functions ``_assemble_temporal_2d_evp`` and
``_assemble_temporal_ozgen_2d_evp`` as pure code motion.  The fixtures in
``validation/data/`` were captured BEFORE that refactor by intercepting the
exact ``(A, B)`` pair each public solver handed to ``scipy.linalg.eig``
(see ``validation/data/capture_2d_assembly_fixtures.py``).

These tests call the NEW extracted functions at the recorded parameter points
and require bitwise equality -- ``np.array_equal`` plus raw-byte identity, not
``allclose`` -- for both matrices at every point.  Any failure means the
refactor reordered arithmetic and must be fixed in the refactor, never here.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pymack.solver import _assemble_temporal_2d_evp
from pymack.temporal_solver import _assemble_temporal_ozgen_2d_evp

DATA_DIR = Path(__file__).resolve().parent / "data"
FIXTURE_FILES = {
    "mack": DATA_DIR / "temporal_2d_mack_assembly_fixtures.npz",
    "ozgen": DATA_DIR / "temporal_2d_ozgen_assembly_fixtures.npz",
}
EXPECTED_FIXTURE_VERSION = 2


def _load_capture_module():
    path = DATA_DIR / "capture_2d_assembly_fixtures.py"
    spec = importlib.util.spec_from_file_location(
        "capture_2d_assembly_fixtures", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CAPTURE = _load_capture_module()


def _load_fixture(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"missing fixture {path}; it must be captured PRE-refactor with "
            "validation/data/capture_2d_assembly_fixtures.py"
        )
    return np.load(path, allow_pickle=False)


_NPZ = {name: _load_fixture(path) for name, path in FIXTURE_FILES.items()}
_CASES = {name: json.loads(npz["cases_json"].item()) for name, npz in _NPZ.items()}


def _stored_inputs(npz, i):
    """Rebuild the discretized assembler inputs recorded for case ``i``."""
    y = npz[f"y_{i}"]
    D1 = npz[f"D1_{i}"]
    D2 = npz[f"D2_{i}"]
    prefix = f"bf_{i}_"
    bf = {key[len(prefix):]: npz[key] for key in npz.files if key.startswith(prefix)}
    assert bf, f"no base-flow fields stored for case {i}"
    return y, D1, D2, bf


def _call_assembler(name, case, y, D1, D2, bf):
    if name == "mack":
        return _assemble_temporal_2d_evp(
            bf, y, D1, D2,
            case["alpha"], case["Re"], case["Ma"], case["Pr"], case["gamma"],
            wall_bc=case["wall_bc"],
            lambda_mu_ratio=case["lambda_mu_ratio"],
        )
    return _assemble_temporal_ozgen_2d_evp(
        bf, y, D1, D2,
        case["alpha"], case["Re"], case["Ma"], case["Pr"], case["gamma"],
        wall_bc=case["wall_bc"],
    )


def _assert_bitwise(actual, reference, label):
    assert actual.dtype == reference.dtype, label
    assert actual.shape == reference.shape, label
    assert np.array_equal(actual, reference), label
    # Raw-byte identity is stricter than array_equal (signed zeros); pure code
    # motion must preserve even those.
    assert np.ascontiguousarray(actual).tobytes() == \
        np.ascontiguousarray(reference).tobytes(), f"{label}: byte mismatch"


def _case_params():
    params = []
    for name, cases in sorted(_CASES.items()):
        for i, case in enumerate(cases):
            params.append(
                pytest.param(
                    name, i,
                    id=f"{name}-{i}-N{case['N']}-Ma{case['Ma']}-{case['wall_bc']}",
                )
            )
    return params


@pytest.mark.parametrize("name,i", _case_params())
def test_extracted_assembler_bitwise_on_stored_inputs(name, i):
    """Stored discretized inputs -> extracted assembler == captured (A, B)."""
    npz = _NPZ[name]
    case = _CASES[name][i]
    y, D1, D2, bf = _stored_inputs(npz, i)
    A, B = _call_assembler(name, case, y, D1, D2, bf)
    _assert_bitwise(A, npz[f"A_{i}"], f"{name}[{i}] A")
    _assert_bitwise(B, npz[f"B_{i}"], f"{name}[{i}] B")


@pytest.mark.parametrize("name,i", _case_params())
def test_extracted_assembler_bitwise_from_recorded_tuples(name, i):
    """Recorded parameter tuple -> profile -> discretization -> assembler.

    Re-derives everything from the recorded parameters through the same
    public-path discretization the solver performs, checks it reproduces the
    stored inputs exactly, then requires the extracted assembler to match the
    pre-refactor capture bitwise.
    """
    npz = _NPZ[name]
    case = _CASES[name][i]
    if name == "mack":
        y, D1, D2, bf = _CAPTURE.mack_discretization(case)
    else:
        y, D1, D2, bf = _CAPTURE.ozgen_discretization(case)

    y_s, D1_s, D2_s, bf_s = _stored_inputs(npz, i)
    assert np.array_equal(y, y_s)
    assert np.array_equal(D1, D1_s)
    assert np.array_equal(D2, D2_s)
    assert set(bf) == set(bf_s)
    for key in sorted(bf_s):
        assert np.array_equal(np.asarray(bf[key]), bf_s[key]), key

    A, B = _call_assembler(name, case, y, D1, D2, bf)
    _assert_bitwise(A, npz[f"A_{i}"], f"{name}[{i}] A")
    _assert_bitwise(B, npz[f"B_{i}"], f"{name}[{i}] B")


@pytest.mark.parametrize("name", sorted(FIXTURE_FILES))
def test_fixture_coverage(name):
    """The fixture set spans the parameter axes the slice contract demands."""
    npz = _NPZ[name]
    cases = _CASES[name]
    assert int(npz["fixture_version"]) == EXPECTED_FIXTURE_VERSION
    assert int(npz["n_cases"]) == len(cases)
    assert len(cases) >= 6
    assert {c["wall_bc"] for c in cases} == {"isothermal", "adiabatic"}
    assert {c["N"] for c in cases} == {31, 64}
    assert len({c["alpha"] for c in cases}) >= 6
    assert len({c["Re"] for c in cases}) >= 6
    assert len({c["Ma"] for c in cases}) >= 6
    for i in range(len(cases)):
        n = len(npz[f"y_{i}"])
        assert npz[f"A_{i}"].shape == (4 * n, 4 * n)
        assert npz[f"B_{i}"].shape == (4 * n, 4 * n)
        assert npz[f"A_{i}"].dtype == np.complex128
        assert npz[f"B_{i}"].dtype == np.complex128
