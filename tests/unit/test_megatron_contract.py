from pathlib import Path

import pytest

from resihp.compat.megatron_contract import CONTRACT, PATCHED_CONTRACT, Requirement, check, is_patched

ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Megatron-LM"


@pytest.mark.skipif(not ROOT.exists(), reason="run scripts/setup_megatron.sh first")
def test_pinned_megatron_satisfies_contract():
    assert check(ROOT) == []


@pytest.mark.skipif(not ROOT.exists() or not is_patched(ROOT), reason="patches not applied")
def test_patch_set_provides_symbols():
    assert check(ROOT, PATCHED_CONTRACT) == []


def test_contract_detects_violations(tmp_path):
    pkg = tmp_path / "megatron" / "core"
    pkg.mkdir(parents=True)
    (pkg / "x.py").write_text("class A:\n    a: int = 0\n    def f(self, p):\n        pass\ndef g(q, **kw):\n    pass\n")
    reqs = (
        Requirement("megatron/core/x.py", "A", fields=("a", "b")),
        Requirement("megatron/core/x.py", "A.f", args=("p", "z")),
        Requirement("megatron/core/x.py", "g", args=("anything",)),
        Requirement("megatron/core/x.py", "missing"),
        Requirement("megatron/core/y.py", "A"),
    )
    problems = check(tmp_path, reqs)
    assert len(problems) == 4
    assert any("lacks fields ['b']" in p for p in problems)
    assert any("lacks args ['z']" in p for p in problems)
    assert len(CONTRACT) > 20
