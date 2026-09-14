"""Select DSH for the unchanged upstream coding_agent_rl orchestration.

Hook: --custom-generate-function-path slime_dsh.generate.generate
The pinned example selects two module-level classes and has no external harness
registry. Bind those classes before its singleton service is created, then export
the original function. This version-specific wiring owns no rollout/eval loop.
Do not mix another coding-agent harness in the same rollout worker process.
"""
import hashlib
import json
from pathlib import Path

from slime.agent.adapters import OpenAIAdapter
from slime.utils.misc import SingletonMeta
from slime_dsh.harness import DshHarness

ROOT = Path(__file__).resolve().parents[1]


def bind_upstream():
    import examples.coding_agent_rl.generate as upstream

    if Path(upstream.__file__).resolve() != ROOT / "vendor/slime/examples/coding_agent_rl/generate.py":
        raise RuntimeError("DSH hook requires the project-local pinned slime checkout")
    pin = ROOT / "configs/slime/upstream-agent-sha256.json"
    for relative, expected in json.loads(pin.read_text()).items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Pinned upstream agent source changed: {relative}")
    if upstream._AdapterService in SingletonMeta._instances:
        raise RuntimeError("Select DSH before the upstream adapter service is initialized")
    upstream.HARNESS_CLS = DshHarness
    from slime_dsh import blocker
    # Anti-cheat interceptor (locked design §2.4) rides the same seam: reply-path
    # subclass, vendor untouched. Opt-in via DSH_BLOCKER=1 (smoke sheet wiring).
    upstream.ADAPTER_CLS = blocker.blocked_adapter_cls() if blocker.blocker_enabled() else OpenAIAdapter
    upstream.AGENT_NAME = "dsh"
    from slime_dsh import local_backend
    local_backend.bind(upstream.swe)
    return upstream.generate


generate = bind_upstream()
