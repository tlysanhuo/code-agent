#!/usr/bin/env python3
"""Multi-turn stitching gate (pre-launch check; round1-runlog incident #5).

Drives the real OpenAIAdapter through a 2-turn session against the vendor CPU
fakes, with a client that echoes assistant messages WITHOUT reasoning_content
(pi-ai behavior). The scripted turn-1 output contains a <think> block, so the
echo renders without the sampled think tokens -- the exact drift that made the
upstream chain builder train only each session's final message.

Acceptance (turn-sample adapter): one sample per turn, every turn's output
fully trainable, total trainable == total scripted output tokens.
Regression proof (vanilla adapter): must train strictly less than that.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/slime"))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
import importlib.util  # noqa: E402

from slime.agent.adapters import OpenAIAdapter  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

# load the vendor fakes by file path: the train venv's path carries another
# vendored snapshot whose top-level `tests` package shadows slime's namespace
_spec = importlib.util.spec_from_file_location(
    "_stitch_fakes", ROOT / "vendor/slime/tests/test_agent/_fakes.py")
_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fakes)
FakeSGLangServer, FakeTokenizer = _fakes.FakeSGLangServer, _fakes.FakeTokenizer

from slime_dsh import turn_samples  # noqa: E402

TURN1_TEXT = "<think>plan things</think>Answer one"
TURN2_TEXT = "Final answer"


async def drive_session(adapter_cls, tok, server, sid):
    adapter = adapter_cls(tokenizer=tok, sglang_url=server.url,
                          reasoning_parser="qwen3", max_turns_per_sid=5)
    adapter.open_session(sid)
    async with TestClient(TestServer(adapter.app)) as client:
        messages = [{"role": "user", "content": "please work"}]
        outs = []
        for obs in (None, "observation data"):
            if obs:
                messages.append({"role": "tool", "content": obs})
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": messages, "stream": False,
                "max_tokens": 32},
                headers={"Authorization": f"Bearer {sid}"})
            assert r.status == 200, await r.text()
            msg = (await r.json())["choices"][0]["message"]
            # pi-ai style echo: content only, no reasoning_content, no ids
            messages.append({"role": "assistant", "content": msg["content"]})
            outs.append(msg)
    base = Sample(index=0, group_index=0,
                  prompt=[{"role": "user", "content": "please work"}])
    samples = await adapter.finish_session(sid, base_sample=base, reward=1.0,
                                           extra_metadata={"instance_id": "gate"})
    return outs, samples


def trainable_tokens(samples):
    return sum(sum(1 for x in s.loss_mask if x == 1) for s in samples)


async def main():
    tok = FakeTokenizer()
    t1 = tok.encode(TURN1_TEXT)          # registers words, stable ids
    t2 = tok.encode(TURN2_TEXT)
    scripted_total = len(t1) + len(t2)

    # --- vanilla adapter: must FAIL full coverage (regression proof) ---
    async with FakeSGLangServer([[(0.0, i) for i in t1],
                                 [(0.0, i) for i in t2]]) as server:
        outs, samples = await drive_session(OpenAIAdapter, tok, server, "vanilla")
        vanilla_cov = trainable_tokens(samples)
        assert outs[0]["content"] == "Answer one", outs[0]
        assert outs[1]["content"] == "Final answer", outs[1]

    # --- turn-sample adapter: must pass every acceptance criterion ---
    tok2 = FakeTokenizer()
    t1b, t2b = tok2.encode(TURN1_TEXT), tok2.encode(TURN2_TEXT)
    async with FakeSGLangServer([[(0.0, i) for i in t1b],
                                 [(0.0, i) for i in t2b]]) as server:
        _, samples = await drive_session(
            turn_samples.adapter_cls(OpenAIAdapter), tok2, server, "turnlevel")
        fails = []
        if len(samples) < 2:
            fails.append(f"expected >=2 samples, got {len(samples)}")
        for k, s in enumerate(samples):
            if len(s.loss_mask) != s.response_length:
                fails.append(f"sample {k}: len(loss_mask) {len(s.loss_mask)} != response_length {s.response_length}")
            if sum(1 for x in s.loss_mask if x == 1) != s.response_length:
                fails.append(f"sample {k}: mask sum != response_length")
            if s.reward != 1.0:
                fails.append(f"sample {k}: reward not assigned")
        cov = trainable_tokens(samples)
        if cov != scripted_total:
            fails.append(f"coverage {cov} != scripted {scripted_total}")
        if len(samples) >= 2 and "Answer one" not in samples[0].response:
            fails.append(f"turn-1 sample trains wrong text: {samples[0].response!r}")

    print(f"vanilla trainable tokens : {vanilla_cov}/{scripted_total}")
    print(f"turn-level trainable     : {cov}/{scripted_total} "
          f"({len(samples)} samples)")
    print("GATE:", "PASS" if not fails else f"FAIL {fails}")
    sys.exit(0 if (not fails and vanilla_cov < scripted_total) else 1)


if __name__ == "__main__":
    asyncio.run(main())
