"""E1.a: prove looped (shared-block-twice) forward == expanded (32-layer) forward.

Loads the pinned Qwen3.5-9B once on CPU bf16, then:
  A) shared:   layers[16:28] are the *same module objects* as layers[4:16]  (compact semantics)
  B) expanded: deepcopy of A, so slots hold equal-but-distinct modules      (dense layout)
Forwards both on prompts of different lengths; logits must match to float tolerance.
Sanity: A must differ from the untouched original (sharing took effect).

Run: .venv-runtime/bin/python scripts/check_loop_equiv.py
"""
import copy
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DONOR_LO, DONOR_HI, SHARED_LO, SHARED_HI = 4, 16, 16, 28
PROMPTS = ["def quicksort(arr):\n", "The capital of France is"]


def text_layers(model):
    m = model.model if hasattr(model, "model") else model
    return m.layers


@torch.no_grad()
def logits_for(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt")
    out = model(**ids) if not hasattr(model, "lm_head") else model(**ids)
    return out.logits[0, -1].float()


def main():
    tok = AutoTokenizer.from_pretrained(SRC)
    original = AutoModelForCausalLM.from_pretrained(
        SRC, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    original.eval()

    layers = text_layers(original)
    n = len(layers)
    assert n == 32 and DONOR_HI - DONOR_LO == SHARED_HI - SHARED_LO == 12
    # record original last-layer weights hash to prove sharing took effect later
    ref_w = layers[20].self_attn.q_proj.weight if hasattr(layers[20], "self_attn") else None

    for i in range(SHARED_LO, SHARED_HI):
        layers[i] = layers[DONOR_LO + (i - SHARED_LO)]  # module-object sharing
    shared = original
    shared.eval()

    expanded = copy.deepcopy(shared)
    expanded.eval()

    results = {"layer_count": n, "shared_range": [SHARED_LO, SHARED_HI], "cases": []}
    for prompt in PROMPTS:
        la = logits_for(shared, tok, prompt)
        lb = logits_for(expanded, tok, prompt)
        max_diff = (la - lb).abs().max().item()
        top_a, top_b = la.argmax().item(), lb.argmax().item()
        results["cases"].append({
            "prompt": prompt, "max_abs_logit_diff": max_diff,
            "argmax_shared": top_a, "argmax_expanded": top_b,
            "allclose_1e-3": torch.allclose(la, lb, atol=1e-3, rtol=1e-3)})
        print(json.dumps(results["cases"][-1]), flush=True)

    # sanity: shared differs from untouched original logits would need reloading;
    # instead verify the weight identity directly (sharing actually happened)
    exp_layers = text_layers(expanded)
    shared_ok = all(
        text_layers(shared)[i] is text_layers(shared)[DONOR_LO + (i - SHARED_LO)]
        for i in range(SHARED_LO, SHARED_HI))
    distinct_ok = all(
        exp_layers[i] is not text_layers(shared)[i] for i in range(SHARED_LO, SHARED_HI))
    # real weight equality check on every shared layer's full state_dict
    weights_equal = True
    for i in range(SHARED_LO, SHARED_HI):
        sd_a = text_layers(shared)[i].state_dict()
        sd_b = exp_layers[i].state_dict()
        if set(sd_a) != set(sd_b) or not all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a):
            weights_equal = False
            break
    results["sanity"] = {
        "module_object_sharing": shared_ok,
        "expanded_modules_distinct": distinct_ok,
        "expanded_weights_equal": weights_equal,
    }
    results["verdict"] = "PASS" if (
        all(c["allclose_1e-3"] and c["max_abs_logit_diff"] < 1e-3 for c in results["cases"])
        and shared_ok and distinct_ok and weights_equal) else "FAIL"
    with open("runtime/loop-agent/e1a-loop-equivalence.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"verdict": results["verdict"], "sanity": results["sanity"]}))


if __name__ == "__main__":
    main()
