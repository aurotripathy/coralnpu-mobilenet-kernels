"""Download google/gemma-3-270m-it and print config + module tree.

Phase 1 scaffolding: confirms the architecture matches the plan and reveals the
exact submodule names we will hook for per-layer activation dumps.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-270m-it"


def main():
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    print("=== config ===")
    print(cfg)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== total params: {n_params:,} ===")

    print("\n=== top-level module tree ===")
    for name, _ in model.named_children():
        print(name)

    print("\n=== one decoder layer submodules ===")
    # Locate the decoder layer list generically.
    for name, module in model.named_modules():
        if name.endswith("layers.0"):
            for sub, _ in module.named_children():
                print(f"  {name}.{sub}")
            break

    print("\n=== param names (first 40) ===")
    for i, (name, p) in enumerate(model.named_parameters()):
        if i >= 40:
            print("  ...")
            break
        print(f"  {name}\t{tuple(p.shape)}\t{p.dtype}")


if __name__ == "__main__":
    main()
