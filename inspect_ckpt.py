import torch
import sys

def inspect_checkpoint(path):
    print(f"Inspecting {path}")
    try:
        ckpt = torch.load(path, weights_only=False)
        print("Keys in checkpoint:", ckpt.keys())
        if "optimizer_states" in ckpt:
            print("Optimizer states found: YES")
        else:
            print("Optimizer states found: NO")
        
        if "state_dict" in ckpt:
            print(f"Model keys: {len(ckpt['state_dict'])} items")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        print("Usage: python inspect_ckpt.py <path_to_ckpt>")
