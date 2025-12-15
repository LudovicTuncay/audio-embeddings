import h5py
import os

def inspect_h5():
    # Path from previous context
    h5_path = "/media/ltuncay/Shared-4TB/dev/audio-embeddings/data/AudioSet/balanced_train_soxrhq.h5"
    
    if not os.path.exists(h5_path):
        print(f"File not found: {h5_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        print("Keys:", list(f.keys()))
        if 'waveform' in f:
            print("Waveform shape:", f['waveform'].shape)
            print("Waveform attrs:", dict(f['waveform'].attrs))
        
        # Check for global attributes
        print("Global attrs:", dict(f.attrs))

if __name__ == "__main__":
    inspect_h5()
