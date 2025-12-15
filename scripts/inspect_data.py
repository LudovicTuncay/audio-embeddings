import h5py
import pandas as pd
import os

data_dir = "data/AudioSet"
h5_path = os.path.join(data_dir, "balanced_train_soxrhq.h5")
csv_path = os.path.join(data_dir, "silent_files_balanced_train_soxrhq.csv")

print(f"Inspecting {h5_path}...")
try:
    with h5py.File(h5_path, 'r') as f:
        print("Keys:", list(f.keys()))
        # Inspect the first few items if it's a group or dataset
        for key in list(f.keys())[:5]:
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"Dataset {key}: shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, h5py.Group):
                print(f"Group {key}: keys={list(item.keys())}")
                # Go one level deeper
                for subkey in list(item.keys())[:3]:
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Dataset):
                        print(f"  Dataset {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
except Exception as e:
    print(f"Error reading HDF5: {e}")

print(f"\nInspecting {csv_path}...")
try:
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.columns)
except Exception as e:
    print(f"Error reading CSV: {e}")
