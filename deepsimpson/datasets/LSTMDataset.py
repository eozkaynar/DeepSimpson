import torch
import pandas as pd
import numpy as np
import os

class Dataset_lstm(torch.utils.data.Dataset):
    def __init__(self, root=None, split="train", max_len = 21, mean = 0., std = 1.):
        super().__init__()
        self.root    = root
        self.split   = split.upper()
        self.mean    = mean
        self.std     = std
        self.samples = [] 
        # Read csv
        file_list_path  = os.path.join(self.root, "fixed_frame_ed_es.csv")
        data            = pd.read_csv(file_list_path)

        # Normalize the 'Split' column to uppercase
        data["Split"]   = data["Split"].str.upper()

        # Filter by dataset split (train/val/test/all)
        if self.split != "ALL":
            data        = data[data["Split"] == self.split]


        for filename, group in data.groupby("Filename"):
            # Split by type
            major_df = group[group["Type"] == "Major Axis"].sort_values("Frame")
            disc_df  = group[group["Type"] == "Simpson's Disc"].sort_values(["Frame", "Length"])

            frames = []

            for _, major_row in major_df.iterrows():
                frame_num = major_row["Frame"]

                # Simpson's Discs belonging to this frame
    
                # Major Axis length
                major_value = major_row["Length"] if pd.notnull(major_row["Length"]) else 0.0
                # Pad 20 Simpson's Discs to fixed length
                discs_fixed = np.zeros(20, dtype=np.float32)
                discs_actual = disc_df[disc_df["Frame"] == frame_num]["Length"].values[:20]
                discs_fixed[:len(discs_actual)] = discs_actual

                # Concatenate: 20 disc + 1 major = (21,)
                frame_vec = np.concatenate([discs_fixed, [major_value]])
                frames.append(frame_vec)

    
            sequence = np.stack(frames, axis=0)  # shape: (20, 21)

            ef = group["EF"].values[0]
            self.samples.append((filename, sequence, ef))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, sequence, ef = self.samples[idx]
        # ef_normalized = (ef - self.mean) / self.std
        ef_normalized = ef / 100 
        return (filename,  torch.tensor(sequence, dtype=torch.float32), torch.tensor(ef_normalized, dtype=torch.float32))
