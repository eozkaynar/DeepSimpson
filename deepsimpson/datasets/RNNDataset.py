import torch
import pandas as pd
import numpy as np
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root=None, split="train", max_len = 21, mean = 0., std = 1.):
        super().__init__()
        self.root    = root
        self.split   = split.upper()
        self.mean    = mean
        self.std     = std
        self.samples = [] 
        # Read csv
        file_list_path  = os.path.join(self.root, "simpsons.csv")
        data            = pd.read_csv(file_list_path)

        # Normalize the 'Split' column to uppercase
        data["Split"]   = data["Split"].str.upper()

        # Filter by dataset split (train/val/test/all)
        if self.split != "ALL":
            data        = data[data["Split"] == self.split]


        for filename, group in data.groupby("Filename"):

            group = group.sort_values(["Phase", "Frame"])

            ed = group[group["Phase"] == "ED"]["Length"].values[:21]
            es = group[group["Phase"] == "ES"]["Length"].values[:21] 

            # Pad ED and ES to max_len = 21
            ed_padded = np.zeros(max_len, dtype=np.float32)
            es_padded = np.zeros(max_len, dtype=np.float32)

            ed_padded[:min(len(ed), max_len)] = ed[:max_len]
            es_padded[:min(len(es), max_len)] = es[:max_len]  


            sequence = np.stack([ed_padded, es_padded], axis=0)
            ef = group["EF"].values[0]

            self.samples.append((filename, sequence, ef))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, sequence, ef = self.samples[idx]
        # ef_normalized = (ef - self.mean) / self.std
        ef_normalized = ef / 100 
        return (filename,  torch.tensor(sequence, dtype=torch.float32), torch.tensor(ef_normalized, dtype=torch.float32))
