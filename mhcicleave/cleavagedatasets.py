import torch
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO


def get_peptide_window(sequence, position, upstream=6, downstream=4, pad_str='X'):
    """
    Returns a window of amino acids around a specified position, handling padding at the sequence ends.

    The residue at the given 'position' is considered the cleavage site and is the
    last residue in the upstream part of the window.

    Args:
        sequence (str): The full protein sequence.
        position (int): The 0-based index of the cleavage site residue.
        upstream (int): Number of residues to include in the upstream segment (including the site).
        downstream (int): Number of residues to include in the downstream segment.
        pad_str (str): The character to use for padding.

    Returns:
        str: The resulting peptide window.
    """
    seq_len = len(sequence)

    if not (0 <= position < seq_len):
        return None
    # The start of the window is 'upstream' residues before the position
    start_pos = position - upstream + 1
    # The end of the window is 'downstream' residues after the position
    end_pos = position + 1 + downstream

    # Take the slice from the sequence
    # This handles cases where start_pos is negative or end_pos is beyond seq_len
    window_slice = sequence[max(0, start_pos): min(seq_len, end_pos)]

    # Calculate how much padding is needed on each side
    pad_left = max(0, -start_pos)
    pad_right = max(0, end_pos - seq_len)

    # Construct the final padded window
    final_window = (pad_str * pad_left) + window_slice + (pad_str * pad_right)

    return final_window


class MultiFastaWindowDataset(Dataset):
    def __init__(self, fasta_file, upstream=6, downstream=4):
        super().__init__()
        self.upstream = upstream
        self.downstream = downstream

        # Parse fasta file into a dictionary in memory {ID:SeqRecord}
        self.records = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

        # iterates to get all possible cleavage sites
        self.index = []
        for record in self.records.values():
            for i in range(len(record.seq)):
                self.index.append((record.id, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        prot_name, position = self.index[idx]
        # acces in-memory dictionary of biopython
        sequence = str(self.records[prot_name].seq)
        window = get_peptide_window(
            sequence, position=position, upstream=self.upstream, downstream=self.downstream
        )
        return {"prot_id": prot_name, "position": position + 1, "window": window}


# 1. Dataset for Training/Validation (Peptides + Labels)
class PeptideDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        peptide = self.data.iloc[idx]['peptide']
        label = self.data.iloc[idx]['label']
        return peptide, torch.tensor(label, dtype=torch.float32)


# 2. DataModule to handle loading and batching
class CleavageDataModule(pl.LightningDataModule):
    def __init__(self, train_file=None, val_file=None, predict_file=None,
                 batch_size=64, num_workers=4,
                 upstream=6, downstream=4):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.predict_file = predict_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.upstream = upstream
        self.downstream = downstream

    def setup(self, stage: str):
        # This is the key part! It loads data based on the task.
        if stage == 'fit':
            # Used for trainer.fit()
            self.train_dataset = PeptideDataset(self.train_file)
            self.val_dataset = PeptideDataset(self.val_file)

        if stage == 'predict':
            # Used for trainer.predict()
            self.predict_dataset = MultiFastaWindowDataset(
                fasta_file=self.predict_file,
                upstream=self.upstream,
                downstream=self.downstream
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)