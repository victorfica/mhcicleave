#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to preprocess a peptide dataset (CSV), apply downsampling/balancing,
and then generate and save ESM-2 embeddings to an HDF5 file.

Output files are automatically named and generated in a specified directory.
"""

import torch
import pandas as pd
import h5py
from pathlib import Path
import argparse

try:
    from mhcicleave.cleavagedatasets import PeptideDataset
    from mhcicleave.encodings import Embedder
except ImportError:
    print("Error: Could not import custom classes 'PeptideDataset' or 'Embedder' from 'mhcicleave'.")
    print("Please ensure the 'mhcicleave' package is installed and accessible.")
    exit(1)


def downsample_data_pd(dataset, label_col, percentage, balance_classes, seed=None):
    """
    Downsamples a pandas DataFrame, with options to balance classes
    or take a percentage of each class.

    Args:
        dataset (pd.DataFrame): The input DataFrame.
        label_col (str): The name of the column to use as the label.
        percentage (float): The percentage (0-100) of each class to keep.
                            Only used if balance_classes is False.
        balance_classes (bool): If True, downsamples the negative class
                                to match the number of positive samples.
        seed (int, optional): The random seed for shuffling and sampling
                              to ensure reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: The downsampled and shuffled DataFrame.
    """
    if label_col not in dataset.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    # Split dataset into positive (1) and negative (not 1)
    pos = dataset[dataset[label_col] == 1]
    neg = dataset[dataset[label_col] != 1]
    n_pos, n_neg = len(pos), len(neg)

    if balance_classes:
        print(f"Balancing classes using {n_pos} positive samples.")
        if n_pos == 0 or n_neg == 0:
            print("Warning: Cannot balance with zero samples in one class.")
            # Return an empty DataFrame with the same columns
            return dataset.iloc[0:0]

        # Sample negative class to match positive
        n_neg_target = min(n_pos, n_neg)
        sampled_neg = neg.sample(n=n_neg_target, random_state=seed)
        final_dataset = pd.concat([pos, sampled_neg])

        # .sample(frac=1) is the equivalent of .shuffle()
        final_dataset = final_dataset.sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)

        print(f"Created balanced dataset with {len(final_dataset)} samples.")

    else:  # Percentage per class
        if not (0 < percentage <= 100):
            raise ValueError('Percentage must be > 0 and <= 100.')
        if percentage == 100:
            print("Using 100% of data, no downsampling applied (still shuffling).")
            # Shuffle nonetheless for consistency
            return dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_pos_target = int(n_pos * (percentage / 100.0))
        n_neg_target = int(n_neg * (percentage / 100.0))
        print(f"Taking {percentage}% of each class: {n_pos_target} positive, {n_neg_target} negative.")

        sampled_pos = pos.sample(n=n_pos_target, random_state=seed) if n_pos > 0 else pos
        sampled_neg = neg.sample(n=n_neg_target, random_state=seed) if n_neg > 0 else neg

        final_dataset = pd.concat([sampled_pos, sampled_neg]).reset_index(drop=True)

        # Shuffle the final combined dataset
        final_dataset = final_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

        print(f"Created {percentage}% per-class dataset with {len(final_dataset)} samples.")

    return final_dataset


def run_preprocessing(args):
    """
    Main function to run the preprocessing and embedding pipeline.
    """
    # 1. Setup Environment
    torch.set_float32_matmul_precision('medium')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("--- Preprocessing and Embedding Generation ---")
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

    # 2. Load and Preprocess Data
    print(f"Loading raw data from: {args.input_file}")
    try:
        dataset_pd = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}")
        return

    print("Preprocessing data (downsampling/balancing)...")
    processed_dataset = downsample_data_pd(
        dataset=dataset_pd,
        label_col=args.label_col,
        percentage=args.downsample_percentage,
        balance_classes=args.balance_classes,
        seed=args.seed
    )

    if processed_dataset.empty:
        print("Processed dataset is empty. Exiting.")
        return

    # --- CONSTRUCT OUTPUT FILENAMES ---
    print("Constructing output paths...")

    # 1. Get base info
    input_stem = Path(args.input_file).stem
    num_samples = len(processed_dataset)

    # 2. Determine dataset suffix
    if args.balance_classes:
        ds_suffix = "balanced"
    elif args.downsample_percentage != 100.0:
        ds_suffix = f"{int(args.downsample_percentage)}pct_per_class"
    else:
        ds_suffix = "full"

    # 3. Build base name for files
    csv_filename_base = f"{input_stem}_{num_samples}samples_{ds_suffix}"

    # 4. Determine short model name for H5 file
    model_name_simple = "model"
    if 'esm2' in args.pretrain_model.lower():
        model_name_simple = 'esm2'
    elif 'protbert' in args.pretrain_model.lower():
        model_name_simple = 'protbert'
    else:
        # Use last part of the path as fallback
        model_name_simple = Path(args.pretrain_model).name

    h5_filename = f"{csv_filename_base}_{model_name_simple}_{args.emb_type}.h5"

    # 5. Create output directories
    output_dir_path = Path(args.output_dir)
    csv_output_dir = output_dir_path / "csv"
    h5_output_dir = output_dir_path / "encodings"

    csv_output_dir.mkdir(parents=True, exist_ok=True)
    h5_output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Define final paths
    processed_output_file = csv_output_dir / f"{csv_filename_base}.csv"
    h5_output_file = h5_output_dir / h5_filename
    # --- END FILENAME LOGIC ---

    print(f"Saving processed dataset to: {processed_output_file}")
    processed_dataset.to_csv(processed_output_file, index=False)

    # 3. Load Data for Embedding
    print(f"Loading processed data for embedding from: {processed_output_file}")
    peptides_dataset = PeptideDataset(
        file_path=processed_output_file,
        seq_col=args.seq_col,
        label_col=args.label_col
    )
    seq_labels = peptides_dataset.data[peptides_dataset.label_col].values
    # 4. Generate Embeddings
    print(f"Initializing embedder with model: {args.pretrain_model}")
    embedder = Embedder(name=args.pretrain_model, device=device)

    print("Generating embeddings for all sequences...")
    seq_embeddings = embedder.encode_esm(
        peptides_dataset,
        emb_type=args.emb_type,
        batch_size=args.batch_size_emb,
        num_workers=args.num_workers
    ).cpu().numpy()  # Move to CPU and convert to numpy for saving
    del peptides_dataset
    print(f"Generated embeddings with shape: {seq_embeddings.shape}")

    # 5. Save Embeddings and Labels to HDF5
    print(f"Saving embeddings and labels to: {h5_output_file}")

    with h5py.File(h5_output_file, 'w') as f:
        f.create_dataset('embeddings', data=seq_embeddings)
        f.create_dataset('labels', data=seq_labels)

    print(f"\n✅ Preprocessing and embeddings saved successfully to: {h5_output_file}")
    print(f"✅ Intermediate CSV saved to: {processed_output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess CSV data and generate HDF5 embeddings.")

    # --- File Arguments ---
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the raw input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save the processed 'csv' and 'encodings' subdirectories.")

    # --- Data Arguments ---
    parser.add_argument("--seq_col", type=str, default="cleavage_window",
                        help="Name of the column in the CSV containing sequences/windows.")
    parser.add_argument("--label_col", type=str, default="elution_cleavage",
                        help="Name of the column in the CSV containing the labels.")

    # --- Preprocessing Arguments ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--balance_classes", action='store_true',
                        help="If set, balance classes by downsampling negative to match positive. Overrides --downsample_percentage.")
    parser.add_argument("--downsample_percentage", type=float, default=100.0,
                        help="Percentage (0-100) of each class to keep. Ignored if --balance_classes is set.")

    # --- Embedding Arguments ---
    parser.add_argument("--pretrain_model", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="Name of the Hugging Face pre-trained model (e.g., 'facebook/esm2_t33_650M_UR50D').")
    parser.add_argument("--emb_type", type=str, default="concat", choices=["mean", "max", "concat"],
                        help="Type of embedding to use from the pre-trained model.")
    parser.add_argument("--batch_size_emb", type=int, default=64,
                        help="Batch size for embedding generation.")

    # --- Compute Arguments ---
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for computation ('cuda' or 'cpu').")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for the embedding DataLoader.")

    args = parser.parse_args()

    # Logical check
    if args.balance_classes and args.downsample_percentage != 100.0:
        print(
            f"Warning: --balance_classes is set. The argument --downsample_percentage={args.downsample_percentage} will be ignored.")

    run_preprocessing(args)