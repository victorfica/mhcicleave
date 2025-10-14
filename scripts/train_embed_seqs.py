import argparse
import logging
import re
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import (
    EsmModel,
    EsmTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: Path) -> datasets.Dataset:
    if not file_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {file_path}")
    logging.info(f"Loading: {file_path}")
    return load_dataset('csv', data_files=str(file_path), split='train')

def downsample_data(
    dataset: datasets.Dataset, label_col: str, percentage: float, balance_classes: bool, seed: int | None = None
) -> datasets.Dataset:
    if label_col not in dataset.column_names:
        raise ValueError(f"Label column '{label_col}' not found.")
    pos = dataset.filter(lambda x: x[label_col] == 1)
    neg = dataset.filter(lambda x: x[label_col] != 1)
    n_pos, n_neg = len(pos), len(neg)

    if balance_classes:
        logging.info(f"Balancing classes using {n_pos} positive samples.")
        if n_pos == 0 or n_neg == 0:
             logging.warning("Cannot balance with zero samples in one class.")
             return dataset.filter(lambda x: False) # Return empty
        n_neg_target = min(n_pos, n_neg)
        sampled_neg = neg.shuffle(seed=seed).select(range(n_neg_target))
        final_dataset = concatenate_datasets([pos, sampled_neg]).shuffle(seed=seed)
        logging.info(f"Created balanced dataset with {len(final_dataset)} samples.")
    else: # Percentage per class
        if not (0 < percentage <= 100):
            raise ValueError('Percentage must be > 0 and <= 100.')
        if percentage == 100:
            return dataset
        n_pos_target = int(n_pos * (percentage / 100.0))
        n_neg_target = int(n_neg * (percentage / 100.0))
        sampled_pos = pos.shuffle(seed=seed).select(range(n_pos_target)) if n_pos > 0 else pos
        sampled_neg = neg.shuffle(seed=seed).select(range(n_neg_target)) if n_neg > 0 else neg
        final_dataset = concatenate_datasets([sampled_pos, sampled_neg]).shuffle(seed=seed)
        logging.info(f"Created {percentage}% per-class dataset with {len(final_dataset)} samples.")
    return final_dataset

def preprocess_t5(batch: dict, sequence_col: str) -> dict:
    # T5 needs spaces between AAs & rare AA replacement
    sequences = batch[sequence_col]
    processed_seqs = [' '.join(list(re.sub(r'[UZOB]', 'X', seq))) for seq in sequences]
    batch[sequence_col] = processed_seqs
    return batch

def load_model_and_tokenizer(model_name: str, device: torch.device) -> tuple[object, object, str]:
    logging.info(f'Loading model: {model_name} on {device}')
    model_kwargs = {'torch_dtype': torch.float16} if device.type == 'cuda' else {}

    if 'esm' in model_name.lower():
        model_type, tok_cls, mod_cls = 'esm', EsmTokenizer, EsmModel
    elif 't5' in model_name.lower():
        model_type, tok_cls, mod_cls = 't5', T5Tokenizer, T5EncoderModel
    else:
        raise ValueError(f'Unsupported model type: {model_name}')
    if model_type == 't5':
        tokenizer = tok_cls.from_pretrained(model_name, legacy=False)
    else:
        tok_cls.from_pretrained(model_name)
    model = mod_cls.from_pretrained(model_name, **model_kwargs)
    model.to(device); model.eval()
    return model, tokenizer, model_type

def generate_embeddings(
    model: torch.nn.Module, tokenizer: object, dataset: datasets.Dataset, sequence_col: str, label_col: str,
    embedding_type: str, batch_size: int, device: torch.device, is_t5: bool,
) -> dict:
    logging.info(f'Generating {embedding_type} embeddings...')
    if is_t5:
        dataset = dataset.map(lambda b: preprocess_t5(b, sequence_col), batched=True, batch_size=batch_size)

    pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device.index if device.type == 'cuda' else -1, framework='pt')
    embeddings_list = []
    for out in tqdm(pipe(KeyDataset(dataset, sequence_col), batch_size=batch_size, return_tensors=True), total=len(dataset), desc='Generating Embeddings'):
        tensor = None
        try: # Standardize pipeline output extraction
            if isinstance(out, list) and out:
                item = out[0]
                if isinstance(item, list) and item:
                    tensor = item[0]
                elif torch.is_tensor(item):
                    tensor = item
            elif torch.is_tensor(out):
                tensor = out
            if torch.is_tensor(tensor):
                embeddings_list.append(tensor)
            else:
                logging.debug("Skipping item: Could not extract valid tensor.")
        except Exception as e: logging.debug(f"Skipping item error: {e}")

    if not embeddings_list: return {'labels': np.array([]), 'embeddings': np.array([])}

    try:
        embeddings_tensor = torch.cat(embeddings_list, dim=0); del embeddings_list
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        embeddings_tensor = embeddings_tensor[:, 1:-1, :] # Remove BOS/EOS
        if embeddings_tensor.shape[1] == 0:
            raise ValueError("Embeddings empty after token removal.")
        if embedding_type == "avg":
            embeddings_tensor = embeddings_tensor.mean(dim=1)
        elif embedding_type == "concat":
            embeddings_tensor = embeddings_tensor.flatten(start_dim=1)
        embeddings_array = embeddings_tensor.cpu().numpy()
        labels_array = np.array(dataset[label_col])
        if len(labels_array) != embeddings_array.shape[0]:
             logging.warning(f"Label/Embedding count mismatch ({len(labels_array)} vs {embeddings_array.shape[0]}).")
             labels_array = labels_array[:embeddings_array.shape[0]]
        logging.info(f"Generated embeddings shape: {embeddings_array.shape}")
        return {"labels": labels_array, "embeddings": embeddings_array}
    except Exception as e:
        logging.error(f"Tensor processing error: {e}", exc_info=True)
        if 'embeddings_tensor' in locals():
            del embeddings_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'labels': np.array([]), 'embeddings': np.array([])}

def save_embeddings(
    data: dict, output_dir: Path, model_name: str, emb_type: str, input_fname: str, suffix: str
) -> None:
    """Saves labels and embeddings to a compressed .npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split('/')[-1]
    out_fname = f'{input_fname.stem}_{model_short}_{emb_type}_{len(data["labels"])}samples_{suffix}.npz'
    output_path = output_dir / out_fname
    logging.info(f'Saving embeddings to: {output_path}')
    try:
        np.savez_compressed(output_path, **data)
    except Exception as e:
        logging.error(f"Failed to save .npz: {e}"); raise

def main():
    parser = argparse.ArgumentParser(description='Generate protein sequence embeddings.')
    parser.add_argument('--input_csv', type=Path, required=True, help='Input CSV.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory.')
    parser.add_argument('--model_name', type=str, required=True, help='HF model name.')
    parser.add_argument('--embedding_type', type=str, required=True, choices=['residue', 'avg', 'concat'], help='Embedding type.')
    parser.add_argument('--downsample_percentage', type=float, default=100.0, help='Percent per class (0-100). Default: 100.')
    parser.add_argument('--balance_classes', action='store_true', help='Match negative count to positive count.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. Default: 32.')
    parser.add_argument('--sequence_col', type=str, default='cleavage_window', help='Sequence column. Default: cleavage_window.')
    parser.add_argument('--label_col', type=str, default='elution_cleavage', help='Label column. Default: elution_cleavage.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default: 42.')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu). Default: auto.')
    args = parser.parse_args()

    # Setup seed and device
    datasets.disable_caching()
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    model, tokenizer = None, None
    try:
        dataset = load_data(args.input_csv)
        if args.sequence_col not in dataset.column_names:
            raise ValueError(f"Sequence column '{args.sequence_col}' not found.")
        processed_dataset = downsample_data(dataset, args.label_col, args.downsample_percentage, args.balance_classes, args.seed)
        ds_suffix = "balanced" if args.balance_classes else (f"{int(args.downsample_percentage)}pct_per_class" if args.downsample_percentage != 100.0 else "full")
        model, tokenizer, model_type = load_model_and_tokenizer(args.model_name, device)
        embeddings_dict = generate_embeddings(model, tokenizer, processed_dataset, args.sequence_col, args.label_col, args.embedding_type, args.batch_size, device, (model_type == 't5'))
        if embeddings_dict and embeddings_dict['embeddings'].size > 0:
            save_embeddings(embeddings_dict, args.output_dir, args.model_name, args.embedding_type, args.input_csv, ds_suffix)
        else:
            logging.warning("No embeddings generated or saved.")
    except Exception as e:
        logging.error(f'Execution error: {e}', exc_info=False)
    finally:
        del model, tokenizer
        if 'pipe' in locals():
            del pipe
        if device.type == 'cuda':
            torch.cuda.empty_cache(); logging.info('GPU memory cleared.')

if __name__ == '__main__':
    main()