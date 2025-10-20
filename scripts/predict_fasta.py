import torch
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import argparse

# --- Assume these custom classes are in files in the same directory or an installed package ---
# You might need to adjust the import paths based on your project structure.
from mhcicleave.models import MLP
from mhcicleave.lightning_model import CleavageLightningModule
from mhcicleave.cleavagedatasets import MultiFastaWindowDataset
from mhcicleave.encodings import Embedder


def run_inference(args):
    """
    Main function to run the inference pipeline based on command-line arguments.
    """
    # 1. Setup Environment
    torch.set_float32_matmul_precision('medium')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("--- MHC-I Cleavage Site Prediction ---")
    print(f"Using device: {device}")

    # 2. Define MLP model architecture (must match the trained model)
    seq_len = args.upstream + args.downstream
    esm2_dim = 1280  # This is specific to the ESM-2 model
    input_size_mlp = seq_len * esm2_dim
    model_params_mlp = {
        "input_size": input_size_mlp,
        "hidden_linear1": 256,
        "hidden_linear2": 128,
        "out_neurons": 1,
        "dropout": 0.5
    }
    mlp_network = MLP(**model_params_mlp)

    # 3. Load the trained Lightning Module
    print(f"Loading cleavage model from: {args.checkpoint}")
    try:
        model = CleavageLightningModule.load_from_checkpoint(args.checkpoint, model=mlp_network)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    # 4. Prepare Data and Embeddings
    print(f"Loading data from: {args.fasta_file}")
    window_dataset = MultiFastaWindowDataset(
        fasta_file=args.fasta_file,
        upstream=args.upstream,
        downstream=args.downstream
    )

    # IMPORTANT: Keep metadata before creating the embedding-only TensorDataset
    windowdata_df = pd.DataFrame(list(window_dataset))

    print("Initializing embedder...")
    embedder = Embedder(name=args.pretrain_model, device=device)

    print("Generating embeddings for all windows...")
    # This step can consume a lot of memory for large files
    seq_embeddings = embedder.encode_esm(window_dataset, emb_type="concat", batch_size = args.batch_size_emb,
                                         num_workers = args.num_workers)

    # 5. Create DataLoader and Run Prediction
    embedding_dataset = TensorDataset(seq_embeddings)
    dataloader = DataLoader(
        embedding_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("Running prediction...")
    trainer = pl.Trainer(accelerator='auto', devices=1, logger=False)
    prediction_batches = trainer.predict(model, dataloader)

    # 6. Process and Save Results
    print("Processing and saving results...")
    predictions = torch.cat([batch.flatten() for batch in prediction_batches]).cpu().numpy()

    # Combine predictions with the saved metadata
    windowdata_df['prediction'] = predictions

    # Sort for a clean output file
    windowdata_df.sort_values(by=['prot_id', 'position'], inplace=True)
    windowdata_df.to_csv(args.output_file, index=False)

    print(f"\n✅ Predictions successfully saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Cleavage Prediction Inference.")

    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.ckpt) file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output predictions CSV file.")

    parser.add_argument("--pretrain_model", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="Name of the Hugging Face pretrain model.")
    parser.add_argument("--emb_type", type=str, default="concat", choices=["mean", "max", "concat"],
                        help="Type of embedding to use from the pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for computation.")

    parser.add_argument("--upstream", type=int, default=6, help="Number of amino acids upstream of the cleavage site (including c-terminal).")
    parser.add_argument("--downstream", type=int, default=4, help="Number of amino acids downstream of the cleavage site.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("--batch_size_emb", type=int, default=32, help="Batch size for embedding generation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader.")

    args = parser.parse_args()
    run_inference(args)

