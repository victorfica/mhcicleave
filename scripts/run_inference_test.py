import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import argparse
import logging
import os
import pandas as pd

# Assuming these imports are correct based on your original script
from mhcicleave.models import MLP
from mhcicleave.lightning_model import CleavageLightningModule
from mhcicleave.cleavagedatasets import PeptideEmbeddings

pl.seed_everything(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_predict(args):
    """
    Main function to run the prediction pipeline using a trained model checkpoint,
    loading hyperparameters automatically.
    """
    # 1. Device and Basic Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("--- MHC-I Cleavage Model Prediction (Hyperparameters Loaded from Checkpoint) ---")
    print(f"Using device: {device}")
    print(f"Loading checkpoint from: {args.checkpoint_path}")

    # 2. Load the model directly from the checkpoint
    # PyTorch Lightning automatically instantiates the model's __init__ using
    # the saved hyperparameters found in the checkpoint file.
    try:
        model = CleavageLightningModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
        )
        model.eval()  # Set to evaluation mode
        print("Model checkpoint loaded successfully with saved hyperparameters.")

        # Optional: Print the loaded hyperparameters to confirm
        print("\n--- Loaded Hyperparameters ---")
        for key, value in model.hparams.items():
            print(f"- {key}: {value}")
        print("------------------------------\n")

    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        logging.error("Ensure the model's __init__ can accept all saved hyperparameters.")
        return

    # 3. Setting up h5py dataset and dataloader
    print(f"Loading data from: {args.data_file}")
    test_dataset = PeptideEmbeddings(h5_file=args.data_file)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,  # Batch size can be changed for inference
        shuffle=False,
        num_workers=args.num_workers
    )
    print("Dataset loaded.")

    # 4. Create Trainer and Run Prediction
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed"  # Use the same precision as training
    )

    print("Running prediction...")
    # trainer.predict() returns a list of outputs from predict_step
    predictions = trainer.predict(model=model, dataloaders=test_loader)
    print(predictions)
    pd_datatest = pd.DataFrame(test_dataset)


    # 5. Process and Save Results for External Analysis
    print("Prediction finished. Processing results...")


    # Create output directory and filename
    output_dir = os.path.dirname(args.checkpoint_path) if os.path.dirname(args.checkpoint_path) else 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Base filename from the checkpoint name
    base_name = os.path.basename(args.checkpoint_path).replace('.ckpt', '')
    output_filename = os.path.join(output_dir, f'predictions_{base_name}.csv')

    # Save the scores and targets
    results_df = pd.DataFrame({
        'prediction_score': all_scores.flatten(),
        'true_label': all_targets.flatten()
    })

    results_df.to_csv(output_filename, index=False)

    print(f"Prediction scores and targets saved to: {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Cleavage Prediction Inference.")

    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the dataset for prediction (e.g., test.h5).")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the best model checkpoint file (*.ckpt).")

    #parser.add_argument("--output_dir", type=str, required=True,help="Directory to save prediction results.")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for computation.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for prediction.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for the DataLoader.")

    args = parser.parse_args()
    run_predict(args)