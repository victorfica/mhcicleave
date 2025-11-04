import torch
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
import argparse
import logging

from mhcicleave.models import MLP
from mhcicleave.lightning_model import CleavageLightningModule
from mhcicleave.cleavagedatasets import PeptideEmbeddings

pl.seed_everything(42)
# New API for setting TF32 precision (replaces the old function)
#torch.backends.cuda.matmul.fp32_precision = 'tf32'
#torch.backends.cudnn.conv.fp32_precision = 'tf32'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_train(args):
    """
    Main function to run the training pipeline based on command-line arguments.
    """
    #torch.set_float32_matmul_precision('medium')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("--- MHC-I Cleavage Model Training ---")
    print(f"Using device: {device}")

    # 2. Setting up h5py dataset and dataloaders
    print("Loading datasets...")
    train_dataset = PeptideEmbeddings(h5_file=args.train_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    val_dataset = PeptideEmbeddings(h5_file=args.val_file)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print("Datasets loaded.")

    # 2. Define MLP model architecture (must match the trained model)
    seq_len = 10 # 6 upstream + 4 downstream
    esm2_dim = 1280
    input_size_mlp = seq_len * esm2_dim
    model_params_mlp = {
        "input_size": input_size_mlp,
        "hidden_linear1": args.hidden_linear1,
        "hidden_linear2": args.hidden_linear2,
        "out_neurons": 1,
        "dropout": args.dropout,
    }
    mlp_network = MLP(**model_params_mlp)

    # Hiperparámetros para el LightningModule y optimizador
    #learning_rate = 0.0001
    scheduler_step_size = 5  # Ajustado
    scheduler_gamma = 0.1

    # --- 3. Callbacks y Logger ---
    callbacks = [
        # Guarda el mejor modelo basado en val_loss
        ModelCheckpoint(monitor="val_BinaryAUROC", mode="max", filename='best_model-{epoch:02d}-{val_BinaryAUROC:.3f}', save_top_k=1,
                        verbose=True),
        # Detiene el entrenamiento si val_loss no mejora en 'patience' épocas
        EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)
    ]

    # Configura el CSVLogger
    csv_logger = CSVLogger(save_dir="results", name=args.experiment_name)

    # 3. Load the trained Lightning Module
    lightning_model = CleavageLightningModule(
        model=mlp_network,
        lr=args.learning_rate,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma
    )

    # 4. Create DataLoaders
    print("Running prediction...")
    # Instancia el Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=csv_logger,
        precision="bf16-mixed"
    )
    # --- 5. Ejecutar Entrenamiento ---
    print("Iniciando entrenamiento...")
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("Entrenamiento finalizado.")
    print(f"Logs guardados en: {csv_logger.log_dir}")
    print(f"Mejor checkpoint (si se guardó) debería estar en: {os.path.join(csv_logger.log_dir, 'checkpoints')}")

    if not args.test_file:
        print("No test file provided, skipping testing.")
        return
    else:
        # Testing bestmodel
        print("Testing best model...")
        test_dataset = PeptideEmbeddings(h5_file=args.test_file)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
            )
        trainer.test(ckpt_path='best', dataloaders=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Cleavage Prediction Inference.")

    parser.add_argument("--train_file", type=str, required=True, default="c_train.h5", help="Path to the training dataset.")
    parser.add_argument("--val_file", type=str, required=True, default="c_val.h5", help="Path to the validation dataset.")
    parser.add_argument("--test_file", type=str, default=None, help="Path to the test dataset.")
    parser.add_argument("--experiment_name", type=str, default="esm2_MLP_v2", help="Name of the experiment.")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for computation.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the DataLoader.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")

    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--scheduler_step_size", type=int, default=5, help="Step size for the learning rate scheduler.")

    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the MLP model.")
    parser.add_argument("--hidden_linear1", type=int, default=256, help="Number of neurons in the first hidden layer of the MLP model.")
    parser.add_argument("--hidden_linear2", type=int, default=128, help="Number of neurons in the second hidden layer of the MLP model.")

    args = parser.parse_args()
    run_train(args)