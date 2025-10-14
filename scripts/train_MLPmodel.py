import logging
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
#from tools import load_prepare_npz_dataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from mhcicleave.models import MLP
from mhcicleave.lightning_model import CleavageLightningModule
import torch # Asegúrate de que torch esté importado

# Configura el logging si no lo has hecho ya en tu script principal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NpzEmbeddingDataset(Dataset):
    """
    PyTorch Dataset para cargar eficientemente embeddings y etiquetas
    desde un archivo .npz usando memory mapping.
    """
    def __init__(self, npz_path: Path | str, label_key: str = 'labels', embedding_key: str = 'embeddings'):
        self.filepath = Path(npz_path)
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Archivo no encontrado: {self.filepath}")

        logging.info(f"Abriendo archivo NPZ con memory mapping: {self.filepath}")
        # Cargar en modo memory map ('r' para lectura)
        # ¡Importante! Mantenemos el objeto NpzFile abierto durante la vida del Dataset
        try:
            self.npz_data = np.load(self.filepath, mmap_mode='r', allow_pickle=True)
        except Exception as e:
             logging.error(f"No se pudo abrir {self.filepath} con mmap: {e}")
             raise

        # Validar y obtener referencias a los arrays mapeados
        if label_key not in self.npz_data:
            raise KeyError(f"Clave '{label_key}' no encontrada en {self.filepath}. Claves: {list(self.npz_data.keys())}")
        if embedding_key not in self.npz_data:
            raise KeyError(f"Clave '{embedding_key}' no encontrada en {self.filepath}. Claves: {list(self.npz_data.keys())}")

        self.labels = self.npz_data[label_key]
        self.embeddings = self.npz_data[embedding_key]

        # Verificar consistencia de longitud
        if len(self.labels) != len(self.embeddings):
            raise ValueError(f"Inconsistencia en longitud de labels ({len(self.labels)}) y embeddings ({len(self.embeddings)}) en {self.filepath}")

        self.is_object_array = (self.embeddings.dtype == object)
        if self.is_object_array:
             logging.info("Embeddings cargados como array de objetos (dtype=object).")


    def __len__(self) -> int:
        """Devuelve el número total de muestras."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtiene una muestra (embedding, label) por índice."""
        label = self.labels[idx]
        embedding = self.embeddings[idx] # Accede solo al elemento idx (mmap lo carga)

        # Convertir a Tensores PyTorch
        label_tensor = torch.tensor(label, dtype=torch.long) # Asumiendo etiquetas enteras para clasificación

        # Si es un array de objetos, el elemento 'embedding' ya es un array NumPy
        if self.is_object_array:
             embedding_tensor = torch.from_numpy(embedding.astype(np.float32)) # Asegurar float32
        else: # Si es un array normal, 'embedding' es una fila/slice
             embedding_tensor = torch.from_numpy(embedding.astype(np.float32)) # Asegurar float32

        # Renombrar implícitamente al devolver la tupla: (embedding, label)
        return embedding_tensor, label_tensor

    # Opcional: método para cerrar el archivo explícitamente si es necesario
    # def close(self):
    #     if hasattr(self.npz_data, 'close'):
    #         self.npz_data.close()
    #         logging.info(f"Archivo NPZ cerrado: {self.filepath}")

if __name__ == '__main__':

    # --- 1. Carga los datasets desde archivos .npz ---
    train_npz_path = Path("../data/encodings/c_train_esm2_t33_650M_UR50D_concat_283900samples_25pct_per_class.npz")
    val_npz_path   = Path("../data/encodings/c_val_esm2_t33_650M_UR50D_concat_35486samples_25pct_per_class.npz")
    #test_npz_path  = Path("../data/encodings/c_test_esm2_t33_650M_UR50D_concat_141950samples_full.npz")

    # --- 2. Carga y prepara cada dataset usando la función ---
    try:
        train_dataset = NpzEmbeddingDataset(train_npz_path)
        val_dataset   = NpzEmbeddingDataset(val_npz_path)
        #test_dataset  = load_prepare_npz_dataset(test_npz_path) # Opcional

        # --- 3. Crea los DataLoaders (sin cambios) ---
        num_workers = 10 # Ajusta según tu sistema
        train_loader = DataLoader(train_dataset, batch_size=512, num_workers=num_workers, shuffle=True, persistent_workers=True if num_workers > 0 else False)
        val_loader = DataLoader(val_dataset, batch_size=512, num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
        # test_loader = DataLoader(test_dataset, batch_size=512, num_workers=num_workers, persistent_workers=True if num_workers > 0 else False) # Si necesitas

        logging.info("DataLoaders creados exitosamente.")

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Verifica las rutas de los archivos .npz.")
    except KeyError as e:
         logging.error(f"Error: {e}. Verifica las claves ('labels', 'embeddings') dentro de los archivos .npz.")
    except Exception as e:
         logging.error(f"Error inesperado durante la carga/preparación: {e}", exc_info=True) # Muestra traceback para errores inesperados


    # Define parámetros comunes y específicos de la arquitectura
    seq_len = 10 # Asegúrate que esto coincide con tus datos
    esm2_dim = 1280
    input_size_mlp = seq_len * esm2_dim

    # Elige qué modelo usar
    # Modelo MLP:
    model_params_mlp = {"input_size": input_size_mlp, "hidden_linear1": 256, "hidden_linear2": 128, "out_neurons": 1, "dropout": 0.5}
    chosen_model = MLP(**model_params_mlp)
    experiment_name = "esm2_MLP_final" # Nombre para el logger

    # Modelo CNN (Ejemplo - ajusta parámetros según tu CNN en models.py):
    # model_params_cnn = {"input_channel": esm2_dim, "out_channel1": 64, ... } # Ajustar!
    # chosen_model = CNN(**model_params_cnn)
    # experiment_name = "esm2_CNN_test"

    # Hiperparámetros para el LightningModule y optimizador
    learning_rate = 0.0001
    scheduler_step_size = 5 # Ajustado
    scheduler_gamma = 0.1

    # --- 3. Callbacks y Logger ---

    callbacks = [
        # Guarda el mejor modelo basado en val_loss
        ModelCheckpoint(monitor="val_loss", mode="min", filename='best_model-{epoch:02d}-{val_loss:.2f}', save_top_k=1, verbose=True),
        # Detiene el entrenamiento si val_loss no mejora en 'patience' épocas
        EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)
    ]

    # Configura el CSVLogger
    csv_logger = CSVLogger(save_dir="csv_logs/", name=experiment_name)
    print(f"CSVLogger configurado. Logs se guardarán en: {csv_logger.log_dir}")

    # --- 4. Instanciar LightningModule y Trainer ---

    # Instancia el LightningModule genérico PASANDO el modelo elegido
    lightning_model = CleavageLightningModule(
        model=chosen_model,
        lr=learning_rate,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma
    )

    # Instancia el Trainer
    trainer = pl.Trainer(
        max_epochs=50, # Aumentado un poco, EarlyStopping controlará el final
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        devices=1,
        callbacks=callbacks,
        logger=csv_logger
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

    # --- 6. (Opcional) Cargar y probar el mejor modelo ---
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # if best_model_path:
    #     print(f"Cargando el mejor modelo desde: {best_model_path}")
    #     best_model = GenericClassifier.load_from_checkpoint(best_model_path)
    #     # Crear test_loader si tienes un conjunto de test separado
    #     # test_results = trainer.test(model=best_model, dataloaders=test_loader)
    #     # print("Resultados del Test:", test_results)
    # else:
    #     print("No se encontró el path al mejor modelo (quizás el entrenamiento fue muy corto o no hubo mejora).")