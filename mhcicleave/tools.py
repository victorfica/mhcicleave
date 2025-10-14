import logging
import numpy as np
from datasets import Dataset
from pathlib import Path

# Configura el logging si no lo has hecho ya en tu script principal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prepare_npz_dataset(
    filepath: Path | str,
    label_key: str = 'labels',
    embedding_key: str = 'embeddings'
) -> Dataset:

    """Carga datos desde .npz, crea y prepara un Dataset de Hugging Face."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    logging.info(f"Cargando y preparando dataset desde: {filepath}")

    try:
        data_npz = np.load(filepath, allow_pickle=True) # Necesario por si embeddings es dtype=object
        labels = data_npz[label_key]
        embeddings = data_npz[embedding_key]
        # Nota: Podría ser necesario convertir embeddings a lista si es object array y from_dict falla
        # if embeddings.dtype == object: embeddings = list(embeddings)
        hf_dataset = Dataset.from_dict({'label': labels, 'embeddings': embeddings})
        if hasattr(data_npz, 'close'): data_npz.close() # Cerrar archivo .npz
    except KeyError as e:
        logging.error(f"Clave {e} no encontrada en {filepath}. Claves: {list(data_npz.keys())}")
        raise
    except Exception as e:
        logging.error(f"Error cargando {filepath}: {e}")
        raise

    # Preparar dataset para PyTorch
    hf_dataset = hf_dataset.with_format("torch")
    hf_dataset = hf_dataset.rename_columns({'label': 'sequence_labels', 'embeddings': 'encoded_sequences'})
    try:
        # Codificar etiquetas si no están ya codificadas
        if 'ClassLabel' not in str(hf_dataset.features['sequence_labels']):
             hf_dataset = hf_dataset.class_encode_column("sequence_labels")
    except ValueError as e:
         logging.warning(f"No se pudo codificar 'sequence_labels' (¿ya codificada?): {e}")

    logging.info(f"Dataset desde {filepath} preparado con {len(hf_dataset)} ejemplos.")
    return hf_dataset