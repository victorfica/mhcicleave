import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

BLOSUM62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1],
    'B': [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1],
    'Z': [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'X': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1],
}


# Meiler et al. 2001 ,  features
PHYSICHE_PROP_1 = {
    "A": [1.28, 0.05, 1.00,  0.31,  6.11, 0.42, 0.23],
    "R": [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    "N": [1.60, 0.13, 2.95, -0.60,  6.52, 0.21, 0.22],
    "D": [1.60, 0.11, 2.78, -0.77,  2.95, 0.25, 0.20],
    "C": [1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
    "Q": [1.56, 0.18, 3.95, -0.22,  5.65, 0.36, 0.25],
    "E": [1.56, 0.15, 3.78, -0.64,  3.09, 0.42, 0.21],
    "G": [0.00, 0.00, 0.00,  0.00,  6.07, 0.13, 0.15],
    "H": [2.99, 0.23, 4.66,  0.13,  7.69, 0.27, 0.30],
    "I": [4.19, 0.19, 4.00,  1.80,  6.04, 0.30, 0.45],
    "L": [2.59, 0.19, 4.00,  1.70,  6.04, 0.39, 0.31],
    "K": [1.89, 0.22, 4.77, -0.99,  9.99, 0.32, 0.27],
    "M": [2.35, 0.22, 4.43,  1.23,  5.71, 0.38, 0.32],
    "F": [2.94, 0.29, 5.89,  1.79,  5.67, 0.30, 0.38],
    "P": [2.67, 0.00, 2.72,  0.72,  6.80, 0.13, 0.34],
    "S": [1.31, 0.06, 1.60, -0.04,  5.70, 0.20, 0.28],
    "T": [3.03, 0.11, 2.60,  0.26,  5.60, 0.21, 0.36],
    "W": [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42],
    "Y": [2.94, 0.30, 6.47,  0.96,  5.66, 0.25, 0.41],
    "V": [3.67, 0.14, 3.00,  1.22,  6.02, 0.27, 0.49],
    "X": [0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00]
}

# CapsNet-MHC
PHYSICHE_PROP_2 = {
    "A": [0.51, 0.169, 0.471, 0.279, 0.141, 0.294, 0, 0.262, 0.512, 0, 0.404],
    "R": [0.667, 0.726, 0.321, 1, 0.905, 0.529, 0.327, 0.169, 0.372, 1, 1],
    "N": [0.745, 0.39, 0.164, 0.658, 0.51, 0.235, 0.14, 0.313, 0.116, 0.065, 0.33],
    "D": [0.745, 0.304, 0.021, 0.793, 0.515, 0.235, 0.14, 0.601, 0.14, 0.956, 0],
    "C": [0.608, 0.314, 0.76, 0.072, 0, 0.559, 0.14, 0.947, 0.907, 0.028, 0.285],
    "Q": [0.667, 0.531, 0.178, 0.649, 0.608, 0.529, 0.14, 0.416, 0.023, 0.068, 0.36],
    "E": [0.667, 0.482, 0.092, 0.883, 0.602, 0.529, 0.14, 0.561, 0.163, 0.96, 0.056],
    "G": [0, 0, 0.275, 0.189, 0.103, 0, 0, 0.24, 0.581, 0, 0.401],
    "H": [0.686, 0.554, 0.326, 0.468, 0.402, 0.529, 0.14, 0.313, 0.581, 0.992, 0.603],
    "I": [1, 0.65, 1, 0, 0.083, 0.824, 0.308, 0.424, 0.93, 0.003, 0.407],
    "L": [0.961, 0.65, 0.734, 0.081, 0.138, 0.824, 0.308, 0.463, 0.907, 0.003, 0.402],
    "K": [0.667, 0.692, 0, 0.568, 1, 0.529, 0.327, 0.313, 0, 0.952, 0.872],
    "M": [0.765, 0.612, 0.603, 0.171, 0.206, 0.765, 0.308, 0.405, 0.814, 0.028, 0.372],
    "F": [0.686, 0.772, 0.665, 0, 0.114, 0.853, 0.682, 0.462, 1, 0.007, 0.339],
    "P": [0.353, 0.372, 0.012, 0.198, 0.411, 0.588, 0.271, 0, 0.302, 0.03, 0.442],
    "S": [0.52, 0.172, 0.155, 0.477, 0.303, 0.206, 0, 0.24, 0.419, 0.032, 0.364],
    "T": [0.49, 0.349, 0.256, 0.523, 0.337, 0.235, 0.14, 0.313, 0.419, 0.032, 0.362],
    "W": [0.686, 1, 0.681, 0.207, 0.219, 1, 1, 0.537, 0.674, 0.04, 0.39],
    "Y": [0.686, 0.796, 0.591, 0.477, 0.454, 0.853, 0.682, 1, 0.419, 0.031, 0.362],
    "V": [0.745, 0.487, 0.859, 0.036, 0.094, 0.647, 0.234, 0.369, 0.674, 0.003, 0.399],
}


def blosum62_encode(seq, max_len):
    # out : (23, max_len)
    encoding = np.zeros((23, max_len))

    for i, aa in enumerate(seq):
        # if it is gap or missing, all channels will be zero
        if aa.upper() in BLOSUM62:
            encoding[:, i] = np.array(BLOSUM62[aa.upper()])
    return encoding


def physic_encode(seq, max_len, prop=1):
    # out : (prop_len, max_len)
    PROPERTIES = None
    encoding = None
    if prop == 1:
        # use Meiler's physicochemical features
        encoding = np.zeros((7, max_len))
        PROPERTIES = PHYSICHE_PROP_1
    elif prop == 2:
        # use CapsNet-MHC  features
        encoding = np.zeros((11, max_len))
        PROPERTIES = PHYSICHE_PROP_1

    for i, aa in enumerate(seq):
        # if it is gap or missing, all channels will be zero
        if aa.upper() in PROPERTIES:
            encoding[:, i] = np.array(PROPERTIES[aa.upper()])

    return encoding

batch_size = 64

# Function to process each batch
def blosum62_batch(batch):
    sequences = batch["cleavage_window"]
    max_len = max(len(seq) for seq in sequences)

    # Initialize tensors for the batch
    batch_encoding = np.zeros((len(sequences), 23, max_len))

    for i, seq in enumerate(sequences):
        encoding = blosum62_encode(seq, max_len)
        batch_encoding[i] = encoding

    return {"encoded_sequences": batch_encoding}

# Function to process each batch
def physic_batch(batch, prop=1):
    sequences = batch["cleavage_window"]
    max_len = max(len(seq) for seq in sequences)

    prop_dim = None
    if prop==1:
        # use Meiler's physicochemical features
        prop_dim = 7
    elif prop==2:
        prop_dim = 11
    # Initialize tensors for the batch
    batch_encoding = np.zeros((len(sequences), prop_dim, max_len))

    for i, seq in enumerate(sequences):
        encoding = physic_encode(seq, max_len, prop=prop)
        batch_encoding[i] = encoding

    return {"encoded_sequences": batch_encoding}


class Embedder:

    def __init__(self, name: str = "facebook/esm2_t33_650M_UR50D", device='cpu'):

        ALLOWED_MODELS = ['Rostlab/prot_bert', 'Rostlab/prot_t5_xl_uniref50', 'facebook/esm2_t33_650M_UR50D', 'onehot']
        assert name in ALLOWED_MODELS, f"This function is only built for {ALLOWED_MODELS}, get {name}"

        self.device = device
        print('Initializing protein sequence embedder ... ', end='')
        # logging.info(f"Loading ESM-2 model ({name})...")

        if name == 'Rostlab/prot_bert':
            self.featureLen = 1024
            self.tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=False)
            self.model = BertModel.from_pretrained(name).to(self.device).eval()
            self.encode = self.encode_prot

        elif name == 'Rostlab/prot_t5_xl_uniref50':
            self.featureLen = 1024
            self.tokenizer = T5Tokenizer.from_pretrained(name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(name).to(self.device).eval()
            self.encode = self.encode_prot

        elif name == 'facebook/esm2_t33_650M_UR50D':
            self.featureLen = 1280
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name).to(self.device).eval()
            self.encode = self.encode_esm

        elif name == 'onehot':
            self.featureLen = 21
            self.encode = self.encode_onehot

    def encode_esm(self, sequences, emb_type = 'concat', batch_size = 32, num_workers = 4, dataset_key='window'):

        if emb_type not in ["avg", "concat"]:
            print("Computing per-residue embedding by default. Other emb_type = 'avg' or 'concat'")

        pipe = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer, framework='pt')
        key_dataset = KeyDataset(sequences, dataset_key)

        embeddings_list = []
        for out in tqdm(pipe(key_dataset, batch_size=batch_size, return_tensors=True, num_workers=num_workers, shuffle=False),
                        total=len(sequences), desc='Generating Embeddings'):
            embeddings = out[:, 1:-1, :]
            if emb_type == 'concat':
                emb = torch.flatten(embeddings, start_dim=1)
            elif emb_type == 'avg':
                emb = torch.mean(embeddings, dim=1)
            else:
                emb = embeddings
            embeddings_list.append(emb)
        embeddings_result = torch.stack(embeddings_list)
        # del self.model

        return embeddings_result