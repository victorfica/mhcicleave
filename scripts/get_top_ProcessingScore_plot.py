import pandas as pd
import numpy as np
import random
from Bio import SeqIO 

# A small constant to prevent log(0) errors
EPSILON = 1e-9

def calculate_cleavage_score(cleavage_preds, segments, core_len = 9):
    """
    Calculates a specialized cleavage score for a generic vaccine construct.

    Args:
        cleavage_preds: A NumPy array of cleavage probabilities for the full sequence.
        segments: A list of tuples, where each tuple is (segment_name, segment_length).
                  Example: [('Leader', 5), ('Linker1', 4), ('Epitope1', 15)]
        core_len: The length of the core binding motif to protect within 'Epitope' segments.

    Returns:
        The final composite cleavage score.
    """
    score_protection = 0.0
    score_processing = 0.0
    current_pos = 0

    for seg_name, seg_len in segments:
        if seg_len == 0:
            continue
        
        # Isolate the predictions for the current segment
        seg_preds = cleavage_preds[current_pos : current_pos + seg_len]
        #print(seg_preds)
        # Apply scoring logic based on segment name
        if 'epitope' in seg_name.lower():
            # --- Epitope Segment: Protect the core, reward flank cleavage ---
            core_offset = (seg_len - core_len) // 2
            core_start_local = core_offset
            core_end_local = core_start_local + core_len
            
            # Penalize cleavage within the core motif
            core_preds = seg_preds[core_start_local : core_end_local - 1]
            score_protection += np.log(1 - core_preds + EPSILON).sum()
            
            # Reward cleavage in the flanking regions
            flank_preds = np.concatenate([
                seg_preds[0 : core_start_local],
                seg_preds[core_end_local - 1 : seg_len]
            ])
            #print(flank_preds)
            if flank_preds.size > 0:
                score_processing += np.log(flank_preds.max() + EPSILON)
        else:
            # --- Other Segments (Linker, Leader, etc.): Reward cleavage ---
            if seg_preds.size > 0:
                score_processing += np.log(seg_preds.max() + EPSILON)
                #score_processing += np.log(seg_preds.sum() + EPSILON)

        # Move to the start of the next segment
        current_pos += seg_len
        
    return score_protection + score_processing


from Bio.SeqUtils.ProtParam import ProteinAnalysis

def analyze_vaccine_candidate(protein_sequence: str):
    """
    Calcula y devuelve las propiedades fisicoquímicas clave de una secuencia de proteína.

    Esta función utiliza Bio.SeqUtils.ProtParam de Biopython, que replica los cálculos
    de la herramienta ProtParam de ExPASy. Es ideal para el primer cribado de
    candidatos a vacunas multiepitópicas.

    Args:
        protein_sequence (str): La secuencia de aminoácidos de la vacuna candidata.

    Returns:
        dict: Un diccionario con las propiedades fisicoquímicas calculadas.
    """
    sequence = protein_sequence.strip().replace("\n", "")
    analysed_seq = ProteinAnalysis(sequence)

    molecular_weight = analysed_seq.molecular_weight()
    isoelectric_point = analysed_seq.isoelectric_point()
    instability_index = analysed_seq.instability_index()
    gravy = analysed_seq.gravy()

    return (molecular_weight,isoelectric_point,instability_index,gravy)


from dna_features_viewer import GraphicFeature, GraphicRecord
import matplotlib.pyplot as plt

color_map = {'CALNN':'#ffcccc','linker1':'#ffcccc','linker2':'#ffcccc','epitopeCD8':'gold',
                 'epitopeCD4':'green','CPP':'#ccccff'}
def get_GraphicFeatures(d):
    
    record_features = list()
    s = 0
    e = 0
    for fname,fseq in d.items(): 
        e += len(fseq)

        gf = GraphicFeature(start=s, end=e, strand=0, color=color_map[fname],label=fname)
        record_features.append(gf)

        s = e
    return record_features

def plot_seqCleavage(dict_construct,cleavage_data, tag="PeptideCD8-CPP"):
    
    sequence = ''.join(dict_construct.values())
    
    features = get_GraphicFeatures(dict_construct)
    record = GraphicRecord(sequence=sequence, features=features)
    
    fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 3), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    record.plot(ax=ax1,figure_width=12,figure_height=2,with_ruler=False)
    record.plot_sequence(ax1,guides_intensity=0.5)

    colormat=np.where(cleavage_data["prediction"]>=0.5,'r','black')
    ax2.bar(cleavage_data["position"] - 1,cleavage_data["prediction"],color=colormat,width=0.7)
    ax2.set_ylim(0,1)
    ax1.set_title(f"Cleavage predictions for {tag}")
    plt.savefig(f'{tag}.png', bbox_inches='tight')






cleavage_df = pd.read_csv("../results/Q76R36_cleavage_results_5000.csv")
fasta_constructs = SeqIO.to_dict(SeqIO.parse("../results/Q76R36_constructs_5000.fasta", "fasta"))

seq_scores = []
for protid, protdf in cleavage_df.groupby("prot_id"):
    segments= [('Epitope1',9),('Linker1', 4), ('Epitope2', 9)]

    pred = protdf.prediction
    vax_process_score = calculate_cleavage_score(pred, segments)
    seq_scores.append((protid,vax_process_score, protdf))
construct_scored = sorted(seq_scores, key=lambda x: x[1],reverse=True)


q95 = pd.DataFrame([s[1] for s in construct_scored]).quantile(0.95)[0]
construct_scored_q95 = list(filter(lambda x: x[1]>=q95,construct_scored))


# plot top 5 
for data in construct_scored_q95[:5]:
    seqname = data[0]
    seqscore = round(data[1],4)
    sequence = str(fasta_constructs[seqname].seq)
    
    constructELE = dict()
    if sequence.startswith("LVVAASIIG"):
        constructELE["epitopeCD4"] = sequence[:9]
        constructELE["linker1"] = sequence[9:9+4]
        constructELE["epitopeCD8"] = sequence[9+4:9+4+9]
    else:
        constructELE["epitopeCD8"] = sequence[:9]
        constructELE["linker1"] = sequence[9:9+4]
        constructELE["epitopeCD4"] = sequence[9+4:9+4+9]
    
    properties = analyze_vaccine_candidate(sequence)
    if properties[2] < 40 and properties[3] <= 1.7:
        print(seqname, sequence)
        plot_seqCleavage(constructELE, data[2], tag=f"{seqname}_ELE")
        print(f"Molecular Weight (Da):{properties[0]:.2f}",
        f"pI: {properties[1]:.2f}",
        f"Instability Index: {properties[2]:.2f}",
        f"GRAVY: {properties[3]:.3f}")