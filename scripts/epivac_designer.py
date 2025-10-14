#!/usr/bin/env python3
import argparse
import pandas as pd
import itertools
import random

import subprocess
import os
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def read_epitope_csv(filepath):
    """Reads epitopes from a CSV file."""
    print(f"INFO: Reading epitopes from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        if 'peptide' not in df.columns:
            raise ValueError("Input CSV must contain a 'peptide' column.")
        epitopes = df['peptide'].tolist()

        if not epitopes:
            raise ValueError("No epitopes found in the provided CSV file.")

        print(f"INFO: Loaded {len(epitopes)} epitopes for construct generation.")
        return epitopes
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found at '{filepath}'.")

def generate_linkers(length: int, total: int, seed = 42, add_linkers = ["EAAY", "GPGP", "EAAK"]):
    """Generates a list of random linkers, excluding Cysteine."""
    print(f"INFO: Generating {total} + {len(add_linkers)} random linkers of length {length}...")
    random.seed(seed)

    AMINOACIDS = "ADEFGHIKLMNPQRSTVWY"
    # Use a set to automatically handle uniqueness
    unique_linkers = set()
    if add_linkers:
        unique_linkers.update(add_linkers)

    # Keep generating until the set reaches the desired total size
    while len(unique_linkers) < total + len(add_linkers):
        linker = "".join(random.choices(AMINOACIDS, k=length))
        unique_linkers.add(linker)

    return list(unique_linkers)

# --- MODULE 3: VACCINE CONSTRUCT ASSEMBLY ---

def generate_constructs(epitopes, linkers, structure, epitopes_must_be_different=True, linkers_must_be_different=False):
    """
    Generates vaccine constructs for any given architecture in a memory-efficient way.

    This function is a generator that yields tuples of (sequence_id, sequence_string).

    Args:
        epitopes: A list of epitope sequences for assembly.
        linkers: A list of linker sequences for assembly.
        structure: A string defining the construct architecture (e.g., 'LELE', 'ELE').
        epitopes_must_be_different: If True, epitopes cannot be repeated.
        linkers_must_be_different: If True, linkers cannot be repeated.

    Yields:
        A tuple containing a unique sequence ID and the full vaccine sequence string.
    """
    if not epitopes or not linkers:
        return

    structure = structure.upper()
    num_e = structure.count('E')
    num_l = structure.count('L')

    # --- 1. Create the appropriate iterator for each component ---

    # Epitope Iterator
    if num_e > 0:
        if epitopes_must_be_different:
            if len(epitopes) < num_e:
                raise ValueError(f"Not enough unique epitopes for structure '{structure}'.")
            epitope_iter = itertools.permutations(epitopes, num_e)
        else:
            epitope_iter = itertools.product(epitopes, repeat=num_e)
    else:
        epitope_iter = [()]  # If no epitopes, provide one empty tuple to iterate over

    # Linker Iterator
    if num_l > 0:
        if linkers_must_be_different:
            if len(linkers) < num_l:
                raise ValueError(f"Not enough unique linkers for structure '{structure}'.")
            linker_iter = itertools.permutations(linkers, num_l)
        else:
            linker_iter = itertools.product(linkers, repeat=num_l)
    else:
        linker_iter = [()]  # If no linkers, provide one empty tuple

    # --- 2. Combine the component iterators and assemble the constructs ---

    i = 0
    # The main product combines all possible epitope sets with all possible linker sets
    for epitope_tuple, linker_tuple in itertools.product(epitope_iter, linker_iter):
        e_idx, l_idx = 0, 0
        result_sequence = []

        # Weave the epitopes and linkers together based on the structure string
        for component_type in structure:
            if component_type == 'E':
                result_sequence.append(epitope_tuple[e_idx])
                e_idx += 1
            elif component_type == 'L':
                result_sequence.append(linker_tuple[l_idx])
                l_idx += 1

        # Format the output ID and sequence
        seq_id = f"construct{i + 1}|{linker_tuple[0]}"
        sequence = "".join(result_sequence)
        i += 1

        yield seq_id, sequence

# --- MODULE 4: OUTPUT WRITING ---
def write_multifasta(sequences, outfile, add_leader = False):
    """Writes sequences from an iterator to a multi-FASTA file."""
    print(f"INFO: Writing sequences to output file '{outfile}'...")
    leader_seq = "CALNN" if add_leader else ""
    count = 0
    with open(outfile, "w") as output_handle:
        for seq_id, seq_string in sequences:
            record = SeqRecord(Seq(leader_seq + seq_string), id=seq_id, description="")
            SeqIO.write(record, output_handle, "fasta-2line")
            count += 1
    print(f"INFO: Successfully wrote {count} constructs.")


# --- MAIN CLI FUNCTION ---
def main():
    """Main function to parse arguments and run the vaccine design pipeline."""
    parser = argparse.ArgumentParser(
        description="A script to assemble multi-epitope vaccine constructs from a list of peptides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Define Arguments ---
    parser.add_argument("input_csv", help="Path to the input CSV file containing a 'peptide' column.")
    parser.add_argument("output_fasta", help="Path for the output multi-FASTA file of constructs.")

    parser.add_argument("-s", "--structure", default='LELE',
                        help="Vaccine construct architecture (e.g., 'LELE', 'ELE', 'LE').")

    parser.add_argument("-nl", "--num-linkers", type=int, default=500, help="Number of random linkers to generate.")
    parser.add_argument("-ll", "--len-linkers", type=int, default=4, help="Length of each random linker.")

    parser.add_argument("--allow-epitope-repetition", action="store_true",
                        help="Allow epitopes to be repeated in a construct.")
    parser.add_argument("--allow-linker-repetition", action="store_true",
                        help="Allow linkers to be repeated in a construct.")

    parser.add_argument("--add-leader", action="store_true",
                        help="Prepend the 'CALNN' leader sequence to each construct.")
    parser.add_argument("--seed", type=int, help="Seed for random linker generation for reproducibility.")

    args = parser.parse_args()

    # --- Pipeline Execution ---
    try:
        # 1. Load epitopes from the provided CSV
        selected_epitopes = read_epitope_csv(args.input_csv)

        # 2. Generate the pool of random linkers
        generated_linkers = generate_linkers(args.len_linkers, args.num_linkers, args.seed)

        # 3. Create the construct generator based on user specifications
        #    Note: The 'must_be_different' flags are the opposite of the 'allow_repetition' flags.
        construct_generator = generate_constructs(
            epitopes=selected_epitopes,
            linkers=generated_linkers,
            structure=args.structure,
            epitopes_must_be_different=not args.allow_epitope_repetition,
            linkers_must_be_different=not args.allow_linker_repetition
        )

        # 4. Write the generated constructs to the output file
        write_multifasta(construct_generator, args.output_fasta, args.add_leader)

    except (ValueError, FileNotFoundError) as e:
        print(f"\nERROR: A critical error occurred: {e}")


if __name__ == '__main__':
    main()
