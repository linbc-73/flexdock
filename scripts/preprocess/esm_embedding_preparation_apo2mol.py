import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from flexdock.data.constants import restype_3to1

biopython_parser = PDBParser()

def get_sequence_from_file(file_path):
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequences = []
    for i, chain in enumerate(structure):
        seq = ""
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue

            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if c_alpha is not None and n is not None and c is not None:
                try:
                    seq += restype_3to1[residue.get_resname()]
                except Exception:
                    seq += "-"
                    print(
                        f"encountered unknown AA: {residue.get_resname()} "
                        f"in the complex {file_path}. Replacing it with a dash - ."
                    )
        sequences.append(seq)
    return sequences

def prepare_files_for_embedding():
    csv_path = "/data/protein/BC_Data/flexdock/examples/inference_apo2mol.csv"
    out_file = "/data/protein/BC_Data/flexdock/data/fasta/prepared_for_esm_apo2mol.fasta"
    
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    sequences = []
    ids = []
    
    for _, row in df.iterrows():
        name = row['pdbid']
        file_path = row['apo_protein_file']
        
        try:
            l = get_sequence_from_file(file_path)

            for i, seq in enumerate(l):
                sequences.append(seq)
                ids.append(f"{name}_chain_{i}")
        except Exception as e:
            print(f"Failed to process {name} due to {e}")

    records = []
    for index, seq in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ""
        records.append(record)

    dirname = os.path.dirname(out_file)
    os.makedirs(dirname, exist_ok=True)
    SeqIO.write(records, out_file, "fasta")

if __name__ == "__main__":
    prepare_files_for_embedding()
