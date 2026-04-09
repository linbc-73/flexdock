import os
import argparse
from collections import defaultdict
from functools import partial

import prody as pr
import pandas as pd


def get_sequences_from_pdbfile(file_path, parser=None):
    parsed_protein = pr.parsePDB(file_path)

    sequence = None
    for i, chain in enumerate(parsed_protein.iterChains()):
        if chain.ca is not None:
            if sequence is None:
                sequence = chain.ca.getSequence()
            else:
                sequence += ":" + chain.ca.getSequence()
    return sequence


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="pdbbind", help="Dataset")
    parser.add_argument("--file_identifier", type=str)
    parser.add_argument("--max_complexes", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--quiet_parse", action="store_true")
    parser.add_argument("--use_prody", action="store_true")
    parser.add_argument("--id_list", type=str, default=None, help="Optional: path to a txt file with pdb ids to process, one per line")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output csv files. If not set, save to current directory.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.use_prody:
        prot_parser = None
    else:
        from Bio.PDB import PDBParser

        prot_parser = PDBParser(QUIET=args.quiet_parse)

    sequence_fn = partial(get_sequences_from_pdbfile, parser=prot_parser)


    if args.id_list is not None:
        with open(args.id_list) as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
    else:
        pdb_ids = os.listdir(args.data_dir)
    if args.max_complexes is not None:
        pdb_ids = pdb_ids[: args.max_complexes]

    sharded_idxs = list(range(len(pdb_ids) // args.shard_size + 1))
    print(sharded_idxs)

    for shard_idx in sharded_idxs:
        shard_ids = pdb_ids[
            args.shard_size * shard_idx : args.shard_size * (shard_idx + 1)
        ]
        pdbid_seqs = defaultdict(list)

        for pdb_id in shard_ids:
            filename = f"{args.data_dir}/{pdb_id}/{pdb_id}_{args.file_identifier}"
            if os.path.exists(filename):
                sequence = sequence_fn(filename)
                pdbid_seqs["name"].append(pdb_id)
                pdbid_seqs["seqres"].append(sequence)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_csv = os.path.join(args.output_dir, f"{args.dataset}_{shard_idx}.csv")
        else:
            out_csv = f"{args.dataset}_{shard_idx}.csv"
        print(f"Saving shard {shard_idx} to {out_csv}")
        df = pd.DataFrame.from_dict(pdbid_seqs)
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
