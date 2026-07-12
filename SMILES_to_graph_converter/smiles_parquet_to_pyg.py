#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.smiles import from_rdmol
from tqdm import tqdm

LABELS = ["dock", "solubility", "sa", "qed", "similarity_to_topK"]

NODE_FEATURES = [
    "atomic_num",
    "chirality",
    "degree",
    "formal_charge",
    "num_hs",
    "num_radical_electrons",
    "hybridization",
    "is_aromatic",
    "is_in_ring",
]

EDGE_FEATURES = [
    "bond_type",
    "stereo",
    "is_conjugated",
]


def make_graph(row_id, smiles, labels, dock_raw, target):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, (row_id, smiles, "MolFromSmiles returned None")

        graph = from_rdmol(mol)

        # Five objectives
        graph.y = torch.tensor([labels], dtype=torch.float32)

        # Preserve the original Parquet row mapping
        graph.row_id = torch.tensor([row_id], dtype=torch.long)
        graph.dock_raw = torch.tensor([dock_raw], dtype=torch.float32)
        graph.smiles = smiles
        graph.target = target
        graph.num_nodes = graph.x.size(0)

        return graph, None
    except Exception as exc:
        return None, (
            row_id,
            smiles,
            f"{type(exc).__name__}: {exc}",
        )


def write_manifest(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows for a smoke test",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()):
        raise SystemExit(
            f"Refusing to write into non-empty directory: {output_dir}"
        )

    parquet = pq.ParquetFile(input_path)
    required = ["smiles", "dock_raw", *LABELS]

    missing = sorted(set(required) - set(parquet.schema_arrow.names))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    limit = parquet.metadata.num_rows
    if args.max_rows is not None:
        limit = min(limit, args.max_rows)

    manifest = {
        "source_parquet": str(input_path),
        "target": args.target,
        "input_rows_requested": limit,
        "labels": LABELS,
        "node_features_x": NODE_FEATURES,
        "edge_features_edge_attr": EDGE_FEATURES,
        "directed_edges": True,
        "shards": [],
    }

    invalid = []
    read_rows = 0
    valid_graphs = 0
    row_base = 0
    shard_id = 0

    batches = parquet.iter_batches(
        batch_size=args.batch_size,
        columns=required,
    )
    progress = tqdm(total=limit, unit="molecule", desc=args.target)

    for batch in batches:
        remaining = limit - read_rows
        if remaining <= 0:
            break

        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)

        frame = batch.to_pandas()
        graphs = []

        for i, row in enumerate(frame.itertuples(index=False)):
            graph, error = make_graph(
                row_base + i,
                str(row.smiles),
                [float(getattr(row, col)) for col in LABELS],
                float(row.dock_raw),
                args.target,
            )

            if graph is not None:
                graphs.append(graph)
            else:
                invalid.append(error)

        shard_name = f"graphs_{shard_id:05d}.pt"

        if graphs:
            InMemoryDataset.save(
                graphs,
                output_dir / shard_name,
            )
        else:
            shard_name = None

        manifest["shards"].append({
            "file": shard_name,
            "first_input_row": row_base,
            "input_rows": len(frame),
            "valid_graphs": len(graphs),
        })

        read_rows += len(frame)
        valid_graphs += len(graphs)
        row_base += len(frame)
        shard_id += 1
        progress.update(len(frame))

        manifest["input_rows_read"] = read_rows
        manifest["valid_graphs"] = valid_graphs
        manifest["invalid_smiles"] = len(invalid)
        write_manifest(output_dir / "manifest.json", manifest)

    progress.close()

    with (output_dir / "invalid_smiles.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "smiles", "error"])
        writer.writerows(invalid)

    write_manifest(output_dir / "manifest.json", manifest)

    print(json.dumps({
        "target": args.target,
        "input_rows": read_rows,
        "valid_graphs": valid_graphs,
        "invalid_smiles": len(invalid),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()