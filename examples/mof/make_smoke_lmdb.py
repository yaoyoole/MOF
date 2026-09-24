"""Build and validate a 100-structure Uni-MOF smoke LMDB on Windows.

The record schema intentionally mirrors the checked-in examples/mof/preprocess.py.
This script does not modify that official file or add project-specific targets.
"""
from __future__ import annotations

import csv
import json
import pickle
import random
import re
import shutil
import sys
from pathlib import Path

import lmdb
import numpy as np
from pymatgen.core import Structure

UNI_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = UNI_ROOT.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "hMOF-10 1039 C2EE23201D-CarbonDioxide-mofdb-version_dc8a0295db"
SMOKE_ROOT = UNI_ROOT / "examples" / "mof"
CIF_ROOT = SMOKE_ROOT / "smoke_cifs"
LMDB_ROOT = SMOKE_ROOT / "smoke_lmdb"
SEED = 42
TARGET_COUNT = 100


def normalize_atoms(atom: str) -> str:
    return re.sub(r"\d+", "", atom)


def parse_official_schema(cif_path: Path) -> tuple[dict, dict]:
    """Use the same Structure.from_file(..., primitive=True) path as official code."""
    structure = Structure.from_file(str(cif_path), primitive=True)
    lattice = structure.lattice
    df = structure.as_dataframe()
    atoms = df["Species"].astype(str).map(normalize_atoms).tolist()
    coordinates = df[["x", "y", "z"]].values.astype(np.float32)
    abc_coordinates = df[["a", "b", "c"]].values.astype(np.float32)
    if len(atoms) != coordinates.shape[0] or len(atoms) != abc_coordinates.shape[0]:
        raise ValueError("atom/coordinate length mismatch")
    payload = {
        "ID": cif_path.stem,
        "atoms": atoms,
        "coordinates": coordinates,
        "abc": lattice.abc,
        "angles": lattice.angles,
        "volume": lattice.volume,
        "lattice_matrix": lattice.matrix,
        "abc_coordinates": abc_coordinates,
    }
    metadata = {
        "source_path": str(cif_path),
        "ID": payload["ID"],
        "atom_count": len(atoms),
        "coordinates_shape": list(coordinates.shape),
        "abc_coordinates_shape": list(abc_coordinates.shape),
        "lattice_matrix_shape": list(np.asarray(lattice.matrix).shape),
        "abc": [float(x) for x in lattice.abc],
        "angles": [float(x) for x in lattice.angles],
        "volume": float(lattice.volume),
    }
    return payload, metadata


def write_lmdb(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMDB: {path}")
    env = lmdb.open(str(path), subdir=False, readonly=False, lock=False, readahead=False, meminit=False, max_readers=1, map_size=int(10e9))
    with env.begin(write=True) as txn:
        for index, record in enumerate(records):
            txn.put(str(index).encode("ascii"), pickle.dumps(record, protocol=-1))
    env.sync()
    env.close()


def finite_array(value, name: str) -> None:
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        raise ValueError(f"NaN/Inf in {name}")


def validate_record(record: dict) -> dict:
    expected = {"ID", "atoms", "coordinates", "abc", "angles", "volume", "lattice_matrix", "abc_coordinates"}
    if set(record) != expected:
        raise ValueError(f"Unexpected record keys: {sorted(record)}")
    coordinates = np.asarray(record["coordinates"])
    abc_coordinates = np.asarray(record["abc_coordinates"])
    lattice_matrix = np.asarray(record["lattice_matrix"])
    if len(record["atoms"]) != len(record["coordinates"]) or len(record["atoms"]) != len(record["abc_coordinates"]):
        raise ValueError("atom count differs from coordinate count")
    if coordinates.shape[1:] != (3,) or abc_coordinates.shape[1:] != (3,) or lattice_matrix.shape != (3, 3):
        raise ValueError("unexpected coordinate/lattice shape")
    for name, value in (("coordinates", coordinates), ("abc_coordinates", abc_coordinates), ("lattice_matrix", lattice_matrix), ("abc", record["abc"]), ("angles", record["angles"]), ("volume", record["volume"])):
        finite_array(value, name)
    return {"keys": sorted(record), "ID": record["ID"], "atom_count": len(record["atoms"]), "coordinates_shape": list(coordinates.shape), "abc_coordinates_shape": list(abc_coordinates.shape), "lattice_matrix_shape": list(lattice_matrix.shape), "coordinates_dtype": str(coordinates.dtype), "abc_coordinates_dtype": str(abc_coordinates.dtype), "lattice_matrix_dtype": str(lattice_matrix.dtype), "abc": [float(x) for x in record["abc"]], "angles": [float(x) for x in record["angles"]], "volume": float(record["volume"]), "finite": True}


def validate_unimat(path: Path, dict_path: Path) -> dict:
    """Exercise Uni-MOF's own LMDBDataset and UniMatTask.load_dataset."""
    sys.path.insert(0, str(UNI_ROOT))
    from unimat.data import LMDBDataset
    from unimat.tasks.unimat import UniMatTask
    from unicore.data import Dictionary
    from types import SimpleNamespace

    raw = LMDBDataset(str(path))
    first = raw[0]
    if hasattr(raw, "env"):
        raw.env.close()
    dictionary = Dictionary.load(str(dict_path))
    args = SimpleNamespace(data=str(path.parent), dict_name=dict_path.name, seed=SEED, remove_hydrogen=False, max_atoms=512, max_seq_len=512, mask_prob=0.15, leave_unmasked_prob=0.05, random_token_prob=0.05, noise_type="uniform", noise=1.0, minkowski_p=2.0, dist_threshold=5.0)
    task = UniMatTask(args, dictionary)
    task.load_dataset(path.stem.replace(".lmdb", ""))
    task.datasets[path.stem.replace(".lmdb", "")].set_epoch(1)
    item = task.datasets[path.stem.replace(".lmdb", "")][0]
    return {"lmdbdataset_count": len(raw), "lmdbdataset_keys": sorted(first), "task_dataset_type": type(task.datasets[path.stem.replace(".lmdb", "")]).__name__, "task_item_loaded": item is not None}


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(SOURCE_ROOT)
    CIF_ROOT.mkdir(parents=True, exist_ok=True)
    LMDB_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = sorted(SOURCE_ROOT.rglob("*.cif"), key=lambda p: str(p).lower())
    selected: list[tuple[Path, dict]] = []
    failures = []
    for source in candidates:
        try:
            payload, meta = parse_official_schema(source)
        except Exception as exc:  # noqa: BLE001 - audit every candidate and continue
            failures.append({"source_path": str(source), "error": f"{type(exc).__name__}: {exc}"})
            continue
        selected.append((source, meta))
        if len(selected) == TARGET_COUNT:
            break
    if len(selected) != TARGET_COUNT:
        raise RuntimeError(f"Only {len(selected)} valid hMOF CIFs found from {len(candidates)} candidates")

    copied = []
    used_names = set()
    for index, (source, meta) in enumerate(selected):
        name = source.name
        if name in used_names:
            name = f"{index:03d}_{name}"
        used_names.add(name)
        target = CIF_ROOT / name
        shutil.copy2(source, target)
        meta = dict(meta)
        meta["copied_path"] = str(target)
        copied.append(meta)

    rng = random.Random(SEED)
    order = list(range(TARGET_COUNT))
    rng.shuffle(order)
    valid_idx = set(order[:10])
    split_records = {"train": [], "valid": []}
    for index, meta in enumerate(copied):
        payload, _ = parse_official_schema(Path(meta["copied_path"]))
        split_records["valid" if index in valid_idx else "train"].append(payload)
    train_path = LMDB_ROOT / "train.lmdb"
    valid_path = LMDB_ROOT / "valid.lmdb"
    if not (train_path.exists() and valid_path.exists()):
        write_lmdb(train_path, split_records["train"])
        write_lmdb(valid_path, split_records["valid"])

    samples = []
    for split, expected in (("train", 90), ("valid", 10)):
        path = LMDB_ROOT / f"{split}.lmdb"
        env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, max_readers=256)
        with env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            if len(keys) != expected:
                raise AssertionError(f"{split} count {len(keys)} != {expected}")
            for key in keys[:3]:
                samples.append({"split": split, "key": key.decode("ascii"), **validate_record(pickle.loads(txn.get(key)))})
        env.close()

    unimat_checks = {split: validate_unimat(LMDB_ROOT / f"{split}.lmdb", SMOKE_ROOT / "dict.txt") for split in ("train", "valid")}
    (LMDB_ROOT / "selection_manifest.csv").write_text("source_path,copied_path,ID,atom_count\n" + "\n".join(f'{m["source_path"]},{m["copied_path"]},{m["ID"]},{m["atom_count"]}' for m in copied) + "\n", encoding="utf-8")
    report = {"source_root": str(SOURCE_ROOT), "candidate_count": len(candidates), "valid_selected": len(selected), "parse_failures_before_target": failures, "seed": SEED, "counts": {"train": 90, "valid": 10}, "record_fields": ["ID", "atoms", "coordinates", "abc", "angles", "volume", "lattice_matrix", "abc_coordinates"], "samples": samples, "unimat_checks": unimat_checks, "official_schema_source": str(UNI_ROOT / "examples/mof/preprocess.py")}
    (LMDB_ROOT / "smoke_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"source_root": str(SOURCE_ROOT), "candidate_count": len(candidates), "valid_selected": len(selected), "train": 90, "valid": 10, "unimat_checks": unimat_checks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
