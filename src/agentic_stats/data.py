"""Synthetic R&D design-of-experiments dataset.

Simulated cosmetic-formulation screening: 3 formulations x 4 dose levels x
3 operators, replicated over 12 pilot batches. Response is continuous (assay
signal) with a real batch-to-batch random effect, so mixed-model tools have
something honest to recover.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
FORMULATION_EFFECT = {"A": 0.0, "B": 3.5, "C": -1.5}
DOSE_SLOPE = 0.22
BATCH_SD = 2.0
RESIDUAL_SD = 1.5

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "doe_experiment.csv"


def generate_doe(
    seed: int = SEED,
    formulations: int = 3,
    doses: int = 4,
    operators: int = 3,
    batches: int = 12,
    replicates: int = 4,
) -> pd.DataFrame:
    """Return the full factorial design as a tidy dataframe."""
    rng = np.random.default_rng(seed)
    levels = list(FORMULATION_EFFECT)[:formulations]
    dose_levels = np.linspace(0, 15, doses)
    ops = [f"Op{i + 1}" for i in range(operators)]
    batch_effects = {f"B{i + 1:02d}": rng.normal(0, BATCH_SD) for i in range(batches)}

    rows: list[dict[str, object]] = []
    for batch, batch_effect in batch_effects.items():
        for formulation in levels:
            for dose in dose_levels:
                for operator in ops:
                    for replicate in range(1, replicates + 1):
                        signal = (
                            20.0
                            + FORMULATION_EFFECT[formulation]
                            + DOSE_SLOPE * dose
                            + batch_effect
                            + rng.normal(0, RESIDUAL_SD)
                        )
                        rows.append(
                            {
                                "batch": batch,
                                "formulation": formulation,
                                "dose": float(dose),
                                "operator": operator,
                                "replicate": replicate,
                                "assay_signal": round(float(signal), 3),
                            }
                        )
    return pd.DataFrame(rows)


def write_doe(path: Path = DATA_PATH, seed: int = SEED) -> Path:
    """Write the dataset to `path` as CSV and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_doe(seed=seed).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic DOE dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    written = write_doe(args.out, args.seed)
    df = pd.read_csv(written)
    print(f"{written} -> {len(df)} rows x {len(df.columns)} columns")


if __name__ == "__main__":
    main()
