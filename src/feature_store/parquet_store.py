"""Parquet-backed feature store for the build → rank handoff."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetFeatureStore:
    """Tiny wrapper over a single Parquet file. Single-process, no concurrency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df: pd.DataFrame | None = None

    def write(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.path, index=False)
        self._df = df

    def read(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        df = pd.read_parquet(self.path)
        self._df = df
        return df

    def get(self, candidate_id: str) -> dict:
        df = self.read()
        rows = df[df["candidate_id"] == candidate_id]
        if rows.empty:
            raise KeyError(candidate_id)
        return rows.iloc[0].to_dict()

    def merge(self, other: pd.DataFrame, on: str = "candidate_id", how: str = "left") -> pd.DataFrame:
        """Merge additional columns onto the in-memory frame and persist."""
        df = self.read()
        merged = df.merge(other, on=on, how=how)
        self.write(merged)
        return merged

    def __len__(self) -> int:
        return len(self.read())
