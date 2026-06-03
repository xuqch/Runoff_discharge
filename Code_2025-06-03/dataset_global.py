from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from build_h5 import basin_h5_path
from utils import load_scalers


class PerBasinBlockH5Dataset(Dataset):
    @staticmethod
    def _block_has_any_valid_target(q_valid_block: np.ndarray) -> bool:
        q_valid_arr = np.asarray(q_valid_block)
        if q_valid_arr.size == 0:
            return False
        return bool(np.any(q_valid_arr.astype(bool)))

    def __init__(
        self,
        h5_path: str | Path,
        scalers_path: str | Path,
        seq_len: int,
        basins: Optional[Sequence[str]] = None,
        target_block_size: int = 512,
        target_block_stride: Optional[int] = None,
        drop_last_block: bool = False,
    ) -> None:
        self.h5_dir = Path(h5_path)
        self.q_scaler = load_scalers(str(scalers_path))["q"]
        self.seq_len = max(1, int(seq_len))
        self.target_block_size = max(1, int(target_block_size))
        self.target_block_stride = max(1, int(target_block_stride or target_block_size))
        self.drop_last_block = bool(drop_last_block)
        self._h5_handles: Dict[str, h5py.File] = {}

        self.samples: List[Dict[str, int | str | Path]] = []
        self.sample_loads: List[int] = []
        self.block_metadata = self.samples
        # NaN qobs are kept on the original target timeline and are masked later by q_valid in loss/metrics.
        # The dataset does not compress away invalid target timesteps, but it drops blocks whose q_valid is all False.

        if not self.h5_dir.exists():
            raise FileNotFoundError(f"H5 directory not found: {self.h5_dir}")
        if not self.h5_dir.is_dir():
            raise NotADirectoryError(f"Expected per-basin H5 directory, got file: {self.h5_dir}")

        # Training/validation load per-basin H5 strictly by basin table order.
        # Each basin must map to <h5_path>/<basin_id>.h5; no separate index file is required.
        self.missing_h5_basins: List[str] = []
        if basins is None:
            raise ValueError(
                "PerBasinBlockH5Dataset requires basin ids from basins_file; "
                "pass the basin_id sequence via the 'basins' argument."
            )
        basin_files = []
        seen_basins = set()
        for basin in basins:
            basin_id = str(basin).strip()
            if not basin_id or basin_id in seen_basins:
                continue
            seen_basins.add(basin_id)
            basin_files.append((basin_id, basin_h5_path(self.h5_dir, basin_id)))

        for basin_id, h5_file in basin_files:
            if not h5_file.exists():
                self.missing_h5_basins.append(str(basin_id))
                continue
            with h5py.File(h5_file, "r") as f:
                target_idx = f["target_idx"][:].astype(np.int64)
                q_valid = f["target_valid"][:].astype(bool)
                p_count = int(f["x_dyn_base"].shape[0])

                n_targets = int(target_idx.shape[0])
                if n_targets <= 0:
                    continue

                for block_start in range(0, n_targets, self.target_block_stride):
                    block_end = min(block_start + self.target_block_size, n_targets)
                    if block_end <= block_start:
                        continue
                    if self.drop_last_block and (block_end - block_start) < self.target_block_size:
                        continue

                    block_target_idx = target_idx[block_start:block_end]
                    if block_target_idx.size == 0:
                        continue
                    q_valid_block = q_valid[block_start:block_end]
                    if not self._block_has_any_valid_target(q_valid_block):
                        continue
                    seq_start = max(0, int(block_target_idx[0]) - (self.seq_len - 1))
                    seq_end = int(block_target_idx[-1]) + 1
                    prefix_len = max(1, seq_end - seq_start)
                    approx_load = max(1, p_count * prefix_len)

                    self.samples.append(
                        {
                            "basin_id": basin_id,
                            "h5_file": h5_file,
                            "block_start": int(block_start),
                            "block_end": int(block_end),
                            "seq_start": int(seq_start),
                            "seq_end": int(seq_end),
                            "prefix_len": int(prefix_len),
                            "p_count": int(p_count),
                            "valid_target_count": int(np.asarray(q_valid_block).astype(bool).sum()),
                        }
                    )
                    self.sample_loads.append(int(approx_load))

        if self.missing_h5_basins:
            preview = ", ".join(self.missing_h5_basins[:10])
            suffix = "" if len(self.missing_h5_basins) <= 10 else f", ... (+{len(self.missing_h5_basins) - 10} more)"
            print(
                f"[warn] skipped {len(self.missing_h5_basins)} basin(s) from basins_file because matching H5 files "
                f"were not found under {self.h5_dir}: {preview}{suffix}"
            )

        if not self.samples:
            raise RuntimeError(
                "No valid basin H5 files found. Please check h5_dir and missing_basin_h5s.csv."
            )

    def _get_h5(self, h5_file: Path) -> h5py.File:
        key = str(h5_file)
        handle = self._h5_handles.get(key)
        if handle is None:
            handle = h5py.File(h5_file, "r")
            self._h5_handles[key] = handle
        return handle

    def __del__(self) -> None:
        for handle in getattr(self, "_h5_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        basin_id = str(sample["basin_id"])
        h5_file = Path(sample["h5_file"])
        block_start = int(sample["block_start"])
        block_end = int(sample["block_end"])
        seq_start = int(sample["seq_start"])
        seq_end = int(sample["seq_end"])

        f = self._get_h5(h5_file)
        x_stat_base = f["x_stat_base"][:].astype(np.float32)
        dist_m = f["dist_m"][:].astype(np.float32)
        area_m2 = f["area_m2"][:].astype(np.float32)
        fraction = f["fraction"][:].astype(np.float32)
        x_dyn_base = f["x_dyn_base"][:, seq_start:seq_end, :].astype(np.float32)
        target_idx_block = f["target_idx"][block_start:block_end].astype(np.int64)
        q_true = f["target_data"][block_start:block_end, :].astype(np.float32)
        q_valid = f["target_valid"][block_start:block_end, :].astype(np.uint8)
        if not self._block_has_any_valid_target(q_valid):
            raise RuntimeError(
                f"Encountered an empty-supervision block after dataset filtering: "
                f"basin_id={basin_id}, block_start={block_start}, block_end={block_end}, h5_file={h5_file}"
            )
        q_std_loss = f["q_stds"][block_start:block_end, :].astype(np.float32)
        target_dates = f["target_dates"][block_start:block_end].astype(np.int64)
        target_years = f["target_years"][block_start:block_end].astype(np.int32) if "target_years" in f else None
        target_first = int(target_idx_block[0])
        target_last = int(target_idx_block[-1]) + 1
        precip_mm = f["precip_base"][:, target_first:target_last].astype(np.float32)
        pet_mm = f["pet_base"][:, target_first:target_last].astype(np.float32) if "pet_base" in f else None
        if precip_mm.shape[1] != (block_end - block_start):
            precip_mm = f["precip_base"][:, target_idx_block].astype(np.float32)
        if pet_mm is not None and pet_mm.shape[1] != (block_end - block_start):
            pet_mm = f["pet_base"][:, target_idx_block].astype(np.float32)

        local_target_idx = target_idx_block - seq_start
        target_idx_min = int(local_target_idx.min()) if local_target_idx.size > 0 else -1
        target_idx_max = int(local_target_idx.max()) if local_target_idx.size > 0 else -1
        item = {
            "basin_id": basin_id,
            "sample_id": f"{basin_id}:{block_start}:{block_end}",
            "h5_file": str(h5_file),
            "p_count": int(x_dyn_base.shape[0]),
            "n_count": int(local_target_idx.shape[0]),
            "block_start": int(block_start),
            "block_end": int(block_end),
            "seq_start": int(seq_start),
            "seq_end": int(seq_end),
            "prefix_len": int(seq_end - seq_start),
            "valid_target_count": int(sample.get("valid_target_count", 0)),
            "target_idx_min": target_idx_min,
            "target_idx_max": target_idx_max,
            "x_dyn_base": torch.from_numpy(x_dyn_base),
            "x_stat_base": torch.from_numpy(x_stat_base),
            "target_idx": torch.from_numpy(local_target_idx.astype(np.int64)),
            "q_true": torch.from_numpy(q_true),
            "q_valid": torch.from_numpy(q_valid.astype(np.float32)),
            "q_std_loss": torch.from_numpy(q_std_loss),
            "q_mean_global": torch.tensor(float(np.asarray(self.q_scaler.mean).reshape(-1)[0]), dtype=torch.float32),
            "q_std_global": torch.tensor(float(np.asarray(self.q_scaler.std).reshape(-1)[0]), dtype=torch.float32),
            "target_dates": target_dates,
            "precip_mm": torch.from_numpy(precip_mm),
            "dist_m": torch.from_numpy(dist_m),
            "area_m2": torch.from_numpy(area_m2),
            "fraction": torch.from_numpy(fraction),
        }
        if target_years is not None:
            item["target_years"] = torch.from_numpy(target_years)
        if pet_mm is not None:
            item["pet_mm"] = torch.from_numpy(pet_mm)
        return item


BlockPerBasinH5Dataset = PerBasinBlockH5Dataset


def collate_block_basin_batch(batch: List[Dict]) -> Dict:
    return {
        "basin_meta": batch,
        "q_mean_global": batch[0]["q_mean_global"],
        "q_std_global": batch[0]["q_std_global"],
    }
