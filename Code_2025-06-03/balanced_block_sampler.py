from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator, List

import torch.distributed as dist
from torch.utils.data import Sampler


class BalancedBlockDistributedSampler(Sampler[int]):
    """Rank-aware sampler that balances target blocks by estimated compute load.

    Binding a whole basin to one device often leaves the rank that receives a large
    basin doing much more work while the others wait. The default strategy assigns
    target blocks across ranks, which fits large basins and long time series better.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        bucket_size: int = 16,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("num_replicas must be set when distributed is not initialized.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("rank must be set when distributed is not initialized.")
            rank = dist.get_rank()

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = max(int(bucket_size), self.num_replicas * self.batch_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self._sample_indices = list(range(len(dataset)))
        self._sample_loads = [int(v) for v in getattr(dataset, "sample_loads", [1] * len(dataset))]
        self._rank_partitions = self._partition_by_load()
        self.raw_counts = [len(indices) for indices in self._rank_partitions]
        self.raw_loads = [
            sum(max(1, self._sample_loads[idx]) for idx in indices)
            for indices in self._rank_partitions
        ]
        self.num_samples = self._compute_global_num_samples()

    def _ceil_to_batch(self, n: int) -> int:
        if n <= 0:
            return 0
        return int(math.ceil(n / self.batch_size)) * self.batch_size

    def _floor_to_batch(self, n: int) -> int:
        return (n // self.batch_size) * self.batch_size

    def _compute_global_num_samples(self) -> int:
        counts = [len(indices) for indices in self._rank_partitions]
        if self.drop_last:
            n = min(self._floor_to_batch(count) for count in counts)
        else:
            n = max(self._ceil_to_batch(count) for count in counts)
        if n <= 0:
            raise RuntimeError(
                "BalancedBlockDistributedSampler produced empty local epoch: "
                f"counts={counts}, batch_size={self.batch_size}, drop_last={self.drop_last}"
            )
        return int(n)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _partition_by_load(self) -> List[List[int]]:
        ordered = sorted(self._sample_indices, key=lambda idx: self._sample_loads[idx], reverse=True)
        rank_bins: List[List[int]] = [[] for _ in range(self.num_replicas)]
        rank_loads = [0 for _ in range(self.num_replicas)]

        for idx in ordered:
            target_rank = min(range(self.num_replicas), key=lambda r: rank_loads[r])
            rank_bins[target_rank].append(idx)
            rank_loads[target_rank] += max(1, self._sample_loads[idx])
        return rank_bins

    def _order_rank_indices(self, indices: List[int]) -> List[int]:
        if not indices:
            return []

        rng = random.Random(self.seed + self.epoch)
        if not self.shuffle:
            return self._basin_aware_order(indices)

        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        rng.shuffle(buckets)

        scheduled: List[int] = []
        for bucket in buckets:
            rng.shuffle(bucket)
            scheduled.extend(self._basin_aware_order(bucket))
        return scheduled

    def _basin_aware_order(self, indices: List[int]) -> List[int]:
        samples = getattr(self.dataset, "samples", [])
        grouped: dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            basin_id = str(samples[idx].get("basin_id", "")) if idx < len(samples) else ""
            grouped[basin_id].append(idx)

        basin_ids = sorted(
            grouped.keys(),
            key=lambda bid: max((self._sample_loads[i] for i in grouped[bid]), default=0),
            reverse=True,
        )
        ordered: List[int] = []
        for basin_id in basin_ids:
            ordered.extend(sorted(grouped[basin_id], key=lambda i: self._sample_loads[i], reverse=True))
        return ordered

    def _rank_indices_for_epoch(self) -> List[int]:
        indices = self._order_rank_indices(list(self._rank_partitions[self.rank]))
        if self.drop_last:
            return indices[:self.num_samples]

        if not indices:
            raise RuntimeError(f"Rank {self.rank} has no samples to pad.")
        if len(indices) < self.num_samples:
            repeat = math.ceil(self.num_samples / len(indices))
            indices = (indices * repeat)[:self.num_samples]
        else:
            indices = indices[:self.num_samples]
        return indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._rank_indices_for_epoch())

    def __len__(self) -> int:
        return self.num_samples
