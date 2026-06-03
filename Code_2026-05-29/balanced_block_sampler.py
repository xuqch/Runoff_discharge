from __future__ import annotations

import math
import random
from typing import Iterator, List

import torch.distributed as dist
from torch.utils.data import Sampler


class BalancedBlockDistributedSampler(Sampler[int]):
    """Rank-aware sampler that balances block samples by estimated block compute load."""

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
        if bucket_size < num_replicas * batch_size:
            bucket_size = num_replicas * batch_size

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.global_batch_size = self.num_replicas * self.batch_size
        self._sample_indices = list(range(len(dataset)))
        self._sample_loads = [int(v) for v in dataset.sample_loads]
        self.num_samples = self._compute_num_samples(len(self._sample_indices))

    def _compute_num_samples(self, sample_count: int) -> int:
        if self.drop_last:
            usable = (sample_count // self.global_batch_size) * self.global_batch_size
        else:
            usable = int(math.ceil(sample_count / self.global_batch_size)) * self.global_batch_size
        return usable // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_indices(self) -> List[int]:
        ordered = sorted(self._sample_indices, key=lambda idx: self._sample_loads[idx], reverse=True)
        if not self.shuffle:
            return ordered

        rng = random.Random(self.seed + self.epoch)
        buckets = [ordered[i:i + self.bucket_size] for i in range(0, len(ordered), self.bucket_size)]
        rng.shuffle(buckets)

        scheduled: List[int] = []
        for bucket in buckets:
            rng.shuffle(bucket)
            scheduled.extend(bucket)
        return scheduled

    def _balanced_indices(self) -> List[int]:
        ordered = self._ordered_indices()
        if self.drop_last:
            total_size = (len(ordered) // self.global_batch_size) * self.global_batch_size
            ordered = ordered[:total_size]
        else:
            padding = (-len(ordered)) % self.global_batch_size
            if padding > 0:
                ordered.extend(ordered[:padding])

        rank_indices: List[int] = []
        for start in range(0, len(ordered), self.global_batch_size):
            global_group = ordered[start:start + self.global_batch_size]
            rank_group = global_group[self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
            rank_indices.extend(rank_group)
        return rank_indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._balanced_indices())

    def __len__(self) -> int:
        return self.num_samples
