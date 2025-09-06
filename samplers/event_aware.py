from __future__ import annotations

import math
import numpy as np
from typing import Iterator, Sequence, List

from torch.utils.data import Sampler


class EventAwareBatchSampler(Sampler[List[int]]):
    """Batch sampler ensuring a minimum fraction of events per batch.

    Parameters
    ----------
    events : Sequence[int]
        Sequence of binary event indicators.
    batch_size : int
        Size of each batch.
    event_fraction : float, optional
        Desired fraction of event samples per batch. At least one event is
        sampled if available.
    num_batches : int, optional
        Number of batches per epoch. If ``None`` it is computed from dataset
        length.
    """

    def __init__(
        self,
        events: Sequence[int],
        batch_size: int,
        event_fraction: float = 0.5,
        num_batches: int | None = None,
    ) -> None:
        self.events = np.asarray(events).astype(bool)
        self.batch_size = int(batch_size)
        self.event_fraction = float(event_fraction)
        n = len(self.events)
        self.num_batches = num_batches or math.ceil(n / batch_size)
        self.event_idx = np.where(self.events)[0]
        self.non_event_idx = np.where(~self.events)[0]

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()
        n_events = max(1, int(self.batch_size * self.event_fraction))
        n_non = self.batch_size - n_events
        for _ in range(self.num_batches):
            if len(self.event_idx) > 0:
                ev = rng.choice(self.event_idx, size=n_events, replace=True)
            else:
                ev = np.array([], dtype=int)
            non = rng.choice(self.non_event_idx, size=n_non, replace=True)
            batch = np.concatenate([ev, non])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches
