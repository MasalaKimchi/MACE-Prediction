import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune_scripts.training_utils import (
    EventAwareBatchSampler,
    EventAwareDistributedBatchSampler,
)


def test_event_aware_batch_sampler_min_events():
    events = [0] * 20
    events[2] = 1
    events[10] = 1
    events[15] = 1  # three events in dataset
    sampler = EventAwareBatchSampler(events, batch_size=6, m_events_per_batch=2, seed=42)
    for batch in sampler:
        event_count = sum(events[i] for i in batch)
        assert event_count >= 2


def test_event_aware_distributed_batch_sampler_min_events():
    events = [0] * 20
    events[3] = 1
    events[12] = 1
    batch_size = 4
    m_events = 1
    sampler0 = EventAwareDistributedBatchSampler(
        events,
        batch_size=batch_size,
        m_events_per_batch=m_events,
        num_batches=5,
        seed=0,
        rank=0,
        world_size=2,
    )
    sampler1 = EventAwareDistributedBatchSampler(
        events,
        batch_size=batch_size,
        m_events_per_batch=m_events,
        num_batches=5,
        seed=0,
        rank=1,
        world_size=2,
    )
    for b0, b1 in zip(sampler0, sampler1):
        count0 = sum(events[i] for i in b0)
        count1 = sum(events[i] for i in b1)
        assert count0 >= m_events
        assert count1 >= m_events
