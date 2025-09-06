import torch
from torch.utils.data import Dataset, DataLoader

from architectures import build_multimodal_network
from finetune_scripts.training_utils import run_epoch
from samplers import EventAwareBatchSampler


class DummyDataset(Dataset):
    def __init__(self, n: int = 12, tab_dim: int = 8):
        self.x = torch.randn(n, tab_dim)
        self.times = torch.arange(1, n + 1, dtype=torch.float32)
        self.events = (torch.arange(n) % 3 == 0).float()
        self.imgs = torch.randn(n, 1, 16, 16, 16)
        self.paths = ["p"] * n

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return self.x[idx], self.times[idx], self.events[idx], self.imgs[idx], self.paths[idx]


def test_training_step_smoke():
    dataset = DummyDataset()
    sampler = EventAwareBatchSampler(dataset.events.numpy(), batch_size=4, event_fraction=0.5)
    loader = DataLoader(dataset, batch_sampler=sampler)
    model = build_multimodal_network(tabular_dim=dataset.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_weights = {"cox": 1.0, "cpl": 0.1, "tmcl": 0.1}
    loss_params = {"cpl": {"temperature": 1.0}, "tmcl": {"margin": 1.0}}
    run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer, loss_weights=loss_weights, loss_params=loss_params)
