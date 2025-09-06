import torch
from losses import (
    cox_ph_loss,
    concordance_pairwise_loss,
    time_aware_triplet_loss,
)


def test_losses_smoke():
    torch.manual_seed(0)
    n = 8
    log_risks = torch.randn(n)
    times = torch.arange(1, n + 1, dtype=torch.float32)
    events = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1], dtype=torch.float32)
    assert cox_ph_loss(log_risks, times, events) >= 0
    assert concordance_pairwise_loss(log_risks, times, events, temperature=1.0) >= 0
    img = torch.randn(n, 4)
    tab = torch.randn(n, 4)
    assert time_aware_triplet_loss(img, tab, times) >= 0
