import sys
from pathlib import Path

import torch

# Directly import module without triggering project package import
sys.path.insert(0, str(Path(__file__).parent.parent / "finetune_scripts"))
from modality_dropout import apply_modality_dropout


def test_modality_dropout_masks():
    torch.manual_seed(42)
    B, E = 4, 3
    h_img = torch.ones(B, E)
    h_tab = torch.ones(B, E)

    # Expected masks using same RNG sequence
    expected_mask_img = torch.bernoulli(torch.full((B,), 0.5))
    expected_mask_tab = torch.bernoulli(torch.full((B,), 0.5))

    torch.manual_seed(42)
    h_img_out, h_tab_out, mask_img, mask_tab = apply_modality_dropout(
        h_img, h_tab, p_img=0.5, p_tab=0.5, tab_group_masks=None, training=True
    )

    assert torch.equal(mask_img, expected_mask_img)
    assert torch.equal(mask_tab, expected_mask_tab)
    assert torch.equal(mask_img, (h_img_out.abs().sum(dim=1) > 0).float())
    assert torch.equal(mask_tab, (h_tab_out.abs().sum(dim=1) > 0).float())


def test_group_dropout_masks():
    torch.manual_seed(0)
    B = 2
    group_sizes = [2, 2]
    E = sum(group_sizes)
    h_img = torch.ones(B, E)
    h_tab = torch.ones(B, E)
    group_masks = {f"g{i}": torch.ones(B, size) for i, size in enumerate(group_sizes)}

    # Expected group keep masks
    torch.manual_seed(0)
    keep_g1 = torch.bernoulli(torch.full((B, 1), 0.5))
    keep_g2 = torch.bernoulli(torch.full((B, 1), 0.5))
    expected_tab = torch.cat([
        torch.ones(B, group_sizes[0]) * keep_g1,
        torch.ones(B, group_sizes[1]) * keep_g2,
    ], dim=1)
    expected_mask_tab = (expected_tab.abs().sum(dim=1) > 0).float()

    torch.manual_seed(0)
    h_img_out, h_tab_out, mask_img, mask_tab = apply_modality_dropout(
        h_img, h_tab, p_img=0.0, p_tab=0.0, tab_group_masks=group_masks, training=True
    )

    assert torch.equal(h_img_out, h_img)
    assert torch.allclose(h_tab_out, expected_tab)
    assert torch.equal(mask_img, torch.ones(B))
    assert torch.equal(mask_tab, expected_mask_tab)
