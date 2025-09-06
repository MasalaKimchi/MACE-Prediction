import torch

def apply_modality_dropout(
    h_img: torch.Tensor,
    h_tab: torch.Tensor,
    p_img: float,
    p_tab: float,
    tab_group_masks: dict[str, torch.Tensor] | None,
    training: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply modality and group-level dropout with availability masks.

    Args:
        h_img: Image modality features ``[B, E]``.
        h_tab: Tabular modality features ``[B, E]``.
        p_img: Probability of dropping the entire image modality.
        p_tab: Probability of dropping the entire tabular modality.
        tab_group_masks: Optional mapping of group name to mask ``[B, F_g]``.
            Used to randomly drop individual groups when training.
        training: If ``True`` dropout is applied, otherwise inputs are returned
            unchanged.

    Returns:
        Tuple ``(h_img_out, h_tab_out, mask_img, mask_tab)`` where ``mask_*`` are
        ``[B]`` tensors indicating availability of each modality.
    """
    B = h_img.shape[0]
    device = h_img.device

    h_img_out = h_img.clone()
    h_tab_out = h_tab.clone()

    # Initialize masks as ones (available)
    mask_img = torch.ones(B, device=device, dtype=h_img.dtype)
    mask_tab = torch.ones(B, device=device, dtype=h_tab.dtype)

    if training:
        if p_img > 0:
            keep_img = torch.bernoulli(torch.full((B,), 1 - p_img, device=device))
            mask_img = keep_img.to(h_img.dtype)
            h_img_out = h_img_out * keep_img.view(-1, 1)

        if p_tab > 0:
            keep_tab = torch.bernoulli(torch.full((B,), 1 - p_tab, device=device))
            mask_tab = keep_tab.to(h_tab.dtype)
            h_tab_out = h_tab_out * keep_tab.view(-1, 1)

        # Group dropout for tabular features
        if tab_group_masks:
            start = 0
            for mask in tab_group_masks.values():
                width = mask.shape[1]
                end = start + width
                group_feats = h_tab_out[:, start:end]
                # Respect existing mask from data
                group_feats = group_feats * mask.to(h_tab_out.device)
                # Randomly drop the entire group with Bernoulli(0.5)
                drop = torch.bernoulli(torch.full((B, 1), 0.5, device=device))
                group_feats = group_feats * drop
                h_tab_out[:, start:end] = group_feats
                start = end

        # Update availability masks after dropout
        mask_img = (h_img_out.abs().sum(dim=1) > 0).to(h_img.dtype)
        mask_tab = (h_tab_out.abs().sum(dim=1) > 0).to(h_tab.dtype)
    else:
        if tab_group_masks:
            start = 0
            for mask in tab_group_masks.values():
                width = mask.shape[1]
                end = start + width
                h_tab_out[:, start:end] = h_tab_out[:, start:end] * mask.to(h_tab_out.device)
                start = end
        mask_img = (h_img_out.abs().sum(dim=1) > 0).to(h_img.dtype)
        mask_tab = (h_tab_out.abs().sum(dim=1) > 0).to(h_tab.dtype)

    return h_img_out, h_tab_out, mask_img, mask_tab
