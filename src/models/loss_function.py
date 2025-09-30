from typing import Optional
import torch
import torch.nn.functional as F


def contrastive_loss_with_queue(im_vecs: torch.Tensor, loc_vecs: torch.Tensor,
                                q_tensor: Optional[torch.Tensor] = None,
                                temperature: float = 0.07) -> torch.Tensor:
    device = im_vecs.device
    B = im_vecs.shape[0]

    im_norm = F.normalize(im_vecs, p=2, dim=-1)
    loc_norm = F.normalize(loc_vecs, p=2, dim=-1)

    if q_tensor is not None:
        loc_concat = torch.cat([loc_norm, F.normalize(q_tensor, p=2, dim=-1)], dim=0)
    else:
        loc_concat = loc_norm

    logits_i2l = (im_norm @ loc_concat.t()) / temperature  # [B, B+Q]
    targets = torch.arange(B, device=device)
    loss_i2l = F.cross_entropy(logits_i2l, targets)

    logits_l2i = (loc_norm @ im_norm.t()) / temperature  # [B, B]
    loss_l2i = F.cross_entropy(logits_l2i, targets)

    return 0.5 * (loss_i2l + loss_l2i)