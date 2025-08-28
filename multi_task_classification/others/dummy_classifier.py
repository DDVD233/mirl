from torch import nn
import torch

# ---------------------------
# DUMMY MODEL
# ---------------------------
class DummyClassifier(nn.Module):
    """
    Tiny stand-in for the real backbone. Preserves:
      - Forward signature
      - Logit shape [B, NUM_CLASSES]
      - Requires grad so loss.backward()/optimizer.step() work

    It just computes mean(input_ids) -> Linear(1->C).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(1, num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Defensive: accept list/tuple from some collates
        if isinstance(input_ids, (list, tuple)):
            input_ids = input_ids[0]
        # Expect [B, L], but degrade gracefully
        x = input_ids.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # mask-aware mean if provided
        if attention_mask is not None:
            attn = attention_mask.float()
            denom = torch.clamp(attn.sum(dim=1, keepdim=True), min=1.0)
            mean_token = (x * attn).sum(dim=1, keepdim=True) / denom
        else:
            mean_token = x.mean(dim=1, keepdim=True)
        return self.proj(mean_token)  # [B, C]
