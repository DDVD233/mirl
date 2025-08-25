import torch.nn as nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

class OmniClassifier(nn.Module):
    """
    Qwen2.5 Omni backbone encoder + classification head for cross-entropy classification tasks.
    Set freeze_backbone=True to freeze the backbone parameters.
    """
    def __init__(self, num_classes=5, freeze_backbone=True, backbone_name='Qwen/Qwen2.5-Omni-7B'):
        super().__init__()
        self.backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(backbone_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Forward through backbone, get last hidden state
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Pool: mean over sequence (can change to [CLS] or other pooling)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
