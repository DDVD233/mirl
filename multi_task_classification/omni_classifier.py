import torch.nn as nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import torch

class OmniClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone="head_only",
                 backbone_name='Qwen/Qwen2.5-Omni-7B', lora_config=None, **from_pretrained_kwargs):
        super().__init__()
        self.backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            backbone_name, **from_pretrained_kwargs
        )

        hidden_size = self._resolve_hidden_size(self.backbone)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self._setup_training_strategy(freeze_backbone, lora_config)

    @staticmethod
    def _resolve_hidden_size(backbone):
        cfg = backbone.config
        # Prefer the text encoder's hidden size (the one you’ll pool over)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        # Some composite configs may also nest other subconfigs (vision/audio/talker)
        for sub in ("language_config", "vision_config", "audio_config", "encoder_config"):
            if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), "hidden_size"):
                return getattr(cfg, sub).hidden_size
        # Last-resort: run a tiny forward to discover the width
        device = next(backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.ones(1, 1, dtype=torch.long, device=device)
            out = backbone(input_ids=dummy, output_hidden_states=True)
            # last layer width
            return out.hidden_states[-1].shape[-1]

    def _apply_lora(self, lora_config):
        """
        Apply LoRA configuration to the backbone model.
        
        Args:
            lora_config (dict): LoRA configuration dictionary with keys:
                - r (int): LoRA rank (default: 16)
                - alpha (int): LoRA alpha parameter (default: 32)
                - dropout (float): LoRA dropout rate (default: 0.1)
                - target_modules (list): List of target module names (default: common attention/MLP modules)
        """
        # Set defaults if not provided
        config = {
            'r': 16,
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }

        config.update(lora_config)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['r'],
            lora_alpha=config['alpha'],
            lora_dropout=config['dropout'],
            target_modules=config['target_modules'],
            bias="none",
        )
        
        self.backbone = get_peft_model(self.backbone, peft_config)
        print(f"Applied LoRA with r={config['r']}, alpha={config['alpha']}, dropout={config['dropout']}")
        print(f"Target modules: {config['target_modules']}")

    def _setup_training_strategy(self, freeze_backbone, lora_config):
        """
        Setup the training strategy based on freeze_backbone parameter.
        
        Args:
            freeze_backbone: Training strategy - "head_only", "lora", "full", or boolean
            lora_config: LoRA configuration dictionary (only used when freeze_backbone="lora")
        """
        if freeze_backbone == "lora":
            # Apply LoRA for efficient backbone training
            if lora_config is None:
                raise ValueError("lora_config must be provided when freeze_backbone='lora'")
            self._apply_lora(lora_config)
            # Backbone is unfrozen but LoRA handles efficient training
            print("Training strategy: LoRA (backbone unfrozen, efficient training)")
            
        elif freeze_backbone == "head_only" or freeze_backbone is True:
            # Freeze backbone, train only classifier head
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Training strategy: Head-only (backbone frozen)")
            
        elif freeze_backbone == "full" or freeze_backbone is False:
            # Full fine-tuning - all parameters trainable
            print("Training strategy: Full fine-tuning (all parameters trainable)")
            
        else:
            # Default to head_only for backward compatibility
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Training strategy: Head-only (default)")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # NOTE: We do not care about the modality imbalance for now when we average everything 
        # for pooling
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        # hidden states: tuple of [layer_0, ..., layer_n]
        h = out.hidden_states[-2]          # penultimate layer, [B, T, H]

        if attention_mask is not None:
            raise Exception(f"Debugging, attention mask found, {attention_mask}")
            # mean-pool only over non-padding tokens
            pooled = (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            raise Exception(f"Debugging, attention mask not found, {attention_mask}")
            pooled = h.mean(dim=1)         # fallback: plain mean over sequence

        return self.classifier(pooled)     # [B, num_classes]

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self):
        """
        Get the number of trainable parameters for monitoring LoRA efficiency.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.backbone.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")
        return trainable_params, all_param
