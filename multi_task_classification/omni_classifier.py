import torch.nn as nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

class OmniClassifier(nn.Module):
    """
    Qwen2.5 Omni backbone encoder + classification head for cross-entropy classification tasks.
    Set freeze_backbone=True to freeze the backbone parameters.
    Enable LoRA by passing a lora_config dictionary for efficient fine-tuning.
    """
    def __init__(self, num_classes=5, freeze_backbone="head_only", backbone_name='Qwen/Qwen2.5-Omni-7B', 
                 lora_config=None):
        super().__init__()
        self.backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(backbone_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        
        # Handle different freezing strategies
        self._setup_training_strategy(freeze_backbone, lora_config)

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
