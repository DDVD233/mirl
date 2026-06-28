# models/qwen3_vl_classifier_heads_decoder.py
"""
Qwen3-VL multi-head classifier / decoder base.

Parallel to MultiHeadOmniClassifier (Qwen2.5-Omni), but targets
Qwen3VLForConditionalGeneration. The Omni file is left untouched; this module is
the VL counterpart used by the BAM-for-VL wrappers.

Confirmed Qwen3-VL-8B-Instruct structure (transformers 4.57.x):
  - config.text_config.hidden_size = 4096
  - tie_word_embeddings = False -> real, separate backbone.lm_head (4096 -> 151936)
  - final RMSNorm at backbone.model.language_model.norm
    (NOTE: backbone.model.norm does NOT exist on Qwen3-VL — the Omni-era path)
  - output_hidden_states=True returns (num_layers + 1) states; hidden_states[-1]
    is the PRE-final-norm last-layer output, so official logits are
    lm_head(norm(hidden_states[-1])). The QA injection math relies on this.
  - forward natively accepts pixel_values_videos / video_grid_thw for native video.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType

# Reuse the label-scheme parsing from the Omni module (pure function, backbone-agnostic).
from .qwen2_5_omni_classifier_heads_decoder import build_domain_specs_from_labelscheme

NEG_INF = -1e9  # safe mask for "irrelevant" classes

DEFAULT_BACKBONE_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def _resolve_backbone_loader(backbone_class):
    """
    Pick the from_pretrained loader for the backbone.

    - None  -> AutoModelForImageTextToText, which dispatches to the correct class from the
               checkpoint's config.architectures: Qwen3VLForConditionalGeneration (dense,
               e.g. 8B), Qwen3VLMoeForConditionalGeneration (MoE, e.g. 30B-A3B),
               Qwen2_5_VLForConditionalGeneration, etc. So switching model family is just a
               backbone_name change.
    - str   -> looked up by class name in `transformers` (escape hatch, e.g.
               "Qwen3VLMoeForConditionalGeneration").
    - class -> used directly.
    """
    if backbone_class is None:
        return AutoModelForImageTextToText
    if isinstance(backbone_class, str):
        cls = getattr(transformers, backbone_class, None)
        if cls is None:
            raise ValueError(f"backbone_class '{backbone_class}' not found in transformers")
        return cls
    return backbone_class


class MultiHeadVLClassifier(nn.Module):
    """
    Qwen3-VL backbone with per-domain classification heads and a decoder (QA) path.

    Mirrors the public surface of MultiHeadOmniClassifier so the BAM wrappers can
    subclass it, but loads a Qwen3-VL backbone and resolves the final norm / lm_head
    through the VL-nested module tree.
    """

    def __init__(self,
                 full_label_scheme: dict,
                 freeze_backbone="head_only",
                 backbone_name=DEFAULT_BACKBONE_NAME,
                 backbone_class=None,
                 attn_implementation="flash_attention_2",
                 lora_config=None,
                 device_map="auto",
                 torch_dtype=torch.bfloat16,
                 **from_pretrained_kwargs):
        super().__init__()

        self.full_label_scheme = full_label_scheme
        (self.domain_name_to_id,
         self.domain_id_to_global_indices,
         self.dataset_to_domain_id,
         self.global_num_classes) = build_domain_specs_from_labelscheme(self.full_label_scheme)

        # === Backbone === (AutoModel dispatch by default; see _resolve_backbone_loader)
        backbone_loader = _resolve_backbone_loader(backbone_class)
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            **from_pretrained_kwargs,
        }
        self.backbone = backbone_loader.from_pretrained(
            backbone_name, attn_implementation=attn_implementation, **model_kwargs
        )
        self.backbone_name = backbone_name
        self.device_map = device_map

        hidden_size = self._resolve_hidden_size(self.backbone)
        self.hidden_size = hidden_size

        # === One head per domain ===
        self.heads = nn.ModuleList()
        for global_indices in self.domain_id_to_global_indices:
            self.heads.append(nn.Linear(hidden_size, len(global_indices)))

        self._setup_training_strategy(freeze_backbone, lora_config)
        self._ensure_heads_alignment()

    # ---------- backbone-structure resolvers ----------
    @staticmethod
    def _resolve_hidden_size(backbone):
        cfg = backbone.config
        # Qwen3-VL nests text dims under text_config.
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        for sub in ("language_config", "vision_config", "audio_config", "encoder_config"):
            if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), "hidden_size"):
                return getattr(cfg, sub).hidden_size
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        device = next(backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.ones(1, 1, dtype=torch.long, device=device)
            out = backbone(input_ids=dummy, output_hidden_states=True)
            return out.hidden_states[-1].shape[-1]

    @staticmethod
    def _resolve_attr(obj, dotted_paths):
        """Return the first attribute reachable by any of the dotted paths, else None."""
        for path in dotted_paths:
            cur = obj
            ok = True
            for attr in path.split("."):
                if hasattr(cur, attr):
                    cur = getattr(cur, attr)
                else:
                    ok = False
                    break
            if ok:
                return cur
        return None

    def _resolve_final_norm(self):
        """
        Locate the backbone's final RMSNorm. Qwen3-VL keeps it at
        model.language_model.norm; the Omni Thinker exposes model.norm. We try the
        VL path first and fall back, raising rather than silently skipping the norm
        (the silent-skip would feed un-normed states to lm_head and corrupt QA logits).
        """
        norm = self._resolve_attr(self.backbone, [
            "model.language_model.norm",   # Qwen3-VL / Qwen2.5-VL
            "language_model.norm",         # top-level alias on some VL wrappers
            "model.norm",                  # Qwen2.5-Omni Thinker / plain decoders
            "transformer.ln_f",            # GPT-style
        ])
        if norm is None:
            raise RuntimeError(
                "Could not locate a final norm on the backbone; checked "
                "model.language_model.norm / language_model.norm / model.norm / transformer.ln_f"
            )
        return norm

    def _resolve_lm_head(self):
        head = self._resolve_attr(self.backbone, ["lm_head", "model.lm_head"])
        if head is None:
            raise RuntimeError("Backbone has no resolvable lm_head")
        return head

    # ---------- training strategy ----------
    def _apply_lora(self, lora_config):
        cfg = {'r': 16, 'alpha': 32, 'dropout': 0.1,
               'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]}
        cfg.update(lora_config or {})
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=cfg['r'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'],
            target_modules=cfg['target_modules'], bias="none",
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        print(f"Applied LoRA r={cfg['r']} alpha={cfg['alpha']} dropout={cfg['dropout']}")

    def _setup_training_strategy(self, freeze_backbone, lora_config):
        if freeze_backbone == "lora":
            if lora_config is None:
                raise ValueError("lora_config must be provided when freeze_backbone='lora'")
            self._apply_lora(lora_config)
            print("Training strategy: LoRA (backbone unfrozen)")
        elif freeze_backbone == "head_only" or freeze_backbone is True:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Training strategy: Head-only (backbone frozen)")
        elif freeze_backbone == "full" or freeze_backbone is False:
            print("Training strategy: Full fine-tuning")
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Training strategy: Head-only (default)")

    def _ensure_heads_alignment(self):
        if self.device_map == "auto":
            with torch.no_grad():
                dummy_input = torch.ones(1, 10, dtype=torch.long)
                dummy_mask = torch.ones(1, 10, dtype=torch.long)
                out = self.backbone(input_ids=dummy_input, attention_mask=dummy_mask,
                                    output_hidden_states=True)
                h = out.hidden_states[-2].mean(dim=1)
                target_device, target_dtype = h.device, h.dtype
            for head in self.heads:
                head.to(device=target_device, dtype=target_dtype)
        else:
            dev = next(self.backbone.parameters()).device
            dt = next(self.backbone.parameters()).dtype
            for head in self.heads:
                head.to(device=dev, dtype=dt)

    def _heads_from_pooled(self, pooled: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """Route pooled hidden state through per-domain heads into global logit space."""
        B = pooled.size(0)
        device, dtype = pooled.device, pooled.dtype
        neg_inf = torch.finfo(dtype).min / 2
        logits_all = torch.full((B, self.global_num_classes), neg_inf, device=device, dtype=dtype)

        domain_ids = domain_ids.to(device)
        for d in domain_ids.unique(sorted=True).tolist():
            if d < 0:  # QA rows (domain_id = -1): no head, leave neg_inf
                continue
            rows = (domain_ids == d).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = torch.as_tensor(self.domain_id_to_global_indices[d], device=device, dtype=torch.long)
            local_logits = self.heads[d](pooled.index_select(0, rows)).to(dtype)
            block = torch.full((rows.numel(), self.global_num_classes), neg_inf, device=device, dtype=dtype)
            block = block.scatter(1, cols.unsqueeze(0).expand(rows.numel(), cols.numel()), local_logits)
            logits_all = logits_all.index_copy(0, rows, block)
        return logits_all

    # ---------- forward ----------
    def forward(self, input_ids, attention_mask=None, domain_ids=None, lm_labels=None, **kwargs):
        if domain_ids is None:
            raise ValueError("domain_ids must be provided for multi-head routing (use -1 for QA rows).")

        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False, **kwargs
        )
        hidden_states = out.hidden_states
        h = hidden_states[-2]  # penultimate, pre-norm [B, T, H]

        if attention_mask is not None:
            pooled = (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            pooled = h.mean(dim=1)

        if lm_labels is None:
            return self._heads_from_pooled(pooled, domain_ids)

        # --- QA path ---
        cls_logits = self._heads_from_pooled(pooled, domain_ids)
        h_for_lm = self._resolve_final_norm()(hidden_states[-1])
        lm_logits = self._resolve_lm_head()(h_for_lm)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = lm_labels[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        out.logits = lm_logits
        out.loss = lm_loss
        return {"cls_logits": cls_logits, "lm_loss": lm_loss, "lm_output": out}

    def get_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.backbone.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Backbone trainable params: {trainable_params:,} || all params: {all_param:,} "
              f"|| trainable%: {100 * trainable_params / max(1, all_param):.2f}%")
        return trainable_params, all_param
