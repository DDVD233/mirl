# models/multi_head_omni_classifier.py
import torch
import torch.nn as nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

NEG_INF = -1e9  # safe mask for "irrelevant" classes

def build_domain_specs_from_labelscheme(full_label_scheme: dict):
    """
    Returns:
      domain_name_to_id: {'sentiment_intensity':0, 'emotion':1, 'mental_health':2}
      domain_id_to_global_indices: List[List[int]] e.g. [[0..6], [7..14], [15..20]]
      dataset_to_domain_id: map dataset_name -> domain_id using meta.dataset_domain
      global_num_classes: int (= 21)
    """
    meta = full_label_scheme["meta"]
    global_classes = meta["global_classes"]
    # preserve the ordering in the JSON
    domain_names = list(global_classes.keys())  # ['sentiment_intensity','emotion','mental_health']
    domain_name_to_id = {d:i for i,d in enumerate(domain_names)}
    domain_id_to_global_indices = []
    for d in domain_names:
        # each entry has {'index': int, 'label': str}
        idxs = [x["index"] for x in global_classes[d]]
        domain_id_to_global_indices.append(idxs)

    dataset_to_domain_name = meta["dataset_domain"]  # e.g. 'mosei_senti':'sentiment_intensity'
    dataset_to_domain_id = {ds: domain_name_to_id[dn] for ds, dn in dataset_to_domain_name.items()}

    global_num_classes = full_label_scheme.get("num_classes", max(max(g) for g in domain_id_to_global_indices)+1)

    return domain_name_to_id, domain_id_to_global_indices, dataset_to_domain_id, global_num_classes


class MultiHeadOmniClassifier(nn.Module):
    def __init__(self,
                 full_label_scheme: dict,
                 freeze_backbone="head_only",
                 backbone_name='Qwen/Qwen2.5-Omni-7B',
                 lora_config=None,
                 device_map="auto",
                 torch_dtype=torch.float16,
                 **from_pretrained_kwargs):
        super().__init__()

        # basically the full label scheme; not just the label mapping;
        # this includes the meta data
        self.full_label_scheme = full_label_scheme 

        # obtaining the domain name, ids, datasets and global number of classes
        (self.domain_name_to_id,
         self.domain_id_to_global_indices,
         self.dataset_to_domain_id,
         self.global_num_classes) = build_domain_specs_from_labelscheme(self.full_label_scheme)

        # === Backbone ===
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            **from_pretrained_kwargs
        }
        self.backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            backbone_name, attn_implementation="flash_attention_2", **model_kwargs
        )
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.device_map = device_map

        # obtain the hidden size of the backbone
        hidden_size = self._resolve_hidden_size(self.backbone)

        # === One head per domain ===
        self.heads = nn.ModuleList()
        for global_indices in self.domain_id_to_global_indices:
            k_t = len(global_indices)
            self.heads.append(nn.Linear(hidden_size, k_t))

        # Training strategy (freeze/LORA/full)
        self._setup_training_strategy(freeze_backbone, lora_config)

        # Align heads’ device/dtype with pooled backbone output
        self._ensure_heads_alignment()

    # ---------- utils ----------
    @staticmethod
    def _resolve_hidden_size(backbone):
        cfg = backbone.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        for sub in ("language_config", "vision_config", "audio_config", "encoder_config"):
            if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), "hidden_size"):
                return getattr(cfg, sub).hidden_size
        # last-resort probe
        device = next(backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.ones(1, 1, dtype=torch.long, device=device)
            out = backbone(input_ids=dummy, output_hidden_states=True)
            return out.hidden_states[-1].shape[-1]

    def _apply_lora(self, lora_config):
        cfg = {'r':16, 'alpha':32, 'dropout':0.1,
               'target_modules': ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]}
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
            for p in self.backbone.parameters(): p.requires_grad = False
            print("Training strategy: Head-only (backbone frozen)")
        elif freeze_backbone == "full" or freeze_backbone is False:
            print("Training strategy: Full fine-tuning")
        else:
            for p in self.backbone.parameters(): p.requires_grad = False
            print("Training strategy: Head-only (default)")

    def _ensure_heads_alignment(self):
        # Place heads where pooled features live
        if self.device_map == "auto":
            with torch.no_grad():
                dummy_input = torch.ones(1, 10, dtype=torch.long)
                dummy_mask  = torch.ones(1, 10, dtype=torch.long)
                out = self.backbone(input_ids=dummy_input, attention_mask=dummy_mask, output_hidden_states=True)
                h = out.hidden_states[-2].mean(dim=1)
                target_device, target_dtype = h.device, h.dtype
            for head in self.heads:
                head.to(device=target_device, dtype=target_dtype)
        else:
            dev = next(self.backbone.parameters()).device
            dt  = next(self.backbone.parameters()).dtype
            for head in self.heads:
                head.to(device=dev, dtype=dt)

    # ---------- forward ----------

    def forward(self, input_ids, attention_mask=None, domain_ids=None, lm_labels=None, **kwargs):
        """
        Single-pass forward for both LM (QA) and classification.
        - Pass `lm_labels` (None or masked with -100); HF computes LM loss internally.
        - Use `domain_ids` with sentinel -1 for QA-only rows. Only valid (>=0) rows are routed to heads.
        Returns:
            {
            "cls_logits": [B, global_num_classes] (neg_inf for irrelevant slots/rows),
            "lm_loss": scalar or None,
            "hidden_states": tuple(T+1) of [B,T,H] (from backbone)
            }
        """
        # ---- one backbone call (teacher-forcing handled via lm_labels masking) ----
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,                 # None or [-100]-masked full-batch labels
            output_hidden_states=True,
            **kwargs
        )

        lm_loss = getattr(out, "loss", None)

        h = out.hidden_states[-2]  # [B,T,H]

        # TODO: Double check what this effect has in the context of omni
        if attention_mask is not None:
            pooled = (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            pooled = h.mean(dim=1)  # [B,H]

        B = pooled.size(0)
        device, dtype = pooled.device, pooled.dtype

        # init masked logits; by default the logits are negative first in order to be ignored by the softmax
        # logits_all = torch.full((B, self.global_num_classes), NEG_INF, device=device, dtype=dtype)

         # Use dtype-aware NEG_INF
        neg_inf = torch.finfo(dtype).min / 2                     # safe huge negative in this dtype
        logits_all = torch.full((B, self.global_num_classes),
                                neg_inf, device=device, dtype=dtype)

        # if domain_ids is None:
        #     # If caller truly has no domains, just return LM path output.
        #     return {"cls_logits": logits_all, "lm_loss": lm_loss}


        # compute per-domain logits and scatter into global slots
        domain_ids = domain_ids.to(device)
    
        unique_domains = domain_ids.unique(sorted=True).tolist()
        
        for d in unique_domains:
            # iterate over the unique domains
            # get the rows for the current domain
            # essentially is a boolean rows of the samples that belong to the current domain
            # TRUE IF BELONGS TO THE CURRENT DOMAIN
            rows = (domain_ids == d).nonzero(as_tuple=True)[0]
            if rows.numel()==0:
                continue

            if d == -1:
                # QA-only rows: explicitly write a neg_inf block so these rows are "processed"
                block = torch.full((rows.numel(), self.global_num_classes),
                                neg_inf, device=device, dtype=dtype)
                logits_all.index_copy_(0, rows, block)
                continue

            #obtain the global indices for the current domain
            # i.e. sentiment is only [0, 6]
            global_slots = self.domain_id_to_global_indices[d]  # list[int]

            # obtain the tensor of the global slots
            cols = torch.as_tensor(
                global_slots,
                device=pooled.device, dtype=torch.long
            )

            # select the head for the current domain based on d
            head = self.heads[d]

            # select the embedding of the samples related to the specific current domain d
            # feed into the head to get the logits for the current domain
            local_logits = head(pooled.index_select(0, rows))  # [B_d, K_d]; B_d number of samples in this domain; K_d is number of classes in this domain

            # basically where the mask is true, we replace the logits with the global logits
            # logits_all is a big tensor of [B, global_num_classes (i.e. 21)], each with -1e9, where B is the batch size
            # logits_all[mask] has a shape of [B_d, 21]
            # mask is which samples belong to domain d;
            # we only prepend to the "global slots" (which refers to the global indices for the current domain),
            # the current local logits.
            # logits_all[mask][:, global_slots] = local_logits

            local_logits = local_logits.to(logits_all.dtype)     # <— key line

            block = torch.full((rows.numel(), self.global_num_classes), neg_inf, device=pooled.device, dtype=pooled.dtype)
            col_index = cols.unsqueeze(0).expand(rows.numel(), cols.numel())
            block = block.scatter(1, col_index, local_logits.to(block.dtype))  # differentiable w.r.t. local

            logits_all = logits_all.index_copy(0, rows, block)           # stitch rows
        
        return {"cls_logits": logits_all, "lm_loss": lm_loss, "lm_output": out}

    # Convenience: expose mappings
    @property
    def dataset_to_domain_id(self):
        # property already set in __init__, but kept for API symmetry
        return self._dataset_to_domain_id
    @dataset_to_domain_id.setter
    def dataset_to_domain_id(self, v):
        self._dataset_to_domain_id = v

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
        
        print(f"Backbonetrainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")
        return trainable_params, all_param
    
    # @torch.no_grad()
    # def generate_qa(
    #     self,
    #     accelerator,
    #     input_ids: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     gen_cfg: Optional[Dict[str, Any]] = None,
    # ) -> torch.Tensor:
    #     """
    #     Safe generation under Accelerate + ZeRO-3/FSDP.
    #     Returns full sequences [prompt | continuation] per row.
    #     """
    #     backbone = accelerator.unwrap_model(self.backbone)          # unwrapped MultiHeadOmniClassifier
    #     # backbone = core.backbone                       # HF CausalLM (possibly PEFT-wrapped)

    #     # devices/dtypes
    #     dev = next(backbone.parameters()).device
    #     input_ids = input_ids.contiguous().long().to(dev)
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.contiguous().long().to(dev)

    #     # deterministic eval defaults unless passed in
    #     if gen_cfg is None:
    #         gen_cfg = {
    #             "max_new_tokens": 64, "do_sample": False,
    #             "temperature": 0.0, "top_p": 1.0, "num_beams": 1,
    #             "eos_token_id": getattr(backbone.config, "eos_token_id", None),
    #             "pad_token_id": getattr(backbone.config, "pad_token_id", None),
    #         }
    #     if hasattr(backbone.config, "use_cache"):
    #         backbone.config.use_cache = True

    #     # plain DDP/single-GPU
    #     with FSDP.summon_full_params(backbone, writeback=False, recurse=True):
    #         return backbone.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_cfg)
    

        # # choose the right sharding context
        # ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        # if ds_plugin is not None:
        #     import deepspeed
        #     # materialize full params during generate (ZeRO-3 safe)
        #     with deepspeed.zero.GatheredParameters(backbone.parameters(), modifier_rank=None):
        #         return backbone.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_cfg)

        # # FSDP path (if wrapped by FSDP)
        # try:
        #     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        #     if isinstance(core, FSDP) or any(isinstance(m, FSDP) for m in core.modules()):
        #         with FSDP.summon_full_params(backbone, writeback=False, recurse=True):
        #             return backbone.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_cfg)
        # except Exception:
        #     pass  # not FSDP, fall through

