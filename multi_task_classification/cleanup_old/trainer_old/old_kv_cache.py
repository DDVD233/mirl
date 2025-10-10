
    # @torch.no_grad()
    # def _greedy_decode_no_generate(self, qa_input_ids, qa_attn, max_new_tokens=64):
    #     """
    #     Greedy decode WITHOUT .generate, using past_key_values for speed.
    #     - Keeps your behavior (no context-window cap).
    #     - Uses KV cache + disables grad ckpt during decode for stability/speed.
    #     Returns: cont_ids [B, L<=max_new_tokens]
    #     """
    #     device = qa_input_ids.device
    #     B, _ = qa_input_ids.shape

    #     eos_id = self.tokenizer.eos_token_id
    #     pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

    #     # --- turn on cache & temporarily disable gradient checkpointing
    #     # safer to call methods on unwrapped model, but config edits are fine either way
    #     bb = self.accelerator.unwrap_model(self.model).backbone if hasattr(self, "accelerator") else self.model.backbone
    #     prev_use_cache = getattr(bb.config, "use_cache", False)
    #     was_ckpt = getattr(bb, "is_gradient_checkpointing", False)

    #     if hasattr(bb.config, "use_cache"):
    #         bb.config.use_cache = True
    #     if was_ckpt and hasattr(bb, "gradient_checkpointing_disable"):
    #         bb.gradient_checkpointing_disable()

    #     try:
    #         generated = []
    #         finished = torch.zeros(B, dtype=torch.bool, device=device)

    #         # ---- 1) Build KV cache with full prompt
    #         domain_ids_q = torch.full((B,), -1, dtype=torch.long, device=device)  # sentinel: no head routing
    #         out = self.model(
    #             input_ids=qa_input_ids,
    #             attention_mask=qa_attn,
    #             domain_ids=domain_ids_q,
    #             lm_labels=None,
    #         )
    #         lm_out = out["lm_output"]
    #         logits = lm_out.logits  # [B, T, V]

    #         past = getattr(lm_out, "past_key_values", None)
    #         if past is None:
    #             past = getattr(out, "past_key_values", None)

    #         next_tokens = logits[:, -1, :].argmax(dim=-1)    # [B]
    #         generated.append(next_tokens.unsqueeze(1))
    #         finished |= (next_tokens == eos_id)

    #         input_ids_step = next_tokens.unsqueeze(1)        # [B,1]
    #         attn_step = None                                  # with past, not needed

    #         # ---- 2) Token-by-token with KV cache
    #         for _ in range(max_new_tokens - 1):
    #             if torch.all(finished):
    #                 break

    #             out = self.model(
    #                 input_ids=input_ids_step,
    #                 attention_mask=attn_step,
    #                 domain_ids=domain_ids_q,
    #                 lm_labels=None,
    #                 past_key_values=past if past is not None else None,
    #             )
    #             lm_out = out["lm_output"]
    #             logits = lm_out.logits  # [B, 1, V]

    #             new_past = getattr(lm_out, "past_key_values", None)
    #             if new_past is None:
    #                 new_past = getattr(out, "past_key_values", None)
    #             if new_past is not None:
    #                 past = new_past

    #             next_tokens = logits[:, -1, :].argmax(dim=-1)         # [B]
    #             next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)

    #             generated.append(next_tokens.unsqueeze(1))
    #             finished |= (next_tokens == eos_id)
    #             input_ids_step = next_tokens.unsqueeze(1)

    #         if not generated:
    #             return torch.empty((B, 0), dtype=qa_input_ids.dtype, device=device)
    #         return torch.cat(generated, dim=1)  # [B, L]

    #     finally:
    #         # ---- restore model flags
    #         if hasattr(bb.config, "use_cache"):
    #             bb.config.use_cache = prev_use_cache
    #         if was_ckpt and hasattr(bb, "gradient_checkpointing_enable"):
    #             try:
    #                 bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #             except TypeError:
    #                 bb.gradient_checkpointing_enable()
