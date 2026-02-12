
    if use_task_mixture_density_adapter:

        # Hyperparams for mixture responsibilities
        BETA_R    = 4.0     # how sharp the signal/background decision is; how crisp the cut between background and tail-signal is.
        DELTA_R   = 0.25    # cutoff threshold for z;  how far into the tail you must go before a sample is treated as “signal.”
        GAMMA     = 1     # how strongly density ratios affect scale
        MIN_SCALE = 0.25
        MAX_SCALE = 8.0

        # Step 1 — compute per-task signal masses
        task_signal_mass = defaultdict(float)
        task_count       = defaultdict(int)  # still useful for logging if you want

        # First pass: compute z-scores + responsibilities per sample
        q2rvalues = {}  # store r_{t,i} so we can reuse them if needed
        
        for qid, vals in q2rollouts.items():
            task = q2tasks[qid]

            # Fetch EMA robust stats
            p50      = float(task_stats[task].get("post_grpo_advantage_ema_p50", 0.0))
            p90      = float(task_stats[task].get("post_grpo_advantage_ema_p90", 0.0))
            abs_mean = float(task_stats[task].get("post_grpo_advantage_ema_abs_mean", 0.0))

            # Denominator for z-score (robust tail spread)
            denom = max((p90 - p50), eps)

            r_list = []
            for v in vals:
                a = abs(v)

                # Step 1a — robust z-score
                z = (a - p50) / (denom + eps)

                # Step 1b — mixture "responsibility"
                r = 1.0 / (1.0 + math.exp(-BETA_R * (z - DELTA_R)))

                # Step 1c — accumulate signal mass for the task
                task_signal_mass[task] += r * a
                task_count[task]       += 1

                r_list.append(r)

            q2rvalues[qid] = r_list

        # Step 2 — compute task-level "densities" ρ_t as *average* signal mass
        task_densities = {}
        density_list   = []

        for task, mass in task_signal_mass.items():
            count = max(task_count[task], 1)
            
            # NORMALIZE BY COUNT
            rho_t = max(mass / count, eps)

            task_densities[task] = rho_t
            density_list.append(rho_t)    

        # Step 3 — reference density (geometric mean across tasks)
        log_rhos    = [math.log(r) for r in density_list]
        log_rho_ref = sum(log_rhos) / max(len(log_rhos), 1)
        rho_ref     = math.exp(log_rho_ref)

        # Step 4 — Apply scaling to each qid via density ratio
        q2norm = {}
        for qid, vals in q2rollouts.items():
            task  = q2tasks[qid]
            rho_t = task_densities.get(task, eps)

            # log ratio: boost sparse (low rho_t) tasks, shrink dense ones
            log_ratio = log_rho_ref - math.log(rho_t)

            scale_t = math.exp(GAMMA * log_ratio)
            scale_t = max(MIN_SCALE, min(MAX_SCALE, scale_t))

            # For logging
            task_stats[task]["mixture_rho_t"]        = float(rho_t)
            task_stats[task]["mixture_rho_ref"]      = float(rho_ref)
            task_stats[task]["mixture_final_scale"]  = float(scale_t)
            task_stats[task]["mixture_signal_mass"]  = float(task_signal_mass[task])
            task_stats[task]["mixture_batch_count"]  = int(task_count[task])

            # Apply scaling
            q2norm[qid] = [v * scale_t for v in vals]

    else:
        # No adapter → identity scaling
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}


    # -------------------------------------------
    # E) Static inverse-frequency class weights
    #     w_{d,c} = ((1/N_{d,c}) / sum_{c'∈C_d} 1/N_{d,c'}) * |C_d|
    # produce a single scalar q2w[qid]
    # -------------------------------------------

 if use_class_weights:
        if class_count_info == "import_from_dict":
            try:
                from verl.trainer.ppo.utils.v7_class_counts import CLASS_COUNT_INFO_DATASET, CLASS_COUNT_INFO_TASK
            except Exception:
                CLASS_COUNT_INFO_DATASET, CLASS_COUNT_INFO_TASK = {}, {}
            # Per-(dataset,class) EMA counts for inverse-frequency weights
            # dc_counts[(d,c)] = float (EMA count), d_counts[d] = float (EMA total), d_classes[d] = set of classes observed

            if class_weight_scope == "dataset":
                class_count_info = CLASS_COUNT_INFO_DATASET
            # elif class_weight_scope == "task":
            #     class_count_info = CLASS_COUNT_INFO_TASK
            elif class_weight_scope == "auto":
                # merged = dict(CLASS_COUNT_INFO_TASK)
                merged = dict(CLASS_COUNT_INFO_DATASET)
                # merged.update(CLASS_COUNT_INFO_TASK)
                class_count_info = merged
            else:
                raise ValueError(
                    "class_count_info must be provided as {group_id -> {class_label -> count}} "
                    "(group_id is a dataset_id or a task_id depending on class_weight_scope)."
                )

        # Precompute denominators S_g and cardinalities |C_g| per group (dataset OR task).
        g2_S: Dict[Any, float] = {}
        g2_C: Dict[Any, int]   = {}
        for group_id, class_count_map in class_count_info.items():
            if not class_count_map:
                g2_S[group_id] = 1.0
                g2_C[group_id] = 1
                continue
            S = 0.0
            for c2, cnt in class_count_map.items():
                # count of the group class label
                Ngc = float(max(cnt, 1))
                S  += 1.0 / max(Ngc, eps)
            g2_S[group_id] = max(S, eps)
            g2_C[group_id] = max(len(class_count_map), 1)

        def _resolve_group_id_based_on_qid(qid: Any) -> Any:
            """Pick dataset or task grouping for this qid."""
            if class_weight_scope == "dataset":
                return q2datasets[qid]
            elif class_weight_scope == "task":
                return q2tasks[qid]
            elif class_weight_scope == "auto":
                d = q2datasets[qid]
                t = q2tasks[qid]
                if d in class_count_info:
                    return d
                if t in class_count_info:
                    return t
                return None
            else:
                raise ValueError(
                    f"Invalid class_weight_scope={class_weight_scope!r}; use 'dataset' | 'task' | 'auto'."
                )

        # Per-qid scalar weight
        q2w: Dict[Any, float] = {}
        for qid in q2norm.keys():
            group_id = _resolve_group_id_based_on_qid(qid)
            if group_id is None or group_id not in class_count_info:
                # Group not found at all → neutral
                q2w[qid] = 1.0
                continue

            class_map = class_count_info[group_id]

            # If the group exists but is EMPTY (QA), keep it strictly neutral.
            if not class_map:
                q2w[qid] = 1.0
                continue

            c = q2class[qid]

            # If class label is None or unseen, treat as count=1 (neutral).
            raw_cnt = class_map.get(c, 1) if c is not None else 1
            Ngc = float(max(raw_cnt, 1))

            inv = 1.0 / max(Ngc, eps)
            Sg  = g2_S[group_id]
            Cg  = g2_C[group_id]

            q2w[qid] = inv * (Cg / max(Sg, eps))
    else:
        q2w = {qid: 1.0 for qid in q2norm.keys()}

    # ----------------------------------------------------------
    # F) CVaR×tail-frequency dynamic boost (per task) → q2k[qid]
    #     Buffer uses per-question MEAN of (task-adapter + class-weighted) scores
    #     k_t = (mean_ema / (cvar_ema + ε)) * ((p_tail_ema / α) ** beta_tail)
    # ----------------------------------------------------------
    if use_cvar_boost:
        # Append per-qid mean(after weighting) to the task’s and dataset’s buffers
        # (task-level buffer drives k_t; dataset-level is logging only)
        for qid, vals in q2norm.items():
            task    = q2tasks[qid]
            dataset = q2datasets[qid]
            w       = q2w[qid]
            # mean of (norm * weight) across rollouts for this question
            mean_q = float(np.mean(vals) * w)

            task_stats[task]["buffer"].append(mean_q)
            dataset_stats[dataset]["buffer"].append(mean_q)

        task2_k: Dict[Any, float] = {}

        # Iterate over tasks that actually appeared in this batch
        for task in task_to_rollouts.keys():
            buf = task_stats[task]["buffer"]
            if not buf:
                task2_k[task] = 1.0
                continue

            x = torch.tensor(buf, device=device, dtype=raw_scores.dtype)

            # Base stats from buffer
            mean_raw = float(x.mean().item())
            # VaR/CVaR and empirical tail mass
            q_alpha = torch.quantile(x, alpha)
            tail = x[x <= q_alpha]
            if tail.numel() == 0:
                cvar_raw   = mean_raw
                # if no tail, treat p_tail as small but non-zero
                p_tail_raw = max(1.0 / max(len(x), 1), alpha * 0.1)
            else:
                cvar_raw   = float(tail.mean().item())
                p_tail_raw = float(tail.numel()) / float(x.numel())

            # EMA smoothing of (mean, cvar, p_tail) at the TASK level
            prev_mean  = float(task_stats[task].get("buffer_mean_ema", mean_raw))
            prev_cvar  = float(task_stats[task].get("buffer_cvar_ema", cvar_raw))
            prev_ptail = float(task_stats[task].get("buffer_ptail_ema", p_tail_raw))

            task_stats[task]["buffer_mean_ema"]  = _ema_update(prev_mean,  mean_raw,  beta_mean)
            task_stats[task]["buffer_cvar_ema"]  = _ema_update(prev_cvar,  cvar_raw,  beta_cvar)
            task_stats[task]["buffer_ptail_ema"] = _ema_update(prev_ptail, p_tail_raw, beta_cvar)

            mean_t  = task_stats[task]["buffer_mean_ema"]
            cvar_t  = max(task_stats[task]["buffer_cvar_ema"], eps)
            ptail_t = max(task_stats[task]["buffer_ptail_ema"], eps)

            # CVaR×frequency ratio
            # Frequency ratio upweighs or downweights based on the number of tail events.
            freq_factor = (ptail_t / max(alpha, eps)) ** beta_tail
            k_t = (mean_t / cvar_t) * freq_factor

            # Safety clamp
            k_t = float(np.clip(k_t, 0.0, 10.0))
            task2_k[task] = k_t

        # Optional: mirror CVaR buffer stats at the DATASET level for logging only
        for dataset in dataset_to_rollouts.keys():
            buf = dataset_stats[dataset]["buffer"]
            if not buf:
                continue

            x = torch.tensor(buf, device=device, dtype=raw_scores.dtype)

            mean_raw = float(x.mean().item())
            q_alpha = torch.quantile(x, alpha)
            tail = x[x <= q_alpha]
            if tail.numel() == 0:
                cvar_raw   = mean_raw
                p_tail_raw = max(1.0 / max(len(x), 1), alpha * 0.1)
            else:
                cvar_raw   = float(tail.mean().item())
                p_tail_raw = float(tail.numel()) / float(x.numel())

            prev_mean  = float(dataset_stats[dataset].get("buffer_mean_ema", mean_raw))
            prev_cvar  = float(dataset_stats[dataset].get("buffer_cvar_ema", cvar_raw))
            prev_ptail = float(dataset_stats[dataset].get("buffer_ptail_ema", p_tail_raw))

            dataset_stats[dataset]["buffer_mean_ema"]  = _ema_update(prev_mean,  mean_raw,  beta_mean)
            dataset_stats[dataset]["buffer_cvar_ema"]  = _ema_update(prev_cvar,  cvar_raw,  beta_cvar)
            dataset_stats[dataset]["buffer_ptail_ema"] = _ema_update(prev_ptail, p_tail_raw, beta_cvar)
            # NOTE: dataset-level buffer EMAs are for analysis only;
            #       they do NOT feed back into k_t in this implementation.

        # Finally, map each qid → its task-level k_t
        q2k: Dict[Any, float] = {qid: task2_k[q2tasks[qid]] for qid in q2norm.keys()}
    else:
        q2k = {qid: 1.0 for qid in q2norm.keys()}

    # -------------------------------
    # U2) Track k_t per task and per dataset (batch + EMA of mean)
    # -------------------------------
    update_k_stats_from_q2k(
        q2k=q2k,
        q2tasks=q2tasks,
        q2datasets=q2datasets,
        beta_mean=beta_mean,
        eps=eps,
    )





### OLDEST WORKING:


@register_adv_est(AdvantageEstimator.HARPO)
def compute_harpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (B, L), where L is the response_length
    response_mask:      torch.Tensor,    # (B, L)
    index:              torch.Tensor,    # (B,) question /prompt ids (unused except parity with others)
    task_ids:           List[Any],       # (B,) task id per sample
    dataset_ids:        List[Any],       # (B,) dataset id per sample
    class_labels:       List[Any],       # (B,) class label per sample (for weighting)
    *,
    # --- Ablation toggles ---
    use_task_adapter:   bool = False,
    use_task_mixture_adapter: bool = False, 
    use_class_weights:  bool = False,
    use_cvar_boost:     bool = False,
    use_grpo_group_norm: bool = True,
    use_task_mixture_density_adapter: bool = True,
    # --- Hyperparameters ---
    eps: float = EPS_DEFAULT,
    alpha: float = 0.2,                  # CVaR tail fraction
    lambda_risk: float = 0.3,            # blend strength for dynamic tail boost
    # EMA decays
    beta_mu: float = 0.99,
    beta_sigma: float = 0.99,
    beta_mean: float = 0.98,    # per task buffer EMA for CVaR's mean references (how much of the prev to keep etc.)
    beta_cvar: float = 0.98,    # per task buffer EMA for CVaR
    beta_tail: float = 1.0,     # how strongly the frequency of tail events modulates CVaR boost
    # Static metadata for class weights
    class_count_info: Optional[Dict[Any, Dict[Any, int]]] = None,  # {dataset: {class: count}}
    class_weight_scope: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    HARPO advantage/return computation (no per-prompt batch norm).
    Pipeline (modular; combine once at the end):
        A) Build q2rollouts/q2tasks/q2datasets/q2class and per-task/dataset stats
        B) (Optional) Global min-max scaling across all rollouts
        C) (Optional) GRPO-style intra-question normalization (zero-mean/unit-variance per qid)
        D) Task-adapter centering -> q2norm[qid][k]
        E) Static inverse-frequency class weight -> q2w[qid] (scalar)
        F) CVaR dynamic tail boost -> q2k[qid] (via per-task buffer)
        G) Combine into final scalar advantages, log stats, and broadcast
    """

    device = token_level_rewards.device
    B, L   = token_level_rewards.shape

    # 1) Rollout-level scalar raw rewards
    # Token_level rewards is essentially the rewards for each token in the response
    # here, we sum them to get the total reward for the entire response
    raw_scores = token_level_rewards.sum(dim=-1)  # (B,)

    # So, essentially, we assume that the scores 
    # correspond to the batch samples in order (B,), where B is 
    # the total number of rollouts in the mini-batch, (i.e. 5 for the same question)
    # if there is 2 questions, then B = 10 (5 rollouts each)
    # and this is the same order as task_ids, dataset_ids, class_labels,
    # qid is basically the question id that each rollout corresponds to
    # i.e. index = [qid_1, qid_1, qid_1] (for a rollout of 3)
    # note that there may be more than one question in the mini-batch
    # which looks like [qid_1, qid_1, qid_1, qid_2, qid_2, qid_2] 
    # (for 2 questions, each with 3 rollouts)
    # we use qid to essentially delineate the different training examples
    use_minmax_scaling: bool = True,    # Global min-max scaling before GRPO
    (
        q2rollouts,
        q2tasks,
        q2datasets,
        q2class,
        task_to_rollouts,
        dataset_to_rollouts,
    ) = build_mappings(
        raw_scores=raw_scores,
        index=index,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
        class_labels=class_labels,
        use_minmax_scaling=use_minmax_scaling,
        eps=eps,
    )

    # NOTE: the returns at the end of this function are essentially the 
    # normalized values of the scores, which are then broadcasted to token-level
    # i.e. for each token in the response, it will now have the same normalized 
    # score as the entire response
    # [1.2], [0.5] --> [1.2, 1.2, 1.2], [0.5, 0.5, 0.5] (for response length of 3 for both responses)

    # -------------------------------------------------------------------------
    # U1) Update per-task and per-dataset stats
    #     - raw_batch_mu/raw_batch_sigma/raw_batch_count: non-EMA, for logging
    #     - ema_mu/ema_sigma/ema_count: EMA (used by task adapter)
    #
    # Note: These stats are updated AFTER optional min-max scaling (which
    # happens in build_mappings), so if use_minmax_scaling=True, these stats
    # reflect the scaled values.
    # -------------------------------------------------------------------------
    update_raw_stats(
        task_to_rollouts=task_to_rollouts,
        dataset_to_rollouts=dataset_to_rollouts,
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        eps=eps,
    )

    # -------------------------------------------------------------------------
    # C) GRPO-style intra-question normalization (zero-mean/unit-variance)
    #
    # This step operates on q2rollouts (per-qid rollout lists) and is applied
    # AFTER optional min-max scaling (in build_mappings) and BEFORE task
    # adapter, class weighting, and CVaR scaling so that those subsequent
    # scalings are not neutralized.
    #
    # Standard GRPO normalization: (v - mean) / std per question
    # -------------------------------------------------------------------------
    if use_grpo_group_norm:
        for qid, vals in q2rollouts.items():
            if len(vals) < 2:
                # With a single rollout, GRPO normalization is ill-defined; keep as is.
                continue

            # Original GRPO-style group normalization: (v - mean) / std
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=0))
            sd = max(sd, eps)
            q2rollouts[qid] = [(v - mu) / sd for v in vals]

        # ---------------------------------------------------------------------
        # U2.5) Track POST-GRPO advantages per task and per dataset
        #       (after GRPO group normalization, before task adapter)
        # ---------------------------------------------------------------------
        update_advantage_stats(
            q2_advantages=q2rollouts,
            q2tasks=q2tasks,
            q2datasets=q2datasets,
            stat_prefix="post_grpo_advantage",
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            eps=eps,
        )

        # Store post-GRPO advantages for saving to JSON
        # Note: responsibilities (q2rvalues) are not yet computed at this stage
        store_advantage_data(
            advantage_type="post_grpo",
            q2_advantages=q2rollouts,
            q2tasks=q2tasks,
            q2datasets=q2datasets,
            q2_responsibilities={}
        )

    # -------------------------------------------------------------------------
    # D.5) Task mixture-density adapter
    #
    # Features:
    #  1) Rarity boosting is toggleable
    #  2) Inter-task responsibilities use unit-advantage normalization (no z-scores)
    #  3) Optional hierarchical rollout-mixture within each task (two-sided,
    #     budget-preserving via log re-centering)
    # -------------------------------------------------------------------------

    # Initialize responsibilities dict (will be populated if mixture adapter is enabled)
    q2rvalues: Dict[Any, List[float]] = {}

    if use_task_mixture_density_adapter:

        # ----------------------------
        # Toggles
        # ----------------------------
        USE_RARITY_BOOST         = False
        USE_HIER_ROLLOUT_MIXTURE = True
        NORMALIZATION_MODE       = "no_responsibilities"  # "z_score", "global_norm", "absolute", or "no_responsibilities"
        SCALING_TYPE             = "geom"  # "geom" (geometric mean redistribution) or "naive_density" (direct division by rho_t)

        # ----------------------------
        # Hyperparameters
        # ----------------------------
        # Responsibility computation
        BETA_R   = 4.0
        DELTA_Z  = 0.25   # threshold in z space (e.g., 0.25 ~ mildly above typical) - only used for z_score mode
        A_THRESHOLD = 0.5  # threshold in absolute advantage space - only used for absolute mode

        # Inter-task density scaling
        GAMMA = 1

        # Rarity boost (optional)
        ETA_K      = 0.5
        K_MAX      = 1.5
        RARE_RATIO = 2.5

        # Hierarchical rollout mixture
        ALPHA_ROLL     = 0.5
        ROLL_MAX_SCALE = 3.0

        # EMA smoothing
        BETA_RHO     = 0.95 # NOTE: ORIGINAL VALUE IS 0.95 ; we can put as 0 for no EMA smoothing
        BETA_LOGMULT = 0.95

        # Final task-scale clamp
        MIN_SCALE = 0.5
        MAX_SCALE = 4.0

        # ----------------------------
        # Step 1 — responsibilities + task signal mass
        # ----------------------------
        task_signal_mass = defaultdict(float)
        task_sig_count   = defaultdict(float)
        task_count       = defaultdict(int)

        # Compute global mean absolute advantage if using global normalization
        global_mean_abs = None
        if NORMALIZATION_MODE == "global_norm":
            all_abs_vals = []
            for vals in q2rollouts.values():
                all_abs_vals.extend([abs(v) for v in vals])
            global_mean_abs = sum(all_abs_vals) / max(len(all_abs_vals), 1) if all_abs_vals else eps
            global_mean_abs = max(global_mean_abs, eps)

        for qid, vals in q2rollouts.items():
            task = q2tasks[qid]
            r_list = []

            if NORMALIZATION_MODE == "global_norm":
                # Global normalization: normalize by global mean |A|
                for v in vals:
                    a = abs(v)
                    # Normalize by global mean
                    norm_a = a / global_mean_abs
                    # Responsibility based on normalized advantage
                    # Use norm_a - 1.0 as input to sigmoid (values > mean have norm_a > 1)
                    r = 1.0 / (1.0 + math.exp(-BETA_R * (norm_a - 1.0)))

                    task_signal_mass[task] += r * a
                    task_sig_count[task]   += r
                    task_count[task]       += 1
                    r_list.append(r)
            elif NORMALIZATION_MODE == "z_score":
                # Z-score normalization: task-specific normalization
                # Fetch per-task EMA stats in |A| space
                mu_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_mean", 0.0))
                mu_abs = max(mu_abs, eps)

                # Prefer abs-sigma if tracked; otherwise fall back to signed sigma
                sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_sigma", 0.0))
                if sd_abs <= 0.0:
                    sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_sigma", 0.0))
                sd_abs = max(sd_abs, eps)

                for v in vals:
                    a = abs(v)

                    # standard z-score in |A| space
                    z = (a - mu_abs) / (sd_abs + eps)

                    # responsibility as soft tail membership
                    r = 1.0 / (1.0 + math.exp(-BETA_R * (z - DELTA_Z)))

                    task_signal_mass[task] += r * a
                    task_sig_count[task]   += r
                    task_count[task]       += 1
                    r_list.append(r)
            elif NORMALIZATION_MODE == "absolute":
                # Absolute mode: sigmoid applied directly to absolute advantage with threshold
                for v in vals:
                    a = abs(v)

                    # Responsibility based on raw absolute advantage with threshold
                    r = 1.0 / (1.0 + math.exp(-BETA_R * (a - A_THRESHOLD)))

                    task_signal_mass[task] += r * a
                    task_sig_count[task]   += r
                    task_count[task]       += 1
                    r_list.append(r)

            elif NORMALIZATION_MODE == "no_responsibilities":
                # No responsibilities ablation: all responsibilities set to 1.0
                # This removes responsibility weighting and uses raw absolute advantage statistics
                # so that we can test the effectiveness of the responsibilities
                for v in vals:
                    a = abs(v)

                    # Responsibility is always 1.0 (no weighting)
                    r = 1.0

                    task_signal_mass[task] += r * a  # Equivalent to: task_signal_mass[task] += a
                    task_sig_count[task]   += r      # Equivalent to: task_sig_count[task] += 1
                    task_count[task]       += 1
                    r_list.append(r)
            else:
                raise ValueError(f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}. Must be 'z_score', 'global_norm', 'absolute', or 'no_responsibilities'")

            q2rvalues[qid] = r_list

        # Store batch-level signal statistics for logging
        for task in task_signal_mass:
            task_stats[task]["mixture_signal_mass"]       = float(task_signal_mass[task])
            task_stats[task]["mixture_sig_count_batch"]   = float(task_sig_count[task])
            task_stats[task]["mixture_batch_count"]       = int(task_count[task])

        if len(task_signal_mass) == 0:
            q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}
        else:
            # ----------------------------
            # Step 2 — task densities rho_t (EMA)
            # ----------------------------
            task_densities = {}
            log_rhos = []

            for task, mass in task_signal_mass.items():
                count = max(task_count[task], 1)
                rho_batch = max(mass / count, eps)

                prev_rho = float(task_stats[task].get("mixture_rho_t_ema", rho_batch))
                rho_ema  = _ema_update(prev_rho, rho_batch, BETA_RHO)
                rho_ema  = max(rho_ema, eps)

                task_stats[task]["mixture_rho_t_batch"] = float(rho_batch)
                task_stats[task]["mixture_rho_t_ema"]   = float(rho_ema)

                task_densities[task] = rho_ema
                log_rhos.append(math.log(rho_ema))

            # geometric mean of task densities
            log_rho_ref = sum(log_rhos) / len(log_rhos)
            rho_ref     = math.exp(log_rho_ref)

            for task in task_signal_mass:
                task_stats[task]["mixture_rho_ref"] = float(rho_ref)

            # ----------------------------
            # Step 3 — rarity boost (optional)
            # ----------------------------
            task_k = defaultdict(lambda: 1.0)

            if USE_RARITY_BOOST:
                for task in task_signal_mass:
                    sig_batch = max(task_sig_count[task], eps)
                    prev = float(task_stats[task].get("mixture_sig_count_ema", sig_batch))
                    sig_ema = _ema_update(prev, sig_batch, BETA_RHO)
                    task_stats[task]["mixture_sig_count_ema"] = float(sig_ema)
                    task_stats[task]["mixture_sig_count"]     = float(sig_batch)

                sig_emas = [
                    max(float(task_stats[t]["mixture_sig_count_ema"]), eps)
                    for t in task_signal_mass
                ]
                sig_ref = math.exp(sum(math.log(x) for x in sig_emas) / len(sig_emas))

                for task in task_signal_mass:
                    n_sig = max(task_stats[task]["mixture_sig_count_ema"], eps)
                    ratio = sig_ref / n_sig
                    excess = max(0.0, ratio - RARE_RATIO)

                    log_k = ETA_K * math.log1p(excess)
                    log_k = min(log_k, math.log(K_MAX))
                    task_k[task] = math.exp(log_k)

                    task_stats[task]["mixture_sig_ref"]          = float(sig_ref)
                    task_stats[task]["mixture_log_rarity_raw"]   = float(math.log(ratio))
                    task_stats[task]["mixture_log_rarity_pos"]   = float(excess)
                    task_stats[task]["mixture_rarity_ratio"]     = float(ratio)
                    task_stats[task]["mixture_rarity_excess"]    = float(excess)
                    task_stats[task]["mixture_k_t"]              = float(task_k[task])
            else:
                for task in task_signal_mass:
                    task_stats[task]["mixture_sig_count"]        = 0.0
                    task_stats[task]["mixture_sig_ref"]          = 0.0
                    task_stats[task]["mixture_log_rarity_raw"]   = 0.0
                    task_stats[task]["mixture_log_rarity_pos"]   = 0.0
                    task_stats[task]["mixture_rarity_ratio"]     = 0.0
                    task_stats[task]["mixture_rarity_excess"]    = 0.0
                    task_stats[task]["mixture_k_t"]              = 1.0

            # ----------------------------
            # Step 4 — task-scale s_t (EMA)
            # ----------------------------
            task_scale = {}

            for task in task_signal_mass:
                rho_t = max(task_densities[task], eps)
                log_ratio = log_rho_ref - math.log(rho_t)
                log_mult_inst = GAMMA * log_ratio + math.log(task_k[task])

                prev = float(task_stats[task].get("mixture_log_mult_ema_scale", 0.0))
                log_mult_ema = _ema_update(prev, log_mult_inst, BETA_LOGMULT)

                scale = math.exp(log_mult_ema)
                scale = max(MIN_SCALE, min(MAX_SCALE, scale))

                task_scale[task] = scale

                task_stats[task]["mixture_log_ratio_scale"]      = float(log_ratio)
                task_stats[task]["mixture_log_mult_inst_scale"]  = float(log_mult_inst)
                task_stats[task]["mixture_log_mult_ema_scale"]   = float(log_mult_ema)
                task_stats[task]["mixture_final_scale"]          = float(scale)

            # ----------------------------
            # Step 5 — hierarchical rollout-mixture (two-sided, budget-preserving)
            # ----------------------------
            qid_rollout_scale = defaultdict(lambda: 1.0)

            if USE_HIER_ROLLOUT_MIXTURE:
                task2qids = defaultdict(list)
                for qid in q2rollouts:
                    task2qids[q2tasks[qid]].append(qid)

                for task, qids in task2qids.items():
                    if len(qids) < 2:
                        continue

                    qid2m = {}
                    log_m = []

                    # computing the per rolloutout m , which is essentially the density of the signal mass
                    for qid in qids:
                        vals = q2rollouts[qid]
                        rs   = q2rvalues[qid]
                        m = sum(r * abs(v) for r, v in zip(rs, vals)) / max(len(vals), 1)
                        m = max(m, eps)
                        qid2m[qid] = m
                        log_m.append(math.log(m))


                    # For a specific task, what's the average log m
                    # across its questions
                    log_mbar = sum(log_m) / len(log_m)

                    log_s_raw = {
                        qid: ALPHA_ROLL * (log_mbar - math.log(qid2m[qid]))
                        for qid in qids
                    }

                    mean_log_s = sum(log_s_raw.values()) / len(log_s_raw)

                    for qid in qids:
                        log_s = log_s_raw[qid] - mean_log_s
                        if ROLL_MAX_SCALE is not None:
                            cap = math.log(ROLL_MAX_SCALE)
                            log_s = max(-cap, min(cap, log_s))
                        qid_rollout_scale[qid] = math.exp(log_s)

            # ----------------------------
            # Step 6 — apply final scaling
            # ----------------------------
            q2norm = {}
            for qid, vals in q2rollouts.items():
                task = q2tasks[qid]
                s_t  = task_scale.get(task, 1.0)
                s_q  = qid_rollout_scale.get(qid, 1.0)
                q2norm[qid] = [v * s_t * s_q for v in vals]

    else:
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}


    # --------------------------------------------
    # G) Combine once at the end, with ablations baked in:
    #    Let base = (norm_or_raw) * (weight_or_1)
    #    Effective lambda: λ_eff = λ_risk if CVaR enabled else 0
    #    Effective k:      k_eff = k_t     if CVaR enabled else 1
    #
    #    score = (1 - λ_eff) * base + λ_eff * (k_eff * base)
    # --------------------------------------------

    q2_final: Dict[str, List[float]] = defaultdict(list)
    qrolloutpos: Dict[Any, int] = defaultdict(int)  # store the positions of the rollouts that we are at

    for i in range(B):
        qid = index[i]
        j   = qrolloutpos[qid]

        # 1) Norm or raw (already prepared upstream)
        norm_or_raw = q2norm[qid][j]

        s_final = norm_or_raw

        q2_final[qid].append(float(s_final))
        qrolloutpos[qid] += 1

    # --------------------------------------------
    # G) Rebuild tensor in original rollout order
    # --------------------------------------------
    final_scores = torch.empty_like(raw_scores)
    qfinalrolloutpos: Dict[Any, int] = defaultdict(int)

    for i in range(B):
        qid = index[i]
        j   = qfinalrolloutpos[qid]
        final_scores[i] = torch.tensor(q2_final[qid][j], device=device, dtype=raw_scores.dtype)
        qfinalrolloutpos[qid] += 1

    # --------------------------------------------
    # U3) Update FINAL HARPO advantages per task and per dataset
    # --------------------------------------------
    update_advantage_stats(
        q2_advantages=q2_final,
        q2tasks=q2tasks,
        q2datasets=q2datasets,
        stat_prefix="final_advantage",
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        eps=eps,
    )

    # Store final advantages and responsibilities for saving to JSON
    store_advantage_data(
        advantage_type="final",
        q2_advantages=q2_final,
        q2tasks=q2tasks,
        q2datasets=q2datasets,
        q2_responsibilities=q2rvalues
    )

    # --------------------------------------------
    # H) Broadcast to token level and return
    # --------------------------------------------
    returns = final_scores.unsqueeze(-1) * response_mask

    return returns, returns