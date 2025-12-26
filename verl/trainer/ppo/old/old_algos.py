
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