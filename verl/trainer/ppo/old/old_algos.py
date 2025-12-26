
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




    # -------------------------------------------------------------------------
    # D) Task-adapter centering
    #
    # Previously: (v - ema_mu) / ema_sigma
    # Now:        (v / ema_mu) ONLY (no division by ema_sigma).
    # -------------------------------------------------------------------------
    if use_task_adapter:
        # # Center each qid’s rollouts with its task’s EMA mean (produce q2norm)
        # q2norm: Dict[Any, List[float]] = {}
        # for qid, vals in q2rollouts.items():
        #     task = q2tasks[qid]
        #     mu_t = float(task_stats[task]["ema_mu"])
        #     # Mean-only centering; no variance scaling.
        #     # q2norm[qid] = [(v - mu_t) for v in vals]
        #     # Rough Mean scaling (to downweight high-performing tasks)
        #     q2norm[qid] = [(v / mu_t) for v in vals]

        # 1) Compute global reference mean and sigma across tasks
        task_mus = [float(stats["ema_mu"]) for stats in task_stats.values()]
        task_sigmas = [float(stats["post_grpo_advantage_ema_sigma"]) for stats in task_stats.values()]

        mu_ref = sum(task_mus) / max(len(task_mus), 1)
        sigma_ref = sum(task_sigmas) / max(len(task_sigmas), 1)

        # 2) Reasonable bounds so we don't explode or vanish
        MIN_SCALE = 0.3     # at most 3x downweight
        MAX_SCALE = 3.0      # at most 3x upweight
        EPS = 1e-6
        SCALE_COEFF = 2.0
        SIGMA_WEIGHT = 0.8   # 0 = pure μ scaling, 1 = pure σ scaling, 0.5 = balanced

        q2norm: Dict[Any, List[float]] = {}
        for qid, vals in q2rollouts.items():
            task = q2tasks[qid]
            mu_t = float(task_stats[task]["ema_mu"])
            sigma_t = float(task_stats[task]["post_grpo_advantage_ema_sigma"])

            # μ-based scaling: >1 if task underperforms, <1 if overperforms
            raw_mu_scale = mu_ref / max(mu_t, EPS)

            # σ-based scaling: >1 if "too quiet" (low variance), <1 if "too loud" (high variance)
            raw_sigma_scale = sigma_ref / max(sigma_t, EPS)

            # Blend: SIGMA_WEIGHT interpolates between mu-only (0) and sigma-only (1)
            # raw_scale = (1 - SIGMA_WEIGHT) * raw_mu_scale + SIGMA_WEIGHT * raw_sigma_scale
            # But multiplicative blend is more stable:
            raw_scale = ((raw_mu_scale ** (1 - SIGMA_WEIGHT)) * (raw_sigma_scale ** SIGMA_WEIGHT)) ** SCALE_COEFF
            
            # Clamp overall scale
            scale_t = max(MIN_SCALE, min(MAX_SCALE, raw_scale))

            # Track scaling factors for this task
            task_stats[task]["adapter_mu_scale"] = float(raw_mu_scale)
            task_stats[task]["adapter_sigma_scale"] = float(raw_sigma_scale)
            task_stats[task]["adapter_final_scale"] = float(scale_t)

            # Scale_t nudges tasks up/down
            q2norm[qid] = [(v) * scale_t for v in vals]

    else:
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}



    if use_task_mixture_adapter:
        # --- Mixture-based sparsity adapter (simple version) ---
        # For each task t:
        #   ρ_t ≈ post_grpo_advantage_ema_abs_mean / |post_grpo_advantage_ema_p90|
        # Then scale advantages by exp( log(ρ_ref) − log(ρ_t) ), clamped.

        task_to_density: Dict[Any, float] = {}
        densities: List[float] = []

        # 1) Compute ρ_t for every task from EMA stats
        for task, stats in task_stats.items():
            abs_mean_ema = float(stats.get("post_grpo_advantage_ema_abs_mean", 0.0))
            p90_ema      = float(stats.get("post_grpo_advantage_ema_p90", 0.0))

            tail_scale = max(abs(p90_ema), eps)

            # RHO_T is actually that of sparsity.
            # Greater rho_t greater sparsity.

            if tail_scale > 0.0:
                rho_t = abs_mean_ema / tail_scale
            else:
                rho_t = 0.0

            task_to_density[task] = rho_t
            densities.append(rho_t)

        # Assume we always have at least one task; use all densities to define reference.
        # 2) Global reference density (geometric mean)
        log_rhos    = [math.log(r + eps) for r in densities]
        log_rho_ref = sum(log_rhos) / len(log_rhos)
        rho_ref     = math.exp(log_rho_ref)  # mostly for logging / inspection

        # LOG_RHO_MAX = 2.0   # clamp log-ratio (~ up to ~7.4x)
        SCALE_COEFF = 1.0
        MIN_SCALE   = 0.25
        MAX_SCALE   = 8.0

        q2norm = {}
        for qid, vals in q2rollouts.items():
            task  = q2tasks[qid]

            rho_t = task_to_density.get(task, 0.0)
            rho_t = max(rho_t, eps)  # avoid log(0)

            log_rho_t = math.log(rho_t)
            # >0 ⇒ task is sparser (lower density) than reference ⇒ boost
            log_sparsity_ratio = log_rho_t - log_rho_ref

            log_scale = SCALE_COEFF * log_sparsity_ratio
            raw_scale = math.exp(log_scale)
            scale_t   = max(MIN_SCALE, min(MAX_SCALE, raw_scale))

            task_stats[task]["mixture_rho_t"]     = float(rho_t)
            task_stats[task]["mixture_rho_ref"]     = float(rho_ref)
            task_stats[task]["mixture_log_scale"]   = float(log_scale)
            task_stats[task]["mixture_final_scale"] = float(scale_t)

            q2norm[qid] = [v * scale_t for v in vals]

    else:
        # No task adapter at all
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}

    
    # -------------------------------------------------------------------------
    # D.5) Task mixture-density adapter (inter-task scaling only)
    #      - Uses soft responsibilities r(z) to estimate per-task "signal mass"
    #      - Uses STANDARD z-score in |A| space: z = (|A| - mu_abs) / (sigma_abs + eps)
    #      - rho_t = E[r * |A|] (count-normalized), EMA-smoothed
    #      - Scales by density-ratio + one-sided rarity boost, with EMA on log-mult
    # -------------------------------------------------------------------------
    if use_task_mixture_density_adapter:
        # ----------------------------
        # Hyperparams (tune these)
        # ----------------------------
        BETA_R      = 4.0     # responsibility sharpness
        DELTA_R     = 0.25    # z threshold (in standard z space)
        GAMMA       = 0.5     # density-ratio temperature (<1 helps avoid saturation)

        # Rarity boost (mixture-derived, one-sided)
        ETA_K       = 0.5     # rarity temperature
        K_MAX       = 1.5    # only boosts up to this (never downweights common tasks)

        # EMA smoothing
        BETA_RHO      = 0.95  # EMA for the per-task density (rho)
        BETA_LOGMULT  = 0.95  # EMA for the per-task log multiplier

        # Final scale clamp (applied directly to scale_t)
        MIN_SCALE   = 0.5
        MAX_SCALE   = 8.0

        # NEW: boost only if task is at least this many times rarer than reference
        RARE_RATIO  = 2.5    # e.g., 20x rarer than reference => "super rare only"

        # ----------------------------
        # Step 1 — compute per-task signal mass and effective signal count
        # ----------------------------
        task_signal_mass = defaultdict(float)  # sum_i r_i * |A_i|
        task_count       = defaultdict(int)    # raw sample count (for logging)
        task_sig_count   = defaultdict(float)  # sum_i r_i (effective signal count)

        q2rvalues = {}  # optional: store r values per qid

        for qid, vals in q2rollouts.items():
            task = q2tasks[qid]

            # STANDARD z-score stats in |A| space (EMA)
            mu_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_mean", 0.0))

            # Prefer abs-sigma if you tracked it; otherwise fall back to signed sigma.
            # Recommended: track post_grpo_advantage_ema_abs_sigma in your stats updater.
            sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_sigma", 0.0))
            if sd_abs <= 0.0:
                sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_sigma", 1.0))
            sd_abs = max(sd_abs, eps)

            r_list = []
            for v in vals:
                a = abs(v)

                # standard z-score on |A|
                z = (a - mu_abs) / (sd_abs + eps)

                # soft responsibility: probability of being "signal"
                r = 1.0 / (1.0 + math.exp(-BETA_R * (z - DELTA_R)))

                task_signal_mass[task] += r * a
                task_sig_count[task]   += r
                task_count[task]       += 1

                r_list.append(r)

            q2rvalues[qid] = r_list

        # If no tasks (shouldn't happen), just passthrough
        if len(task_signal_mass) == 0:
            q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}
        else:
            # ----------------------------
            # Step 2 — compute density rho_t and EMA smooth it
            #         rho_batch = (sum r*|A|) / count
            # ----------------------------
            task_densities = {}
            log_rhos = []

            for task, mass in task_signal_mass.items():
                count = max(task_count[task], 1)
                rho_batch = max(mass / count, eps)

                # EMA update for the rho density for each task, which is essentially
                # the signal mass per sample
                prev_rho = float(task_stats[task].get("mixture_rho_ema", rho_batch))
                rho_ema  = _ema_update(prev_rho, rho_batch, BETA_RHO)
                rho_ema  = max(float(rho_ema), eps)

                task_stats[task]["mixture_rho_batch"] = float(rho_batch)  # Log batch-level density
                task_stats[task]["mixture_rho_ema"] = float(rho_ema)
                task_densities[task] = rho_ema
                log_rhos.append(math.log(rho_ema))

            # Reference density across tasks (geometric mean in log-space)
            log_rho_ref = sum(log_rhos) / max(len(log_rhos), 1)
            rho_ref     = math.exp(log_rho_ref)

            # ----------------------------
            # Step 3 — one-sided rarity boost k_t from effective signal counts (intuitive)
            #         boost only if (sig_ref / n_sig) >= RARE_RATIO
            #         Uses EMA-smoothed signal counts for stability
            # ----------------------------
            # First pass: update EMA of signal counts
            for task in task_signal_mass.keys():
                sig_count_batch = max(task_sig_count[task], eps)
                prev_ema = task_stats[task].get("mixture_sig_count_ema", 0.0)

                if prev_ema == 0.0:
                    # First time seeing this task - initialize EMA with batch value
                    sig_count_ema = sig_count_batch
                else:
                    # EMA update with BETA_RHO (same as density EMA)
                    sig_count_ema = _ema_update(prev_ema, sig_count_batch, BETA_RHO)

                task_stats[task]["mixture_sig_count_ema"] = float(sig_count_ema)

            # Compute reference using EMA values for smoothness
            sig_counts_ema = [max(task_stats[t]["mixture_sig_count_ema"], eps) for t in task_signal_mass.keys()]
            log_sig_ref = sum(math.log(x) for x in sig_counts_ema) / max(len(sig_counts_ema), 1)
            sig_ref     = math.exp(log_sig_ref)

            task_k = {}
            for task in task_signal_mass.keys():
                # Use EMA for rarity computation (smoother than batch)
                n_sig_ema = max(task_stats[task]["mixture_sig_count_ema"], eps)

                # Intuitive rarity ratio in normal space:
                # >1 means rarer-than-reference; 20 means 20x rarer.
                rarity_ratio = sig_ref / n_sig_ema

                # One-sided threshold: only boost if "super rare"
                rarity_excess = max(0.0, rarity_ratio - RARE_RATIO)

                # Smooth, saturating growth in log-space to avoid blow-ups for extreme rarity
                # (0 if below threshold)
                log_k = ETA_K * math.log1p(rarity_excess)
                log_k = min(math.log(K_MAX), log_k)
                k_t   = math.exp(log_k)

                task_k[task] = float(k_t)

                # logging (batch count + EMA + computed values)
                task_stats[task]["mixture_sig_count"]         = float(task_sig_count[task])  # batch value
                task_stats[task]["mixture_sig_ref"]           = float(sig_ref)
                task_stats[task]["mixture_rarity_ratio"]      = float(rarity_ratio)
                task_stats[task]["mixture_rarity_excess"]     = float(rarity_excess)
                task_stats[task]["mixture_k_t"]               = float(k_t)

            # ----------------------------
            # Step 4 — compute per-task log multiplier, EMA it, apply, then clamp
            #         log_mult = GAMMA*(log rho_ref - log rho_t) + log k_t
            # ----------------------------
            q2norm = {}
            for qid, vals in q2rollouts.items():
                task = q2tasks[qid]

                rho_t = max(task_densities.get(task, eps), eps)

                # density log-ratio: sparse (low rho) => positive => boost
                log_ratio = log_rho_ref - math.log(rho_t)
                log_scale_density = GAMMA * log_ratio

                k_t = float(task_k.get(task, 1.0))
                log_k = math.log(max(k_t, eps))

                # instantaneous controller signal
                log_mult_inst = log_scale_density + log_k

                # EMA smooth log multiplier (per task)
                prev_log_mult = float(task_stats[task].get("mixture_log_mult_ema", 0.0))
                log_mult_ema  = _ema_update(prev_log_mult, float(log_mult_inst), BETA_LOGMULT)

                # exponentiate to get scale, then clamp directly
                scale_t_unclamped = math.exp(float(log_mult_ema))
                scale_t = max(MIN_SCALE, min(MAX_SCALE, scale_t_unclamped))

                # logging
                task_stats[task]["mixture_rho_t"]            = float(rho_t)
                task_stats[task]["mixture_rho_ref"]          = float(rho_ref)
                task_stats[task]["mixture_log_ratio"]        = float(log_ratio)
                task_stats[task]["mixture_log_mult_inst"]    = float(log_mult_inst)
                task_stats[task]["mixture_log_mult_ema"]     = float(log_mult_ema)
                task_stats[task]["mixture_final_scale"]      = float(scale_t)
                task_stats[task]["mixture_signal_mass"]      = float(task_signal_mass[task])
                task_stats[task]["mixture_batch_count"]      = int(task_count[task])
                task_stats[task]["mixture_sig_count_batch"]  = float(task_sig_count[task])

                q2norm[qid] = [v * scale_t for v in vals]

    else:
        # No adapter → identity scaling
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}
