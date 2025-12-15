
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