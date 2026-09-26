| optimizer | final rl2_error_path | best | steps | time (s) | run | hyperparameters |
|---|---|---|---|---|---|---|
| SGD | 3.133e-02 | 3.123e-02 | 40851 | 3000 | bsb-100d-sweeps/astral-sweep-4 (4joen64r) | lr=0.0179, momentum=0.902 |
| Adam | 1.483e-02 | 1.483e-02 | 40850 | 3000 | bsb-100d-sweeps/glorious-sweep-14 (hyd3yurf) | lr=0.00195, eps=1e-08 |
| KFAC | 1.313e-02 | 1.313e-02 | 3312 | 3000 | bsb-100d-sweeps/driven-sweep-14 (yoc1uk6d) | T_inv=1, T_kfac=1, damping=8.56e-06, ggn_type=type-2, momentum=0, inv_dtype=torch.float64, ema_factor=0.95, kfac_approx=expand, inv_strategy=invert kronecker sum, damping_heuristic=same, initialize_to_identity=False |
| Hessian-free | 7.552e-02 | 3.532e-02 | 398 | 3001 | bsb-100d-sweeps/charmed-sweep-16 (m6bkpk7w) | lr=1, damping=0.606, verbose=False, cg_decay_x0=0.95, cg_max_iter=175, adapt_damping=True, curvature_opt=ggn, use_linesearch=True, use_cg_backtracking=True |
| ENGD (Woodbury) | 1.243e-02 | 1.137e-02 | 2535 | 3000 | bsb-100d-sweeps/radiant-sweep-6 (h1xg6lfg) | damping=7.99e-11, momentum=0, rank_val=0, approximation=exact, norm_constraint=0.001 |
