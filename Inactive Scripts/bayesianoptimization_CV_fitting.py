# ======================== BAYESIAN OPTIMIZATION ADD-ON =========================
# Tries scikit-optimize (Bayesian). Falls back to SciPy differential_evolution.
from skopt import gp_minimize
from skopt.space import Real


# ----- 1) User: provide your array of GHad values (eV) you want to fit over -----
GHad_eV_list = np.array([
    -0.45, -0.425, -0.40, -0.375, -0.35, -0.325, -0.30
])  # <--- EDIT ME

# ----- 2) Helper: simulate for (kV_rel_log10, beta, this_GHad_eV) and return 1 - R^2 -----
def simulate_and_score(kV_rel_log10, beta_scalar, GHad_eV_val):
    """
    kV_rel_log10: search variable; k_V = k_V_RDS * 10**kV_rel_log10
    beta_scalar : single beta used for both steps
    GHad_eV_val : selected ΔG_Had in eV for this evaluation
    Returns: 1 - R^2 (minimize) so BO doesn't need 'maximize'
    """
    # Make sure these are visible/updated inside
    global k_V, k_T, k_H, GHad_eV, GHad, beta

    try:
        # Update parameters
        k_V = k_V_RDS * (10.0 ** kV_rel_log10)
        if mechanism_choice == 0:
            k_T = k_V * 1000.0
        else:
            k_H = k_V * 1000.0

        # single beta used everywhere in your rate expressions
        if isinstance(beta, list):
            beta[:] = [beta_scalar, beta_scalar]
        else:
            # if user later changes type, still keep both indices accessible
            beta = [beta_scalar, beta_scalar]

        # Set this GHad and its Joule conversion
        GHad_eV = float(GHad_eV_val)
        GHad = GHad_eV * AvoNum * conversion_factor

        # Re-run model
        soln_local = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF')
        if (not soln_local.success) or (soln_local.y.shape[1] != len(t)):
            return np.inf

        r0_vals_local = np.array([rates_r0(time, th) for time, th in zip(t, soln_local.y.T)])
        volmer_rate_local = r0_vals_local[:, 0]
        curr_model_local = volmer_rate_local * -F * 1000.0  # mA/cm²

        V_model_local = np.array([potential(ti) for ti in t])
        model_mask_local = (V_model_local <= mask_max) & (V_model_local >= mask_min)

        # guard for pathological masks
        if model_mask_local.sum() < 8 or len(V_exp_masked) < 8:
            return np.inf

        V_model_masked_local = V_model_local[model_mask_local]
        curr_model_masked_local = curr_model_local[model_mask_local]

        # Interpolate model onto experimental V grid
        interp_local = interp1d(V_model_masked_local, curr_model_masked_local,
                                kind='linear', fill_value='extrapolate', assume_sorted=False)
        I_model_interp_local = interp_local(V_exp_masked)

        r2_local = r_squared(I_exp_masked[4:], I_model_interp_local[4:])
        # Minimize (1 - R^2)
        return 1.0 - float(r2_local)

    except Exception:
        # any numerical failure = bad candidate
        return np.inf

# ----- 3) Optimization harness over GHad_eV_list -----
results = []
print("\n=== Parameter fitting with Bayesian optimization (per GHad) ===")
for g in GHad_eV_list:
    print(f"\nOptimizing for GHad = {g:.3f} eV ...")

    # Search spaces:
    #   - kV_rel_log10 in [-3, 3]  -> i.e., k_V in [k_V_RDS * 1e-3, k_V_RDS * 1e+3]
    #   - beta in [0.20, 0.90]
    if HAVE_SKOPT:
        space = [Real(-3.0, 3.0, name="kV_rel_log10"),
                 Real(0.20, 0.90, name="beta")]
        # Wrap to match skopt signature
        def obj_skopt(x):
            return simulate_and_score(x[0], x[1], g)

        res = gp_minimize(
            obj_skopt,
            dimensions=space,
            n_calls=30,            # increase if you want more thorough search
            n_initial_points=6,    # random starts before GP model kicks in
            acq_func="EI",
            random_state=0
        )
        best_rel, best_beta = res.x
        best_score = res.fun
    else:
        # Fallback: SciPy global optimizer
        bounds = [(-3.0, 3.0), (0.20, 0.90)]
        def obj_de(x):
            return simulate_and_score(x[0], x[1], g)
        res = differential_evolution(obj_de, bounds=bounds, seed=0, maxiter=60, tol=1e-3)
        best_rel, best_beta = res.x
        best_score = res.fun

    best_kV = k_V_RDS * (10.0 ** best_rel)
    best_R2 = 1.0 - best_score
    print(f"  -> best k_V = {best_kV:.6g}, best beta = {best_beta:.3f}, R² = {best_R2:.4f}")
    results.append((g, best_kV, best_beta, best_R2))

# ----- 4) Save & display a tidy results table -----
df_bo = pd.DataFrame(results, columns=["GHad_eV", "k_V_best", "beta_best", "R2_best"])
print("\nBayesian optimization results:")
print(df_bo)
df_bo.to_excel("bo_results.xlsx", index=False)
print("Saved: bo_results.xlsx")

# ----- 5) (Optional) Re-run model at the last GHad using the found best params and re-plot overlay -----
if len(GHad_eV_list) > 0:
    g_last, kV_last, beta_last, _ = results[-1]
    # set params
    k_V = kV_last
    if mechanism_choice == 0:
        k_T = k_V * 1000.0
    else:
        k_H = k_V * 1000.0
    if isinstance(beta, list):
        beta[:] = [beta_last, beta_last]
    else:
        beta = [beta_last, beta_last]
    GHad_eV = float(g_last)
    GHad = GHad_eV * AvoNum * conversion_factor

    # simulate & plot on top of experimental, like your existing plot
    soln_last = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF')
    r0_vals_last = np.array([rates_r0(time, th) for time, th in zip(t, soln_last.y.T)])
    curr_model_last = r0_vals_last[:, 0] * -F * 1000.0
    V_model_last = np.array([potential(ti) for ti in t])

    model_mask_last = (V_model_last <= mask_max) & (V_model_last >= mask_min)
    V_m_last = V_model_last[model_mask_last]
    I_m_last = curr_model_last[model_mask_last]
    interp_last = interp1d(V_m_last, I_m_last, kind='linear', fill_value='extrapolate', assume_sorted=False)
    I_model_interp_last = interp_last(V_exp_masked)

    r2_last = r_squared(I_exp_masked[4:], I_model_interp_last[4:])
    fig, axs = plt.subplots(figsize=(8, 10))
    axs.plot(V_model_last[4:], curr_model_last[4:], 'r', label='Model (best fit)')
    axs.plot(V_exp_masked, I_exp_masked, 'b', label=f'Exp H2 Sat, {scan_id}')
    axs.set_xlabel('Voltage vs. RHE (V)')
    axs.set_ylabel('Kinetic current (mA/cm²)')
    axs.set_title(fr'Best Fit at GHad={g_last:.3f} eV, $R^2$ = {r2_last:.4f}')
    axs.grid(); axs.legend(); plt.show()
# ====================== END: BAYESIAN OPTIMIZATION ADD-ON ======================


