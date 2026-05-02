import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Parameters
J_TRUE = 2.5
KAPPA  = 0.3
ETA    = 0.8
J_INIT = 1.0      # 60% initial error

DT      = 0.001
T_TOTAL = 5000.0
N       = int(T_TOTAL / DT)
N_BURN  = int(0.3 * N)
N_ITER  = 7

TAU0 = np.pi / (4.0 * np.sqrt(J_TRUE**2 + 1.0))
LAG  = int(TAU0 / DT)

print("=" * 60)
print("  NSPT Qubit J Estimation  (corrected)")
print("=" * 60)
print(f"  J_true={J_TRUE}, k={KAPPA}, eta={ETA}, J_init={J_INIT}")
print(f"  omega = 2sqrt(J^2+1) = {2*np.sqrt(J_TRUE**2+1):.4f}")
print(f"  tau_0  = pi/(4omega) = {TAU0:.4f}  (lag={LAG} steps)")
print()

# Helpers
def soft_norm(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    if r > 1.0:
        x /= r; y /= r; z /= r
    return x, y, z

# Data generation
def generate_data(J, kappa, eta, dt, N, seed=42):
    rng  = np.random.default_rng(seed)
    c    = 2.0 * np.sqrt(eta * kappa)
    sqdt = np.sqrt(dt)
    x, y, z = 0.0, 0.0, 1.0

    z_traj = np.empty(N)
    for i in range(N):
        dW = rng.standard_normal() * sqdt
        dx = (-2.0*y - 2.0*kappa*x)             * dt - c*x*z * dW
        dy = ( 2.0*x - 2.0*J*z - 2.0*kappa*y)   * dt - c*y*z * dW
        dz = ( 2.0*J*y)                          * dt + c*(1.0-z**2) * dW
        x += dx; y += dy; z += dz
        x, y, z = soft_norm(x, y, z)
        z_traj[i] = z
    return z_traj


def C_from_trajectory(z_traj, lag, n_burn):
    z = z_traj[n_burn:]
    n = len(z) - lag
    return np.mean(z[:n] * z[lag:n+lag])


def SE_from_trajectory(z_traj, lag, n_burn):
    z  = z_traj[n_burn:]
    n  = len(z) - lag
    products = z[:n] * z[lag:n+lag]
    return np.std(products) / np.sqrt(n)


# Shot noise SE, shown for comparison to demonstrate why raw dy fails
def shot_noise_SE(eta, kappa, dt, n_eff):
    return 1.0 / (4.0 * eta * kappa * dt * np.sqrt(n_eff))


# NSPT integrator
def run_nspt(J_star, kappa, eta, dt, N, seed=99):
    rng  = np.random.default_rng(seed)
    c    = 2.0 * np.sqrt(eta * kappa)
    sqdt = np.sqrt(dt)

    x0, y0, z0 = 0.0, 0.0, 1.0
    x1, y1, z1 = 0.0, 0.0, 0.0

    z0_arr = np.empty(N)
    z1_arr = np.empty(N)

    for i in range(N):
        dW = rng.standard_normal() * sqdt

        # Order 0
        dx0 = (-2.0*y0 - 2.0*kappa*x0)                  * dt - c*x0*z0 * dW
        dy0 = ( 2.0*x0 - 2.0*J_star*z0 - 2.0*kappa*y0)  * dt - c*y0*z0 * dW
        dz0 = ( 2.0*J_star*y0)                           * dt + c*(1.0-z0**2) * dW

        # Order 1
        dx1 = (-2.0*y1 - 2.0*kappa*x1)                          * dt \
              - c*(x1*z0 + x0*z1) * dW

        dy1 = ( 2.0*x1 - 2.0*J_star*z1 - 2.0*z0 - 2.0*kappa*y1) * dt \
              - c*(y1*z0 + y0*z1) * dW

        dz1 = ( 2.0*J_star*y1 + 2.0*y0)                          * dt \
              - 2.0*c*z0*z1 * dW

        x0 += dx0; y0 += dy0; z0 += dz0
        x1 += dx1; y1 += dy1; z1 += dz1
        x0, y0, z0 = soft_norm(x0, y0, z0)

        z0_arr[i] = z0
        z1_arr[i] = z1

    return z0_arr, z1_arr


def nspt_observables(z0, z1, lag, n_burn):
    z0s = z0[n_burn:]; z1s = z1[n_burn:]
    n   = len(z0s) - lag
    C0  = np.mean(z0s[:n] * z0s[lag:n+lag])
    C1  = (np.mean(z0s[:n] * z1s[lag:n+lag])
         + np.mean(z1s[:n] * z0s[lag:n+lag]))
    return C0, C1


# Newton estimator
def newton_estimate(J_init, C_target, kappa, eta, dt, N, n_burn, lag,
                    n_iter=7, seed_base=99):
    J_star  = J_init
    history = [J_star]
    C0_hist = []; C1_hist = []

    print(f"  {'Iter':>4}  {'J*':>9}  {'C_0':>12}  {'C_1=∂C/∂J':>12}  {'dJ':>10}")
    print("  " + "─" * 54)

    for it in range(n_iter):
        # Fixed seed per iteration, same noise realization for convergence
        z0, z1 = run_nspt(J_star, kappa, eta, dt, N, seed=seed_base)
        C0, C1 = nspt_observables(z0, z1, lag, n_burn)
        C0_hist.append(C0); C1_hist.append(C1)

        if abs(C1) < 1e-12:
            print("  C1 ~ 0 — Newton singular."); break

        dJ    = (C_target - C0) / C1
        J_new = J_star + dJ
        print(f"  {it+1:>4}  {J_star:>9.5f}  {C0:>12.7f}  {C1:>12.7f}  {dJ:>10.6f}")

        history.append(J_new)
        J_star = J_new

    return J_star, history, C0_hist, C1_hist


# Run
print("─ Step 1 — Generating data ──────")
z_data = generate_data(J_TRUE, KAPPA, ETA, DT, N)

C_target = C_from_trajectory(z_data, LAG, N_BURN)
C_target_SE = SE_from_trajectory(z_data, LAG, N_BURN)

n_eff = int(N * 0.7) - LAG
sn_SE = shot_noise_SE(ETA, KAPPA, DT, n_eff)

print(f"  C_z^data(tau_0) = {C_target:.6f} +-  {C_target_SE:.6f}  (from z_traj)")
print(f"  Shot-noise SE if using raw dy: +- {sn_SE:.3f}  <- unusable")
print(f"  SNR improvement: {sn_SE/C_target_SE:.0f}x\n")

print("── Step 2 — Sanity check at J_true ────────────────────────")
z0_s, z1_s = run_nspt(J_TRUE, KAPPA, ETA, DT, N, seed=42)
C0_s, C1_s = nspt_observables(z0_s, z1_s, LAG, N_BURN)
print(f"  C0 at J_true = {C0_s:.6f}  (should match C_target={C_target:.6f})")
print(f"  Difference   = {abs(C0_s-C_target):.2e}  (Monte Carlo noise between sims)")
print(f"  C1 at J_true = {C1_s:.6f}\n")

print("── Step 3 — Newton iterations ──────────────────────────────")
J_est, history, C0_hist, C1_hist = newton_estimate(
    J_INIT, C_target, KAPPA, ETA, DT, N, N_BURN, LAG, N_ITER)

err_abs = abs(J_est - J_TRUE)
err_pct = 100 * err_abs / J_TRUE
print(f"\n  J_true = {J_TRUE:.5f}")
print(f"  J_init = {J_INIT:.5f}   ({100*abs(J_INIT-J_TRUE)/J_TRUE:.0f}% initial error)")
print(f"  J_est  = {J_est:.5f}")
print(f"  Error  = {err_abs:.2e}   ({err_pct:.2f}%)")

# Plots (Help of ChatGPT was taken for the complete plotting section below)
print("\n── Step 4 — Building figure ────────────────────────────────")

# NSPT run at J_true for trajectory plots
z0_full, z1_full = run_nspt(J_TRUE, KAPPA, ETA, DT, N, seed=200)

# Full autocorrelation curves
MAX_LAG = min(800, N // 20)
lags    = np.arange(1, MAX_LAG + 1)
C_data_curve  = np.array([C_from_trajectory(z_data,  l, N_BURN) for l in lags])
C_model_curve = np.array([C_from_trajectory(z0_full, l, N_BURN) for l in lags])

# J sweep for inversion landscape
N_SH   = N // 5;  N_BU_SH = N_SH // 3
J_vals = np.linspace(0.4*J_TRUE, 1.7*J_TRUE, 24)
C_sw   = []
for Jv in J_vals:
    z0v, _ = run_nspt(Jv, KAPPA, ETA, DT, N_SH, seed=7)
    C_sw.append(C_from_trajectory(z0v, LAG, N_BU_SH))
C_sw = np.array(C_sw)

# ── Figure ──────────────────────────────────────────────────────────────────
BLUE = '#2C6FBF'; ORANGE = '#E06A1A'
GREEN = '#2E9C4F'; RED = '#C63030'; GRAY = '#888888'

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5,
                     'axes.labelsize': 10, 'lines.linewidth': 1.5})

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.33)
ax  = [fig.add_subplot(gs[r, c]) for r, c in
       [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]]

t = np.arange(N) * DT
W = min(int(80/DT), N)

# 1. z0 trajectory
ax[0].plot(t[:W], z0_full[:W], color=BLUE, lw=0.5, alpha=0.8)
ax[0].axvline(N_BURN*DT, color=RED, ls='--', lw=1.2, label='Burn-in')
ax[0].set_xlabel('Time'); ax[0].set_ylabel('z⁽⁰⁾(t)')
ax[0].set_title('Zeroth-order Bloch trajectory (stochastic)')
ax[0].legend(fontsize=8)

# 2. z1 trajectory
ax[1].plot(t[:W], z1_full[:W], color=ORANGE, lw=0.7, alpha=0.85)
ax[1].axvline(N_BURN*DT, color=RED, ls='--', lw=1.2, label='Burn-in')
ax[1].axhline(0, color=GRAY, lw=0.5, ls=':')
ax[1].set_xlabel('Time'); ax[1].set_ylabel('z⁽¹⁾(t) = ∂z/∂J')
ax[1].set_title('First-order sensitivity trajectory (same dW)')
ax[1].legend(fontsize=8)

# 3. Autocorrelation curves
tau_phys = lags * DT
ax[2].plot(tau_phys, C_data_curve,  color=BLUE,   lw=1.2,
           label='Data $C_z(τ)$ (from z_traj)')
ax[2].plot(tau_phys, C_model_curve, color=ORANGE, lw=1.2, ls='--',
           label=f'NSPT model (J={J_TRUE})')
ax[2].axvline(TAU0, color=GREEN, ls=':', lw=2.0,
              label=f'τ₀={TAU0:.3f}  (zero crossing)')
ax[2].axhline(0, color=GRAY, lw=0.5, ls=':')
ax[2].axhline(C_target, color=GRAY, lw=0.8, ls=':')
ax[2].set_xlabel('τ'); ax[2].set_ylabel('$C_z(τ)$')
ax[2].set_title('Autocorrelation: data vs NSPT model')
ax[2].legend(fontsize=8)

# 4. C0 and C1 per Newton iteration
iters = np.arange(1, len(C0_hist)+1)
ax3b  = ax[3].twinx()
ax[3].bar(iters-0.18, C0_hist, 0.32, color=BLUE,   alpha=0.75, label='C₀ (model)')
ax3b.bar(iters+0.18,  C1_hist, 0.32, color=ORANGE, alpha=0.75, label='C₁ = ∂C/∂J')
ax[3].axhline(C_target, color='k', ls='--', lw=1.2, label=f'C_data={C_target:.4f}')
ax[3].set_xlabel('Newton iteration')
ax[3].set_ylabel('C₀', color=BLUE)
ax3b.set_ylabel('C₁', color=ORANGE)
ax[3].set_title('NSPT observables per iteration')
ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))
h0, l0 = ax[3].get_legend_handles_labels()
h1, l1 = ax3b.get_legend_handles_labels()
ax[3].legend(h0+h1, l0+l1, fontsize=8, loc='upper right')

# 5. Newton convergence
iters_f = np.arange(len(history))
ax[4].plot(iters_f, history, 'o-', color=BLUE, ms=7, lw=1.8)
ax[4].axhline(J_TRUE, color=RED,  ls='--', lw=1.5, label=f'J_true={J_TRUE}')
ax[4].axhline(J_INIT, color=GRAY, ls=':',  lw=1.0, label=f'J_init={J_INIT}')
ax[4].fill_between(iters_f, J_TRUE*0.99, J_TRUE*1.01,
                   alpha=0.12, color=RED, label='±1% band')
ax[4].set_xlabel('Newton iteration'); ax[4].set_ylabel('J estimate')
ax[4].set_title(f'Newton convergence  (final error = {err_pct:.2f}%)')
ax[4].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[4].legend(fontsize=8)

# 6. C_z(τ₀) vs J sweep
ax[5].plot(J_vals, C_sw, 'o-', color=BLUE, ms=5, lw=1.5,
           label='$C_z(τ_0)$ from NSPT sweep')
ax[5].axhline(C_target, color='k', ls='--', lw=1.2,
              label=f'$C_{{data}}$={C_target:.4f}')
ax[5].axhline(0, color=GRAY, lw=0.5, ls=':')
ax[5].axvline(J_TRUE, color=RED,   ls='--', lw=1.5, label=f'J_true={J_TRUE}')
ax[5].axvline(J_est,  color=GREEN, ls=':',  lw=1.5, label=f'J_est={J_est:.4f}')
# tangent from last Newton step
Jp = history[-2]; C0l = C0_hist[-1]; C1l = C1_hist[-1]
Jt = np.array([Jp-0.5, Jp+0.5])
ax[5].plot(Jt, C0l + C1l*(Jt-Jp), color=ORANGE, lw=1.3, ls='--',
           label='NSPT tangent')
ax[5].set_xlabel('J'); ax[5].set_ylabel('$C_z(τ_0)$')
ax[5].set_title('Inversion landscape  C_z(τ₀) vs J')
ax[5].legend(fontsize=8)

fig.suptitle(
    f'NSPT Autocorrelation J-Estimator  |  '
    f'J_true={J_TRUE}, J_init={J_INIT}, κ={KAPPA}, η={ETA},  '
    f'τ₀=π/(4|Ω|)={TAU0:.3f}',
    fontsize=12, fontweight='bold')

plt.savefig('Quantum Parameter Estimation (Newton)/nspt_J_estimation.png',
            dpi=150, bbox_inches='tight')
print("  Figure saved → nspt_J_estimation.png")