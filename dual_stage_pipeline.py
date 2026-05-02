import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from collections import deque

# Parameters
J_TRUE  = 2.5
KAPPA   = 0.3
ETA     = 0.8
J_INIT  = 1.0        # 60% initial error

DT      = 0.005
T_TOTAL = 600.0
N       = int(T_TOTAL / DT)   # 120000 steps
N_BURN  = int(0.3 * N)        # 36000 steps

N_NEWTON = 1     # one Newton step per outer pass
N_OUTER  = 25    # more outer passes to cover the gap
TOL        = 1e-3 # early stopping 


TAU0 = np.pi / (4.0 * np.sqrt(J_TRUE**2 + 1.0)) # optimal lag
LAG  = max(1, int(TAU0 / DT))
TAU0 = LAG * DT 

print("=" * 65)
print("  NSPT Qubit J Estimator — Online Filter + Offline Newton")
print("=" * 65)
print(f"  J_true={J_TRUE},  k={KAPPA},  eta={ETA},  J_init={J_INIT}")
print(f"  omega = 2sqrt(J^2+1) = {2*np.sqrt(J_TRUE**2+1):.4f}")
print(f"  tau_0  = {TAU0:.4f}  ->  LAG={LAG} steps = {LAG*4} bytes")
print(f"  T={T_TOTAL}, dt={DT}, N={N:,}, N_burn={N_BURN:,}")
print(f"  Outer passes={N_OUTER}, Newton steps per pass={N_NEWTON}, tol={TOL}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def soft_norm(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    if r > 1.0:
        x /= r; y /= r; z /= r
    return x, y, z


# ─────────────────────────────────────────────────────────────────────────────
# Experiment — returns raw homodyne stream only
# z_true is retained separately for diagnostics; NOT passed to the estimator
# ─────────────────────────────────────────────────────────────────────────────
def generate_experiment(J, kappa, eta, dt, N, seed=42):
    rng  = np.random.default_rng(seed)
    c    = 2.0 * np.sqrt(eta * kappa)
    sqdt = np.sqrt(dt)
    x, y, z = 0.0, 0.0, 1.0
    dy_record = np.empty(N)
    z_true    = np.empty(N)
    for i in range(N):
        dW = rng.standard_normal() * sqdt
        dx = (-2.0*y - 2.0*kappa*x)            * dt - c*x*z * dW
        dy = ( 2.0*x - 2.0*J*z - 2.0*kappa*y)  * dt - c*y*z * dW
        dz = ( 2.0*J*y)                         * dt + c*(1.0-z**2) * dW
        x+=dx; y+=dy; z+=dz; x,y,z=soft_norm(x,y,z)
        dy_record[i] = c*z*dt + dW
        z_true[i]    = z
    return dy_record, z_true

# LAYER 1 — Online Filter
def online_filter(dy_record, J_star, kappa, eta, dt, n_burn, lag,
                    store_zhat=True):
    c    = 2.0 * np.sqrt(eta * kappa)
    xh, yh, zh = 0.0, 0.0, 1.0

    #Hardware: circular shift register, the only buffer needed
    shift_reg   = deque([zh]*lag, maxlen=lag)
    running_sum = 0.0
    count       = 0

    z_hat = np.empty(len(dy_record)) if store_zhat else None

    for i, dy in enumerate(dy_record):
        # Innovation: strips shot noise, leaving dW_est conditioned on z
        dW_est = dy - c * zh * dt

        # Online filter step (optimal quantum state estimator)
        dxh = (-2.0*yh - 2.0*kappa*xh)                  * dt - c*xh*zh * dW_est
        dyh = ( 2.0*xh - 2.0*J_star*zh - 2.0*kappa*yh)  * dt - c*yh*zh * dW_est
        dzh = ( 2.0*J_star*yh)                           * dt + c*(1.0-zh**2) * dW_est
        xh+=dxh; yh+=dyh; zh+=dzh; xh,yh,zh=soft_norm(xh,yh,zh)

        # Shift register: push z_new, pop z(t−tau_0)
        z_delayed = shift_reg[0]
        shift_reg.append(zh)

        # Accumulate after burn-in (hardware: enable line held low during burn-in)
        if i >= n_burn:
            running_sum += zh * z_delayed
            count       += 1

        if store_zhat:
            z_hat[i] = zh

    C_target = running_sum / count if count > 0 else 0.0
    return C_target, z_hat

# LAYER 2 — NSPT Newton
def run_nspt(J_star, kappa, eta, dt, N, seed=99):
    rng  = np.random.default_rng(seed)
    c    = 2.0 * np.sqrt(eta * kappa)
    sqdt = np.sqrt(dt)
    x0,y0,z0 = 0.,0.,1.;  x1,y1,z1 = 0.,0.,0.
    z0_arr = np.empty(N);  z1_arr = np.empty(N)
    for i in range(N):
        dW = rng.standard_normal()*sqdt
        dx0=(-2*y0-2*kappa*x0)*dt-c*x0*z0*dW
        dy0=(2*x0-2*J_star*z0-2*kappa*y0)*dt-c*y0*z0*dW
        dz0=(2*J_star*y0)*dt+c*(1-z0**2)*dW
        dx1=(-2*y1-2*kappa*x1)*dt-c*(x1*z0+x0*z1)*dW
        dy1=(2*x1-2*J_star*z1-2*z0-2*kappa*y1)*dt-c*(y1*z0+y0*z1)*dW
        dz1=(2*J_star*y1+2*y0)*dt-2*c*z0*z1*dW
        x0+=dx0;y0+=dy0;z0+=dz0; x0,y0,z0=soft_norm(x0,y0,z0)
        x1+=dx1;y1+=dy1;z1+=dz1
        z0_arr[i]=z0; z1_arr[i]=z1
    return z0_arr, z1_arr


def nspt_observables(z0, z1, lag, n_burn):
    a=z0[n_burn:]; b=z1[n_burn:]; n=len(a)-lag
    C0 = np.mean(a[:n]*a[lag:n+lag])
    C1 = np.mean(a[:n]*b[lag:n+lag]) + np.mean(b[:n]*a[lag:n+lag])
    return C0, C1


def newton_inner(J_star, C_target, kappa, eta, dt, N,
                 n_burn, lag, n_iter, seed=99, max_step=0.4):
    for it in range(n_iter):
        z0, z1 = run_nspt(J_star, kappa, eta, dt, N, seed=seed)
        C0, C1 = nspt_observables(z0, z1, lag, n_burn)
        if abs(C1) < 1e-12: break
        dJ = (C_target - C0) / C1
        dJ = np.clip(dJ, -max_step, max_step)  # trust region
        J_star = J_star + dJ
    return J_star, C0, C1


# Outer loop with convergence tracking
def run_pipeline(dy_record, J_init, kappa, eta, dt, N,
                 n_burn, lag, n_outer, n_newton, tol):
    J_star   = J_init
    J_history= [J_init]   # J* after each outer pass
    C_history= []          # C_target from each filter pass

    print(f"  {'Pass':>4}  {'J*(in)':>9}  {'C_filter':>11}  "
          f"{'J*(out)':>9}  {'dJ':>9}  {'err%':>7}")
    print("  " + "─"*56)

    z_hat_final = None
    for outer in range(n_outer):
        store = (outer == n_outer-1)

        # Recomputing lag based on current J*,  adaptive zero-crossing
        tau0_current = np.pi / (4.0 * np.sqrt(J_star**2 + 1.0))
        lag_current  = max(1, int(tau0_current / dt))

        C_target, z_hat = online_filter(
            dy_record, J_star, kappa, eta, dt, n_burn, lag_current,
            store_zhat=True)
        z_hat_final = z_hat
        C_history.append(C_target)

        J_new, C0_final, C1_final = newton_inner(
            J_star, C_target, kappa, eta, dt, N,
            n_burn, lag_current, n_newton, seed=99)

        dJ      = J_new - J_star
        err_pct = 100 * abs(J_new - J_TRUE) / J_TRUE
        print(f"  {outer+1:>4}  {J_star:>9.5f}  {C_target:>11.6f}  "
            f"{J_new:>9.5f}  {dJ:>9.5f}  {err_pct:>6.2f}%")

        J_history.append(J_new)
        if abs(dJ) < tol:
            print(f"  Converged at pass {outer+1} (|dJ|={abs(dJ):.2e} < tol={tol})")
            break
        J_star = J_new

    return J_star, J_history, C_history, z_hat_final


# Run
print("── Step 1: Generate raw homodyne stream ─────────────────────────")
dy_record, z_true_diag = generate_experiment(J_TRUE, KAPPA, ETA, DT, N)

print("── Step 2: Run full pipeline ────────────────────────────────────")
J_est, J_history, C_history, z_hat = run_pipeline(
    dy_record, J_INIT, KAPPA, ETA, DT, N,
    N_BURN, LAG, N_OUTER, N_NEWTON, TOL)

err_abs = abs(J_est - J_TRUE)
err_pct = 100 * err_abs / J_TRUE
C_true_ref = np.mean(z_true_diag[N_BURN:N-LAG] * z_true_diag[N_BURN+LAG:])

print(f"\n{'═'*50}")
print(f"  J_true  = {J_TRUE:.5f}")
print(f"  J_init  = {J_INIT:.5f}  ({100*abs(J_INIT-J_TRUE)/J_TRUE:.0f}% initial error)")
print(f"  J_est   = {J_est:.5f}")
print(f"  Error   = {err_abs:.2e}  ({err_pct:.2f}%)")
print(f"  C_filter (final pass) = {C_history[-1]:.6f}")
print(f"  C_true_ref (diagn.)   = {C_true_ref:.6f}")
print(f"{'═'*50}\n")

# Plots (Help of chatgpt was taken for the plotting part below)
print("── Step 3: Building figure ──────────────────────────────────────")

z0_ref, z1_ref = run_nspt(J_TRUE, KAPPA, ETA, DT, N, seed=200)
z0_init, _     = run_nspt(J_INIT, KAPPA, ETA, DT, N//2, seed=201)

MAX_LAG = min(400, N//8)
lags    = np.arange(1, MAX_LAG+1)
def ac(z, n_burn):
    zs = z[n_burn:]; return np.array([np.mean(zs[:len(zs)-l]*zs[l:]) for l in lags])

C_hat_curve  = ac(z_hat,        N_BURN)
C_true_curve = ac(z_true_diag,  N_BURN)
C_model_true = ac(z0_ref,       N_BURN)

# J sweep
N_SH=N//4; NB_SH=N_SH//3
J_vals = np.linspace(0.35*J_TRUE, 1.8*J_TRUE, 22)
C_sw = []
for Jv in J_vals:
    z0v,_=run_nspt(Jv,KAPPA,ETA,DT,N_SH,seed=7)
    n=len(z0v)-NB_SH-LAG
    C_sw.append(np.mean(z0v[NB_SH:NB_SH+n]*z0v[NB_SH+LAG:NB_SH+n+LAG]))
C_sw = np.array(C_sw)

BLUE='#2C6FBF'; ORANGE='#E06A1A'; GREEN='#2E9C4F'
RED='#C63030';  GRAY='#888888';   PURPLE='#7B2FBE'

plt.rcParams.update({'font.size':10,'axes.titlesize':10.5,
                     'axes.labelsize':10,'lines.linewidth':1.5})

fig = plt.figure(figsize=(18,11))
gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.43,wspace=0.34)
ax  = [fig.add_subplot(gs[r,c]) for r,c in
       [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]]

t=np.arange(N)*DT; W=min(int(50/DT),N)

# 1. Filter tracking
ax[0].plot(t[:W], z_true_diag[:W], color=RED,  lw=0.7, alpha=0.7,
           label='z_true (diagnostic)')
ax[0].plot(t[:W], z_hat[:W],       color=BLUE, lw=0.7, alpha=0.8,
           label='ẑ  (Belavkin, final J*)')
ax[0].axvline(N_BURN*DT, color=GRAY, ls='--', lw=1.0, label='Burn-in end')
ax[0].set_xlabel('Time'); ax[0].set_ylabel('z')
ax[0].set_title('Layer 1 — Belavkin filter: ẑ vs z_true  (final pass)')
ax[0].legend(fontsize=8)

# 2. Outer loop J convergence
passes = np.arange(len(J_history))
ax[1].plot(passes, J_history, 'o-', color=BLUE, ms=7, lw=1.8)
ax[1].axhline(J_TRUE, color=RED,  ls='--', lw=1.5, label=f'J_true={J_TRUE}')
ax[1].axhline(J_INIT, color=GRAY, ls=':',  lw=1.0, label=f'J_init={J_INIT}')
ax[1].fill_between(passes, J_TRUE*0.99, J_TRUE*1.01,
                   alpha=0.12, color=RED, label='±1% band')
ax[1].set_xlabel('Outer pass'); ax[1].set_ylabel('J*')
ax[1].set_title(f'Outer loop convergence  (final error={err_pct:.2f}%)')
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].legend(fontsize=8)

# 3. C_filter per outer pass → shows approach to zero
passes_C = np.arange(1, len(C_history)+1)
ax[2].plot(passes_C, C_history, 'o-', color=PURPLE, ms=7, lw=1.8,
           label='C_filter(J*, τ₀) per pass')
ax[2].axhline(C_true_ref, color=RED,  ls='--', lw=1.5,
              label=f'C_true_ref={C_true_ref:.4f}')
ax[2].axhline(0, color=GRAY, lw=0.5, ls=':')
ax[2].set_xlabel('Outer pass'); ax[2].set_ylabel('C_filter(J*, τ₀)')
ax[2].set_title('C_filter converges to 0 as J*→J_true')
ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[2].legend(fontsize=8)

# 4. Autocorrelation curves
tau_phys = lags * DT
ax[3].plot(tau_phys, C_hat_curve,  color=BLUE,   lw=1.2, label='ẑ (final filter pass)')
ax[3].plot(tau_phys, C_true_curve, color=RED,    lw=1.2, ls=':', label='z_true (diagnostic)')
ax[3].plot(tau_phys, C_model_true, color=ORANGE, lw=1.2, ls='--', label='NSPT model at J_true')
ax[3].axvline(TAU0, color=GREEN, ls=':', lw=2.0, label=f'τ₀={TAU0:.3f}')
ax[3].axhline(0, color=GRAY, lw=0.5, ls=':')
ax[3].set_xlabel('τ'); ax[3].set_ylabel('$C_z(τ)$')
ax[3].set_title('Autocorrelation: filter vs truth vs NSPT model')
ax[3].legend(fontsize=8)

# 5. Inversion landscape with C_filter curve overlay
ax[4].plot(J_vals, C_sw, 'o-', color=BLUE, ms=4, lw=1.5,
           label='C_model(J, τ₀)  [NSPT sweep]')
# Overlay C_filter values at the J* used in each pass
J_in_passes = J_history[:-1]   # J* used as filter input at each pass
ax[4].plot(J_in_passes, C_history, 's--', color=PURPLE, ms=7, lw=1.3,
           label='C_filter(J*, τ₀)  [per pass]')
ax[4].axhline(0, color=GRAY, lw=0.5, ls=':')
ax[4].axvline(J_TRUE, color=RED,   ls='--', lw=1.5, label=f'J_true={J_TRUE}')
ax[4].axvline(J_est,  color=GREEN, ls=':',  lw=1.5, label=f'J_est={J_est:.4f}')
ax[4].set_xlabel('J'); ax[4].set_ylabel('$C_z(τ_0)$')
ax[4].set_title('Inversion landscape + filter C_target trajectory')
ax[4].legend(fontsize=8)

# 6. Filter tracking error (rolling RMS)
win = max(1, int(5/DT))
err = z_hat - z_true_diag
rms_vals, t_rms = [], []
for i in range(0, N-win, win):
    rms_vals.append(np.sqrt(np.mean(err[i:i+win]**2)))
    t_rms.append((i+win//2)*DT)
ax[5].plot(t_rms, rms_vals, color=PURPLE, lw=1.2)
ax[5].axvline(N_BURN*DT, color=GRAY, ls='--', lw=1.0, label='Burn-in end')
ax[5].set_xlabel('Time'); ax[5].set_ylabel('RMS(ẑ − z_true)')
ax[5].set_title('Filter tracking error  (final pass, J*≈J_true)')
ax[5].legend(fontsize=8)

fig.suptitle(
    f'NSPT Qubit J Estimator: Online Belavkin Filter + Offline Newton  |  '
    f'J_true={J_TRUE},  J_init={J_INIT},  κ={KAPPA},  η={ETA},  '
    f'τ₀={TAU0:.3f},  final error={err_pct:.2f}%',
    fontsize=10.5, fontweight='bold')

plt.savefig('Quantum Parameter Estimation (Newton)/nspt_merged_pipeline.png',
            dpi=150, bbox_inches='tight')
print("  Figure saved.")