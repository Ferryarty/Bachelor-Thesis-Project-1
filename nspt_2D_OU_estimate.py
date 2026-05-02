import numpy as np
import matplotlib.pyplot as plt

class TruncatedSeries:
    def __init__(self, coeffs, max_order=None):
        if np.isscalar(coeffs):
            self.coeffs = np.array([coeffs], dtype=float)
        else:
            self.coeffs = np.array(coeffs, dtype=float)
            
        self.max_order = len(self.coeffs) - 1 if max_order is None else max_order
        
        # Pad with zeros if necessary
        if len(self.coeffs) < self.max_order + 1:
            pad_width = (self.max_order + 1) - len(self.coeffs)
            self.coeffs = np.pad(self.coeffs, (0, pad_width), 'constant')

    def __repr__(self):
        terms = [f"{c:.4e}d^{i}" for i, c in enumerate(self.coeffs)]
        return " + ".join(terms)
    
    def __neg__(self):
        return TruncatedSeries(-self.coeffs, self.max_order)
    
    def __pos__(self): return self
    
    def __add__(self, other):
        other = self._ensure_type(other)
        return TruncatedSeries(self.coeffs + other.coeffs, self.max_order)
    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        other = self._ensure_type(other)
        return TruncatedSeries(self.coeffs - other.coeffs, self.max_order)
    def __rsub__(self, other):
        other = self._ensure_type(other)
        return TruncatedSeries(other.coeffs - self.coeffs, self.max_order)

    def __mul__(self, other):
        other = self._ensure_type(other)
        new_coeffs = np.convolve(self.coeffs, other.coeffs)[:self.max_order + 1]
        return TruncatedSeries(new_coeffs, self.max_order)
    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        other = self._ensure_type(other)
        q = np.zeros(self.max_order + 1)
        b = other.coeffs; a = self.coeffs
        for i in range(self.max_order + 1):
            term = a[i]
            for j in range(i): term -= q[j] * b[i-j]
            q[i] = term / b[0]
        return TruncatedSeries(q, self.max_order)
    def __rtruediv__(self, other): return other.__truediv__(self)
        
    def __pow__(self, power):
        if not isinstance(power, int): raise NotImplementedError("Only integer powers")
        result = TruncatedSeries([1.0], self.max_order)
        base = self
        for _ in range(power): result = result * base
        return result

    def _ensure_type(self, other):
        if isinstance(other, TruncatedSeries): return other
        return TruncatedSeries(other, self.max_order)
    
    @property
    def value(self): return self.coeffs[0]
    @property
    def grad(self): return self.coeffs[1] if self.max_order >= 1 else 0.0


# 2D OU Physics Engine


def generate_2d_trajectory(bx, by, D, dt, T):
    """Generates X and Y paths (uncoupled for simplicity, but handled together)."""
    N = int(T / dt)
    traj = np.zeros((N, 2)) # Column 0 = x, Column 1 = y
    
    const_diff = np.sqrt(2 * D)
    
    # Pre-generate noise
    dw = np.random.normal(0, np.sqrt(dt), (N, 2))
    
    # Simulate
    for i in range(N-1):
        x_curr, y_curr = traj[i]
        
        # Euler-Maruyama updates
        dx = -bx * x_curr * dt + const_diff * dw[i, 0]
        dy = -by * y_curr * dt + const_diff * dw[i, 1]
        
        traj[i+1] = [x_curr + dx, y_curr + dy]
        
    return traj

def get_analytical_mle_2d(traj, dt):
    # Helper for 1D MLE
    def solve_1d(arr):
        x_curr = arr[:-1]
        x_next = arr[1:]
        # MLE Formula: - sum(x_i * (x_i+1 - x_i)) / (dt * sum(x_i^2))
        num = np.sum(x_curr * (x_next - x_curr))
        den = np.sum(x_curr**2) * dt
        return -num / den

    bx_mle = solve_1d(traj[:, 0])
    by_mle = solve_1d(traj[:, 1])
    return bx_mle, by_mle

# NSPT Logic

def calculate_log_likelihood_2d(traj, bx_in, by_in, D, dt):
    L_total = TruncatedSeries([0.0, 0.0], max_order=1)
    
    # We iterate through the trajectory
    
    for i in range(len(traj) - 1):
        x_c, y_c = traj[i]
        x_n, y_n = traj[i+1]
        
        # res_x = x_next - x_prev + bx * x_prev * dt
        res_x = x_n - x_c + bx_in * x_c * dt
        
        # y-component residual
        res_y = y_n - y_c + by_in * y_c * dt
        
        L_total = L_total + (res_x ** 2) + (res_y ** 2)
        
    const_factor = -1.0 / (4 * D * dt)
    return L_total * const_factor

def get_2d_gradients(traj, bx_val, by_val, D, dt):
    bx_series = TruncatedSeries([bx_val, 1.0], max_order=1)
    by_const  = TruncatedSeries([by_val, 0.0], max_order=1)
    
    L_1 = calculate_log_likelihood_2d(traj, bx_series, by_const, D, dt)
    grad_bx = L_1.grad
    
    bx_const  = TruncatedSeries([bx_val, 0.0], max_order=1)
    by_series = TruncatedSeries([by_val, 1.0], max_order=1)
    
    L_2 = calculate_log_likelihood_2d(traj, bx_const, by_series, D, dt)
    grad_by = L_2.grad
    
    return np.array([grad_bx, grad_by])

# Verification Loop

if __name__ == "__main__":
    # Parameters
    D = 0.5
    dt = 0.002
    T = 10.0
    
    # True Physics Parameters
    true_bx = 2.0
    true_by = 4.0 
    
    print("1. Generating 2D Experimental Data...")
    traj = generate_2d_trajectory(true_bx, true_by, D, dt, T)
    
    print("2. Calculating Analytical MLE (The 'Gold Standard')...")
    anal_bx, anal_by = get_analytical_mle_2d(traj, dt)
    print(f"   > Analytical MLE Target: bx={anal_bx:.4f}, by={anal_by:.4f}")
    
    print("\n3. Running NSPT Optimization (Starting from [0.5, 0.5])...")
    
    # Initial Guesses
    curr_bx = 0.5
    curr_by = 0.5
    
    # History for plotting
    history_bx = [curr_bx]
    history_by = [curr_by]
    
    # Optimization Params
    learning_rate = 0.8 # Fixed small rate for stability verification
    steps = 40
    
    for k in range(steps):
        # Compute Gradients using NSPT Class
        grads = get_2d_gradients(traj, curr_bx, curr_by, D, dt)
        
        # Normalize Gradient by T (Robust update rule)
        norm_grads = grads / T
        
        # Update
        curr_bx += learning_rate * norm_grads[0]
        curr_by += learning_rate * norm_grads[1]
        
        history_bx.append(curr_bx)
        history_by.append(curr_by)
        
        if k % 5 == 0:
            print(f"   Step {k:02d}: Est=[{curr_bx:.4f}, {curr_by:.4f}] | Grad=[{grads[0]:.2f}, {grads[1]:.2f}]")

    print(f"   > Final NSPT Estimate:   bx={curr_bx:.4f}, by={curr_by:.4f}")

    # Plotting

    plt.figure(figsize=(8, 8))
    
    plt.plot(history_bx, history_by, 'o-', label='NSPT Optimization Path', color='blue', markersize=4)
    plt.plot(history_bx[0], history_by[0], 'kD', label='Start')
    
    plt.plot(anal_bx, anal_by, 'r*', markersize=20, label='Analytical MLE (Target)')
    
    plt.plot(true_bx, true_by, 'gx', markersize=10, label='True Physics Params')
    
    plt.title("2D NSPT Verification: Convergence to Analytical MLE")
    plt.xlabel("Stiffness X ($b_x$)")
    plt.ylabel("Stiffness Y ($b_y$)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()