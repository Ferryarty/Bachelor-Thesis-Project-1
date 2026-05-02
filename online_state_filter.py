import numpy as np
import matplotlib.pyplot as plt
import csv



# Simulation Parameters

T = 10.0               # Total simulation time
dt = 1e-4              # Time step length (Euler-Maruyama)
steps = int(T / dt)    # Total number of steps
time = np.linspace(0, T, steps)



# Physics Parameters
kappa = 1.0            # Measurement strength / Decoherence rate
eta = 1.0              # Measurement efficiency
J_true = 2.5           # The true parameter we are trying to estimate


# Filter Parameters
alpha = 0.5           # Initial learning rate for gradient descent
J_guess_initial = 2.4  # The drone's initial blind guess for J



# CSV Logging Parameters
log_interval = 100     # Log data every 100 steps (generates around 200 rows)
debug_log = []         # Array to hold our row data




x_true, y_true, z_true = np.zeros(steps), np.zeros(steps), np.zeros(steps)
z_true[0] = 1.0  

x_est, y_est, z_est = np.zeros(steps), np.zeros(steps), np.zeros(steps)
z_est[0] = 1.0  

x_grad, y_grad, z_grad = np.zeros(steps), np.zeros(steps), np.zeros(steps)

J_est = np.zeros(steps)
J_est[0] = J_guess_initial

noise_coeff = 2.0 * np.sqrt(eta * kappa)



# Estimation Loop

for i in range(1, steps):

    # BLOCK A: NATURE (The True System)
    dW_true = np.sqrt(dt) * np.random.randn()
    dy_meas = noise_coeff * z_true[i-1] * dt + dW_true

   

    dx_true = (-2*y_true[i-1] - 2*kappa*x_true[i-1])*dt - noise_coeff*x_true[i-1]*z_true[i-1]*dW_true
    dy_true = (2*x_true[i-1] - 2*J_true*z_true[i-1] - 2*kappa*y_true[i-1])*dt - noise_coeff*y_true[i-1]*z_true[i-1]*dW_true
    dz_true = (2*J_true*y_true[i-1])*dt + noise_coeff*(1 - z_true[i-1]**2)*dW_true

   

    x_true[i] = x_true[i-1] + dx_true
    y_true[i] = y_true[i-1] + dy_true
    z_true[i] = z_true[i-1] + dz_true



    norm = np.sqrt(x_true[i]**2 + y_true[i]**2 + z_true[i]**2)

    if norm > 1:
        x_true[i] /= norm
        y_true[i] /= norm
        z_true[i] /= norm


    # BLOCK B: THE FILTER (The Drone's FPGA)

    dW_est = dy_meas - noise_coeff * z_est[i-1] * dt

    current_alpha = alpha / (1.0 + time[i])
    J_est[i] = J_est[i-1] + current_alpha* noise_coeff*z_grad[i-1] * dW_est

   
    dx_est = (-2*y_est[i-1] - 2*kappa*x_est[i-1])*dt - noise_coeff*x_est[i-1]*z_est[i-1]*dW_est
    dy_est = (2*x_est[i-1] - 2*J_est[i]*z_est[i-1] - 2*kappa*y_est[i-1])*dt - noise_coeff*y_est[i-1]*z_est[i-1]*dW_est
    dz_est = (2*J_est[i]*y_est[i-1])*dt + noise_coeff*(1 - z_est[i-1]**2)*dW_est

   

    x_est[i] = x_est[i-1] + dx_est
    y_est[i] = y_est[i-1] + dy_est
    z_est[i] = z_est[i-1] + dz_est



    norm_est = np.sqrt(x_est[i]**2 + y_est[i]**2 + z_est[i]**2)

    if norm_est > 1.0:
        x_est[i] /= norm_est
        y_est[i] /= norm_est
        z_est[i] /= norm_est


    # This is (d_Innovation / d_J)
    d_innov_dJ = - noise_coeff * z_grad[i-1] * dt

    # Defining the diffusion coefficients (The terms that multiply dW in the state equations)
    Gx = - noise_coeff * x_est[i-1] * z_est[i-1]
    Gy = - noise_coeff * y_est[i-1] * z_est[i-1]
    Gz =   noise_coeff * (1 - z_est[i-1]**2)

    dx_grad = (-2*y_grad[i-1] - 2*kappa*x_grad[i-1])*dt - noise_coeff*(x_grad[i-1]*z_est[i-1] + x_est[i-1]*z_grad[i-1])*dW_est + Gx * d_innov_dJ
    dy_grad = (2*x_grad[i-1] - 2*z_est[i-1] - 2*J_est[i]*z_grad[i-1] - 2*kappa*y_grad[i-1])*dt - noise_coeff*(y_grad[i-1]*z_est[i-1] + y_est[i-1]*z_grad[i-1])*dW_est + Gy * d_innov_dJ
    dz_grad = (2*y_est[i-1] + 2*J_est[i]*y_grad[i-1])*dt - 2*noise_coeff*(z_est[i-1]*z_grad[i-1])*dW_est + Gz * d_innov_dJ 

   
    x_grad[i] = x_grad[i-1] + dx_grad
    y_grad[i] = y_grad[i-1] + dy_grad
    z_grad[i] = z_grad[i-1] + dz_grad



    # After updating x_grad[i], y_grad[i], z_grad[i]

    r_est = np.array([x_est[i], y_est[i], z_est[i]])
    s = np.array([x_grad[i], y_grad[i], z_grad[i]])
    s -= np.dot(s, r_est) * r_est  # Remove radial componen
    x_grad[i], y_grad[i], z_grad[i] = s



    # BLOCK C: CSV TELEMETRY LOGGING

    if i % log_interval == 0:
        debug_log.append([
            i,

            f"{time[i]:.5f}",

            f"{J_est[i]:.5f}", f"{current_alpha:.5f}",

            f"{z_true[i]:.5f}", f"{z_est[i]:.5f}",

            f"{y_true[i]:.5f}", f"{y_est[i]:.5f}",

            f"{dW_true:.5f}", f"{dW_est:.5f}", f"{dy_meas:.5f}",

            f"{x_grad[i]:.5f}", f"{y_grad[i]:.5f}", f"{z_grad[i]:.5f}"

        ])


# EXPORT TO CSV

csv_headers = [

    'Step', 'Time',

    'J_est', 'Alpha',

    'z_true', 'z_est',

    'y_true', 'y_est',

    'dW_true', 'dW_est', 'dy_meas',

    'x_grad', 'y_grad', 'z_grad'

]



with open('nspt_debug_log.csv', mode='w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(csv_headers)

    writer.writerows(debug_log)

print("Simulation complete. Telemetry saved to 'nspt_debug_log.csv'.")



# Plotting

plt.figure(figsize=(14, 10))

# Top-left: J estimation
plt.subplot(2, 2, 1)
plt.plot(time, J_est, label="Estimated J (NSPT)", color='blue')
plt.axhline(J_true, color='red', linestyle='--', label="True J")
plt.title("Real-Time Parameter Estimation using NSPT")
plt.ylabel("Parameter Value (J)")
plt.legend()
plt.grid(True)



plt.subplot(2, 2, 2)
plt.plot(time, z_true, label="True z", alpha=0.6, color='red')
plt.plot(time, z_est, label="Estimated z", alpha=0.6, color='blue')
plt.title("Bloch Vector Z-Component Tracking")
plt.ylabel("<sigma_z>")
plt.legend()
plt.grid(True)


plt.subplot(2, 2, 3)
plt.plot(time, x_true, label="True x", alpha=0.6, color='red')
plt.plot(time, x_est, label="Estimated x", alpha=0.6, color='blue')
plt.title("Bloch Vector X-Component Tracking")
plt.xlabel("Time")
plt.ylabel("<sigma_x>")
plt.legend()
plt.grid(True)


plt.subplot(2, 2, 4)
plt.plot(time, y_true, label="True y", alpha=0.6, color='red')
plt.plot(time, y_est, label="Estimated y", alpha=0.6, color='blue')
plt.title("Bloch Vector Y-Component Tracking")
plt.xlabel("Time")
plt.ylabel("<sigma_y>")
plt.legend()
plt.grid(True)



plt.tight_layout()
plt.show()