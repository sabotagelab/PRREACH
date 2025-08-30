import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from scipy.integrate import solve_ivp

# === Define Quadrotor Parameters ===
m = 1.5    # Mass (kg)
g = 9.81   # Gravity (m/s^2)
Jx = 0.02  # Moment of inertia around x (kg·m^2)
Jy = 0.02  # Moment of inertia around y (kg·m^2)
Jz = 0.04  # Moment of inertia around z (kg·m^2)

# State Vector
# x= [x,y,z, roll,pitch,yaw, vx,vy,vz, roll_dot,pitch_dot,yaw_dot]
state_size = 12

# Input vector
# u = [thrust, roll, pitch, yaw]
control_size = 4

# === Define State-Space Matrices ===
A_null = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
])

B_null = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1/Jx, 0, 0],
    [0, 0, 1/Jy, 0],
    [0, 0, 0, 1/Jz],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1/m, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# considering only forces in x and y directions
B_wind = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1/Jx, 0],
    [0, 0, 0, 1/Jy],
    [0, 0, 0, 0],
    [1/m, 0, 0, 0],
    [0, 1/m, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])


dt = 0.02  # Discretization time step (50 ms)
T_final = 10  # Total simulation time (seconds)

def get_dynamics(hazard=None, dt=0.02):
    A = A_null
    B = B_null
    if hazard == 'rotor':
        B = 0.4 * B_null
    if hazard == 'drift':
        A[0, 0] += 0.6
        A[1, 1] += 0.55
        # A[0, 3] += 0.15
        # A[1, 4] += 0.25

    # === Discretize the System ===
    Ad = scipy.linalg.expm(A * dt)  # Discretized A using matrix exponential
    Bd = np.zeros_like(B)
    n = 1000 # number of integration steps
    for i in range(1, n + 1):
        tau = i * dt/n
        Bdd = scipy.linalg.expm(A * tau) @ B * dt/n
        Bd += Bdd

    # === Define Cost Matrices for LQR ===
    Q = np.diag([5.0, 2, .1, .001, .001, .001, .01, .01, .01, .001, .001, .001])

    if hazard == 'rotor':
        Q = np.diag([2.5, 2.5, .1, .001, .001, .001, .01, .01, .01, .001, .001, .001])
    if hazard == 'drift':
        Q = np.diag([1, 2, .1, .001, .001, .001, .01, .01, .01, .001, .001, .001])
    if hazard == 'wind':
        Q = np.diag([1., 2, .1, 1, .5, 1, .01, .01, .01, .01, .01, .01])
    
    R = np.diag([1, .1, .1, .1])  # Control cost

    # Solve Discrete-Time Algebraic Riccati Equation (DARE)
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

    # Compute Discrete-Time LQR Gain: K = (B^T P B + R)^(-1) B^T P A
    Kd = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    return Ad, Bd, np.asarray(Kd), Q, R
    

def plot_disc_sim(Ad, Bd, Kd, dt=0.02, T_final=10, wind=False):
    # === Define Simulation Parameters ===
    num_steps = int(T_final / dt)  # Number of discrete time steps
    time = np.arange(0, T_final, dt)

    # === Initialize State Variables ===
    if Ad.shape[0] == 12:
        x = np.zeros((12, num_steps))  # State vector over time
        # x[:, 0] = np.array([5, 5, 1, 0.1, -0.05, 0.05, 0, 0, 0, 0, 0, 0])  # Initial state
        x[:, 0] = np.array([5, 5, 1, 0.1, -1, 0.05, 0, 0, 0, 0, 0, 0])  # Initial state
    elif Ad.shape[0] == 13:
        x = np.zeros((13, num_steps))  # State vector over time
        x[:, 0] = np.array([0, 0, 1, 0.1, -0.05, 0.05, 0, 0, 0, 0, 0, 0, 10])  # Initial state

    dyn_b = 0.15 * np.array([[0, 0], [0, 0], [0, 0],
                            [0, 0], [0, 0],[0, 0],
                            [1, 0], [0, 1], [0, 0],
                            [0, 0],[0, 0], [0, 0]])

    # === Simulate the System in Discrete-Time ===
    for k in range(num_steps - 1):
        u = -Kd @ (x[:, k] - np.array([0, 0, 5, 0.0, 0, 0.0, 0, 0, 0, 0, 0, 0]))  # Apply LQR control
        x[:, k+1] = Ad @ x[:, k] + Bd @ u  # Update state
        if wind:
            x[:, k+1] += dyn_b @ np.array([2, -1])

    # === Plot Results ===
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    labels = ["X Position (m)", "Y Position (m)", "Altitude (m)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    states = [x[i, :] for i in range(6)]

    for i, ax in enumerate(axs.flat):
        ax.plot(time, states[i])
        ax.set_ylabel(labels[i])
        ax.set_xlabel("Time (s)")
        ax.grid()

    plt.tight_layout()
    plt.show()

    x_pos = x[0, :]
    y_pos = x[1, :] 
    plt.plot(x_pos, y_pos)
    plt.show()


if __name__== "__main__":
    Ad, Bd, Kd, Q, R  = get_dynamics()
    plot_disc_sim(Ad, Bd, Kd)
    # Ad, Bd, Kd  = get_dynamics(hazard='rotor')
    # plot_disc_sim(Ad, Bd, Kd)
    # Ad, Bd, Kd  = get_dynamics(hazard='drift_y')
    # plot_disc_sim(Ad, Bd, Kd)
    Ad, Bd, Kd, Q, R  = get_dynamics(hazard='wind')
    plot_disc_sim(Ad, Bd, Kd, wind=True)
