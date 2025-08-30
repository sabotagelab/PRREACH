import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from scipy.integrate import solve_ivp

# === Point Mass Model ===

# State Vector
# x= [x,y,vx,vy]
state_size = 4

# Input vector
# u = [ax, ay]
control_size = 2

# === Define State-Space Matrices ===
A_null = np.array([[0, 0, 1, 0],
            [0, 0, 0, 1], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]])
# Dynamics control matrix B
B_null = np.array([[0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.]])

A_wind = np.array([[0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 1], 
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1]])
B_wind = np.array([[0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [0., 0.]])

A_wind2 = np.array([[0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1]])
B_wind2 = np.array([[0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [0., 0.],
            [0., 0.]])


dt = 0.02  # Discretization time step (50 ms)
T_final = 10  # Total simulation time (seconds)

def get_dynamics(hazard=None, dt=0.02, Q=None):
    A = A_null
    B = B_null
    if hazard == 'rotor':
        B = 0.75 * B_null
    if hazard == 'drift':
        # A[1, 1] += .1
        A[0, 0] += 1
        A[1, 1] -= .75

    # === Discretize the System ===
    Ad = scipy.linalg.expm(A * dt)  # Discretized A using matrix exponential
    Bd = np.zeros_like(B)
    n = 1000 # number of integration steps
    for i in range(1, n + 1):
        tau = i * dt/n
        Bdd = scipy.linalg.expm(A * tau) @ B * dt/n
        Bd += Bdd

    # === Define Cost Matrices for LQR ===
    if Q is None:
        Q = np.diag([12, 5, .1, .1])
        if hazard == 'drift':
            Q = np.diag([12, 5, .1, .1])
        if hazard == 'wind':
            # Q = np.diag([5, 5, 1, 1])
            Q = np.diag([12, 5, .1, .1])
        if hazard == 'rotor':
            Q = np.diag([5, 5, .1, .1])
        if hazard == 'wind2':
            Q = np.diag([6.2, 3.1, .1, .1, .1, .1])
    
    R = np.diag([1, 1])  # Control cost

    # Solve Discrete-Time Algebraic Riccati Equation (DARE)
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

    # Compute Discrete-Time LQR Gain: K = (B^T P B + R)^(-1) B^T P A
    Kd = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    return Ad, Bd, np.asarray(Kd), Q, R

    

def plot_disc_sim(Ad, Bd, Kd, dt=0.02, T_final=10):
    # === Define Simulation Parameters ===
    num_steps = int(T_final / dt)  # Number of discrete time steps
    time = np.arange(0, T_final, dt)

    # === Initialize State Variables ===
    if Ad.shape[0] == 4:
        x = np.zeros((4, num_steps))  # State vector over time
        x[:, 0] = np.array([7, 8, 1, 1])  # Initial state
    elif Ad.shape[0] == 5:
        x = np.zeros((5, num_steps))  # State vector over time
        x[:, 0] = np.array([7, 8, 1, 1, 10])  # Initial state

    # === Simulate the System in Discrete-Time ===
    for k in range(num_steps - 1):
        u = -Kd @ x[:, k]  # Apply LQR control
        x[:, k+1] = Ad @ x[:, k] + Bd @ u  # Update state

    # === Plot Results ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    labels = ["X Position (m)", "Y Position (m)"]
    states = [x[i, :] for i in range(2)]

    for i, ax in enumerate(axs.flat):
        ax.plot(time, states[i])
        ax.set_ylabel(labels[i])
        ax.set_xlabel("Time (s)")
        ax.grid()

    plt.tight_layout()
    plt.show()


if __name__== "__main__":
    Ad, Bd, Kd  = get_dynamics()
    plot_disc_sim(Ad, Bd, Kd)
    Ad, Bd, Kd  = get_dynamics(hazard='rotor')
    plot_disc_sim(Ad, Bd, Kd)
    Ad, Bd, Kd  = get_dynamics(hazard='drift')
    plot_disc_sim(Ad, Bd, Kd)
    Ad, Bd, Kd  = get_dynamics(hazard='wind')
    plot_disc_sim(Ad, Bd, Kd)
