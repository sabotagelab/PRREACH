import time

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, minimize

from dynamics2 import B_wind, get_dynamics, state_size
from opt_helper_funcs import (SAMPLES, build_heatmap, c,
                              get_wind_vector_center, lqr_control,
                              p_building_theta_params, p_theta_params,
                              pop_heatmap, rho, run_single_simulation,
                              simplex_map, v1, v2, v3, v4)

hazard_cause = 'wind'
outcome = 'building'

dt = 0.01
steps = 25
hazard_prob = 0.2
trigger_time_start = 1
trigger_time_end = 10

num_trials = 100

n_rows = state_size
n_cols = state_size

if outcome == 'population':
    heatmap = pop_heatmap
    model_params = p_theta_params
else:
    heatmap = build_heatmap
    model_params = p_building_theta_params

fig_params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'text.usetex': False,
    'figure.figsize': (6.4, 2.8)
}
mpl.rcParams.update(fig_params)

# Define your objective function
def objective(A, B, K):
    def my_obj(flat_D):
        # Reshape flat_x into matrix
        D = flat_D.reshape((n_rows, n_cols))
        obj = np.linalg.norm((A - np.matmul(B, K)) - D, ord='fro')**2
        return obj
    return my_obj

def gen_constraint(k, simplex_map=simplex_map, wind_params=None, model_params=model_params, dyn_b=None):
    # rho_k <= rk
    def constraintk(flat_D):
        D = flat_D.reshape((n_rows, n_cols))
        rhok = rho(k, D, simplex_map=simplex_map, wind_params=wind_params, model_params=model_params, dyn_b=dyn_b)
        return rhok
    return constraintk

print("==========OFFLINE COMPARISON WITH AND WITHOUT PRREACH==========")
A, B, K, Q, R = get_dynamics(dt=dt)  # get dynamics of null cause 
D = A - np.matmul(B, K)
D_null = np.copy(D)
A_null = np.copy(A)
B_null = np.copy(B)
K_null = np.copy(K)

np.savetxt('global_opt/baseline_controller.csv', K, delimiter=",", fmt='%f')

# Evolve D and get risk at each time point
orig_conops_reach_risk_eval = []
for i in range(1, steps):
    orig_risk = gen_constraint(i)(D.flatten())
    orig_conops_reach_risk_eval.append(np.maximum(orig_risk, 0))

nom_risk = 1 - np.prod(1 - np.array(orig_conops_reach_risk_eval))
print('Nominal Trajectory Risk {}'.format(nom_risk))
np.savetxt('global_opt/baseline_risk_thresholds.csv', orig_conops_reach_risk_eval, delimiter=",", fmt='%f')

# Visualize a trajectory and the reachset for the dynamics
start_state = np.zeros(state_size)
sample = SAMPLES[np.random.randint(100), :]
start_state[0] = c[0]  # x pos
start_state[1] = c[1]  # y pos
start_state[2] = 5

target = np.zeros(state_size)
target[2] = 5

states, controls, risk_measure, null_J = run_single_simulation(start_state, target, A, B, K, steps, Q, R, dt=dt, cl=D)
nominal_states = states

hazard = hazard_cause

A, B, K, Q, R = get_dynamics(hazard=hazard, dt=dt)
D = A - np.matmul(B, K)
D_unopt = np.copy(D)
K_unopt = np.copy(K)

if hazard == 'wind':
    wind_params = {'x': -1, 'y': -2.5, 'var_x': 0.2, 'var_y': 0.2}
    if D.shape[0] == 12:
        # simple model, effects xdot,ydot directly
        dyn_b = .1 * np.array([[0, 0], [0, 0], [0, 0],
                            [0, 0], [0, 0],[0, 0],
                            [1, 0], [0, 1], [0, 0],
                            [0, 0],[0, 0], [0, 0]])
        dyn_b = B_wind
    else:
        dyn_b = B
    
    wind_disturbance_center = get_wind_vector_center(wind_params['x'], wind_params['y'])
    wind_vals_x = np.random.normal(loc=wind_disturbance_center[0], scale=np.sqrt(np.abs(wind_disturbance_center[2])), size=steps)
    wind_vals_y = np.random.normal(loc=wind_disturbance_center[1], scale=np.sqrt(np.abs(wind_disturbance_center[3])), size=steps)
    wind_vals = {'dyn_b': dyn_b, 'x': wind_vals_x, 'y': wind_vals_y, 'p': wind_disturbance_center[2], 'q': wind_disturbance_center[3]}
else:
    wind_params = None
    dyn_b = B
    wind_vals = None

for doopt in [False, True]:
    fig, axs = plt.subplots(1, 1, figsize=(8, 5.2))
    if doopt:
        constraints = [NonlinearConstraint(gen_constraint(i, wind_params=wind_params, dyn_b=dyn_b), -1*np.inf, orig_conops_reach_risk_eval[i-1]) for i in range(1, steps)]
        start = time.time()
        result = minimize(objective(A, B, K), D.flatten(), constraints=constraints, options={'maxiter': 10000}) #, jac=grad_objective)
        end = time.time()

        print("Time elapsed for optimization {}".format(end - start))
        print("Optimization result: {}".format(result.success))

        # Save the computed Ks
        np.savetxt('global_opt/offline_optimized_cl_{}_controller.csv'.format(hazard), D_opt, delimiter=",", fmt='%f')
        np.savetxt('global_opt/offline_optimized_{}_controller.csv'.format(hazard), K_opt, delimiter=",", fmt='%f')
        
        D_opt = result.x.reshape((n_rows, n_cols))
        K_opt = np.matmul(np.linalg.pinv(B), A - D_opt)
        ax = axs
    else:
        D_opt = D
        K_opt = K
        ax = axs

    conops_reach_risk_eval = []
    conops_reach_risk_eval_unopt = []
    for i in range(1, steps):
        risk = gen_constraint(i, wind_params=wind_params, dyn_b=dyn_b)(D_opt.flatten())
        conops_reach_risk_eval.append(np.maximum(risk, 0))
        unopt_risk = gen_constraint(i, wind_params=wind_params, dyn_b=dyn_b)(D.flatten())
        conops_reach_risk_eval_unopt.append(np.maximum(unopt_risk, 0))

    opt_reach_risk = 1 - np.prod(1 - np.array(conops_reach_risk_eval))
    unopt_reach_risk = 1 - np.prod(1 - np.array(conops_reach_risk_eval_unopt))

    states, controls, risk_measure, J = run_single_simulation(start_state, target, A, B, K_opt, steps, Q, R, dt=dt, cl=D_opt, wind_vals=wind_vals)
    unopt_states, _, _, J_unopt = run_single_simulation(start_state, target, A, B, K, steps, Q, R, dt=dt, cl=D, wind_vals=wind_vals)

    # Plot results
    if doopt:
        label = 'Trajectory with PRREACH'
    else:
        label = 'Trajectory without PRREACH'

    print('Reach Risk No Optimization {}'.format(unopt_reach_risk))
    print('Risk Reduction {}'.format((unopt_reach_risk - opt_reach_risk)/unopt_reach_risk))
    print('Dist to Target {}'.format( (np.linalg.norm(states[-1, 0:2] - target[0:2]) - np.linalg.norm(unopt_states[-1, 0:2] - target[0:2]))/np.linalg.norm(unopt_states[-1, 0:2] - target[0:2]) ))

    # GET PATCHES
    cc = np.zeros(n_cols)
    cc[0] = c[0]
    cc[1] = c[1]
    cc[2] = 5
    v1c = np.zeros(n_cols)
    v1c[0] = v1[0]
    v1c[1] = v1[1]
    v1c[2] = 5
    v2c = np.zeros(n_cols)
    v2c[0] = v2[0]
    v2c[1] = v2[1]
    v2c[2] = 5
    v3c = np.zeros(n_cols)
    v3c[0] = v3[0]
    v3c[1] = v3[1]
    v3c[2] = 5
    v4c = np.zeros(n_cols)
    v4c[0] = v4[0]
    v4c[1] = v4[1]
    v4c[2] = 5
    len_x = np.abs(v1c[0]-v2c[0])
    len_y = np.abs(v1c[1]-v4c[1])
    bottom_left_x = cc[0] - np.abs((v1c[0]-v2c[0]) / 2)
    bottom_left_y = cc[1] - np.abs((v1c[1]-v4c[1]) / 2)
    rc_patches = []
    patch_alpha = 0.2
    rc = patches.Rectangle(((bottom_left_x, bottom_left_y)), len_x, len_y, 
                            linewidth=1, edgecolor='black', facecolor='blue', alpha=0.2)
    rc_patches.append(rc)

    cc_states, controls, risk_measure, J = run_single_simulation(cc, target, A, B, K_opt, steps, Q, R, dt=dt, cl=D_opt, wind_vals=wind_vals)
    v1c_states, controls, risk_measure, J = run_single_simulation(v1c, target, A, B, K_opt, steps, Q, R, dt=dt, cl=D_opt, wind_vals=wind_vals)
    v2c_states, controls, risk_measure, J = run_single_simulation(v2c, target, A, B, K_opt, steps, Q, R, dt=dt, cl=D_opt, wind_vals=wind_vals)
    v3c_states, controls, risk_measure, J = run_single_simulation(v3c, target, A, B, K_opt, steps, Q, R, dt=dt, cl=D_opt, wind_vals=wind_vals)
    v4c_states, controls, risk_measure, J = run_single_simulation(v4c, target, A, B, K_opt, steps, Q, R, dt=dt, cl=D_opt, wind_vals=wind_vals)


    for k in range(steps):
        cc = cc_states[k]
        v1c = v1c_states[k]
        v2c = v2c_states[k]
        v3c = v3c_states[k]
        v4c = v4c_states[k]
        len_x = np.abs(v1c[0]-v2c[0])
        len_y = np.abs(v1c[1]-v4c[1])
        bottom_left_x = cc[0] - np.abs((v1c[0]-v2c[0]) / 2)
        bottom_left_y = cc[1] - np.abs((v1c[1]-v4c[1]) / 2)
        if hazard == 'wind':
            r, k_zono = rho(k, D_opt, wind_params=wind_params, model_params=model_params, return_ztope=True, dyn_b=dyn_b)
            cc = k_zono['c']
            G = k_zono['G']
            g1 = G[:, 0]
            g2 = G[:, 1]
            v1c = cc - g1 - g2
            v2c = cc + g1 - g2
            v3c = cc + g1 + g2
            v4c = cc - g1 + g2
            len_x = np.abs(v1c[0]-v2c[0])
            len_y = np.abs(v1c[1]-v4c[1])
            bottom_left_x = cc[0] - np.abs((v1c[0]-v2c[0]) / 2)
            bottom_left_y = cc[1] - np.abs((v1c[1]-v4c[1]) / 2)
        else:
            r = rho(k, D_opt, wind_params=wind_params, model_params=model_params, return_ztope=True, dyn_b=dyn_b)
        
        if k == 1:
            rc = patches.Rectangle((bottom_left_x, bottom_left_y), len_x, len_y, 
                                linewidth=1, edgecolor='black', facecolor='blue', alpha=patch_alpha, label='Reach Sets')
        else:
            rc = patches.Rectangle((bottom_left_x, bottom_left_y), len_x, len_y,
                                linewidth=1, edgecolor='black', facecolor='blue', alpha=patch_alpha)
        rc_patches.append(rc)

    states = np.array(states)
    im = ax.imshow(heatmap, origin='lower', extent=[1,5,1,6], cmap='viridis', aspect='auto')
    if outcome == 'population':
        fig.colorbar(im, ax=ax, label="Population Density")
    else:
        fig.colorbar(im, ax=ax, label="Building Density")
    ax.plot(nominal_states[:, 0], nominal_states[:, 1], "-o", label='Nominal trajectory')
    ax.plot(states[:, 0], states[:, 1], "-o", label=label)
    for p in rc_patches:
        ax.add_patch(p)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid()
    ax.set_title('Hazard cause: {}, Risk: {}'.format(hazard, np.round(opt_reach_risk, 2)))
    ax.scatter([0], [0], color='red', label='Target', zorder=5, marker='x', s=100)
    ax.legend(loc='best')

    plt.show()

print("==========ONLINE COMPARISON WITH PRREACH (OFFLINE VS ONLINE) AND WITHOUT ==========")
D_offopt = np.copy(D_opt)
K_offopt = np.copy(K_opt)

report = {
    'start_point_id': [],
    'time_hazard_triggered': [],
    'class_PRA_decision': [],
    'eval_baseline_risk_at_hazard_time': [],
    'eval_unopt_reachset_risk_at_hazard_time': [],
    'eval_offopt_reachset_risk_at_hazard_time': [],
    'eval_onopt_reachset_risk_at_hazard_time': [],
    'unopt_cost': [],
    'offopt_cost': [],
    'onopt_cost': [],
    'unopt_dist': [],
    'offopt_dist': [],
    'onopt_dist': [],    
    'opt_runtime': [],
}

sample_id = 0

for sample_id in range(num_trials):
    n_rows = state_size  
    n_cols = state_size

    print('Sample Id {}'.format(sample_id))    
    start_point_x_y = SAMPLES[sample_id] # pick arbitrary point in initial reachset

    start_point = np.zeros(n_cols)
    start_point[0] = start_point_x_y[0]
    start_point[1] = start_point_x_y[1]
    target = np.zeros(n_cols)

    hazard_triggered = False
    class_PRA_decision_fly = False
    unopt_hazard_cause_fly = False
    offopt_hazard_cause_fly = False
    onopt_hazard_cause_fly = False

    cumul_unopt_hazard_cause_reachset_risk = 0
    cumul_offopt_hazard_cause_reachset_risk = 0
    cumul_onopt_hazard_cause_reachset_risk = 0

    possible_controllers = {}
    possible_cl = {}

    traj_xy_points = [start_point_x_y]

    state = start_point

    # Points Needed for Updating the Simplices 
    cc = np.zeros(n_cols)
    cc[0] = c[0]
    cc[1] = c[1]
    v1c = np.zeros(n_cols)
    v1c[0] = v1[0]
    v1c[1] = v1[1]
    v2c = np.zeros(n_cols)
    v2c[0] = v2[0]
    v2c[1] = v2[1]
    v3c = np.zeros(n_cols)
    v3c[0] = v3[0]
    v3c[1] = v3[1]
    v4c = np.zeros(n_cols)
    v4c[0] = v4[0]
    v4c[1] = v4[1]

    # Begin simulation trial
    cl_system = D_null
    p = np.random.random(1)
    trigger_time = trigger_time_start
    if p <= hazard_prob:
        # TODO: Get Decision (continue/land) for classical PRA
        class_PRA_decision_fly = True  # FOR NOW SET TO TRUE 

    for k in range(1, steps):
        if (trigger_time_start <= k <= trigger_time_end) and (not hazard_triggered):
            if p <= hazard_prob:
                print('Hazard Triggered at time {}'.format(k))
                hazard_triggered = True
                trigger_time = k
            else:
                p = np.random.random(1)  # if no prob this time try again next iteration

        if hazard_triggered:
            # Risk Threshold for remaining flight ie RISK TO GO
            risk_threshold = orig_conops_reach_risk_eval[k-1:]
            cumul_risk_threshold = 1 - np.prod(1 - np.array(risk_threshold))

            # Update Simplex Map Computing remaining online risk
            cS1 = np.array([v1c[0:2], cc[0:2], v2c[0:2]]).T
            cS2 = np.array([v2c[0:2], cc[0:2], v3c[0:2]]).T
            cS3 = np.array([v3c[0:2], cc[0:2], v4c[0:2]]).T
            cS4 = np.array([v4c[0:2], cc[0:2], v1c[0:2]]).T
            csimplex_map = {0: cS1, 1: cS2, 2: cS3, 3: cS4}

            # Use remaining risk threshold to online optimize a new controller
            constraints = [NonlinearConstraint(gen_constraint(jk-trigger_time, simplex_map=csimplex_map, wind_params=wind_params, dyn_b=dyn_b), -1*np.inf, orig_conops_reach_risk_eval[jk-1]) for jk in range(k+1, steps)]
            start = time.time()
            result = minimize(objective(A, B, K), D_offopt.flatten(), constraints=constraints, options={'maxiter': 10000}) #, jac=grad_objective)
            end = time.time()
            runtime = end - start
            print("Time elapsed for optimization {}".format(runtime))
            print("Opt Result {}".format(result.success))
            D_onopt = result.x.reshape((n_rows, n_cols))
            K_onopt = np.matmul(np.linalg.pinv(B), A - D_onopt)

            # Compute and compare Risk To go
            unopt_hazard_cause_reachset_risk = []
            offopt_hazard_cause_reachset_risk = []
            onopt_hazard_cause_reachset_risk = []
            for jk in range(k+1, steps):
                fr = jk-trigger_time
                unopt_risk = gen_constraint(fr, simplex_map=csimplex_map, wind_params=wind_params, dyn_b=dyn_b)(D_unopt.flatten())
                unopt_hazard_cause_reachset_risk.append(np.maximum(unopt_risk, 0))
                offopt_risk = gen_constraint(fr, simplex_map=csimplex_map, wind_params=wind_params, dyn_b=dyn_b)(D_offopt.flatten())
                offopt_hazard_cause_reachset_risk.append(np.maximum(offopt_risk, 0))
                onopt_risk = gen_constraint(fr, simplex_map=csimplex_map, wind_params=wind_params, dyn_b=dyn_b)(D_onopt.flatten())
                onopt_hazard_cause_reachset_risk.append(np.maximum(onopt_risk, 0))

            cumul_unopt_hazard_cause_reachset_risk = 1 - np.prod(1 - np.array(unopt_hazard_cause_reachset_risk))
            cumul_offopt_hazard_cause_reachset_risk = 1 - np.prod(1 - np.array(offopt_hazard_cause_reachset_risk))
            cumul_onopt_hazard_cause_reachset_risk = 1 - np.prod(1 - np.array(onopt_hazard_cause_reachset_risk))

            print('Unopt Risk {}'.format(cumul_unopt_hazard_cause_reachset_risk))
            print('Off Opt Risk {}'.format(cumul_offopt_hazard_cause_reachset_risk))
            print('Online Opt Risk {}'.format(cumul_onopt_hazard_cause_reachset_risk))

            # Compute and compare controller costs
            unopt_traj, _, _, J_unopt = run_single_simulation(state, target, A, B, K_unopt, steps-k, Q, R, dt=dt, cl=D_unopt, wind_vals=wind_vals)
            offopt_traj, _, _, J_offopt = run_single_simulation(state, target, A, B, K_offopt, steps-k, Q, R, dt=dt, cl=D_offopt, wind_vals=wind_vals)
            onopt_traj, _, _, J_onopt = run_single_simulation(state, target, A, B, K_onopt, steps-k, Q, R, dt=dt, cl=D_onopt, wind_vals=wind_vals)
            unopt_traj = np.array(unopt_traj)
            offopt_traj = np.array(offopt_traj)
            onopt_traj = np.array(onopt_traj)

            print('Unopt J {}'.format(J_unopt))
            print('Off Opt J {}'.format(J_offopt))
            print('Online Opt J {}'.format(J_onopt))

            # Compute and compare distance from target
            final_dist_unopt = np.linalg.norm(unopt_traj[-1, 0:2] - target[0:2])**2
            final_dist_offopt = np.linalg.norm(offopt_traj[-1, 0:2] - target[0:2])**2
            final_dist_onopt = np.linalg.norm(onopt_traj[-1, 0:2] - target[0:2])**2

            print('Unopt distance {}'.format(final_dist_unopt))
            print('Off Opt distance {}'.format(final_dist_offopt))
            print('Online Opt distance {}'.format(final_dist_onopt))

            report['start_point_id'].append(sample_id)
            report['time_hazard_triggered'].append(trigger_time)
            report['class_PRA_decision'].append(class_PRA_decision_fly)
            report['eval_baseline_risk_at_hazard_time'].append(cumul_risk_threshold)
            report['eval_unopt_reachset_risk_at_hazard_time'].append(cumul_unopt_hazard_cause_reachset_risk)
            report['eval_offopt_reachset_risk_at_hazard_time'].append(cumul_offopt_hazard_cause_reachset_risk)
            report['eval_onopt_reachset_risk_at_hazard_time'].append(cumul_onopt_hazard_cause_reachset_risk)
            report['unopt_cost'].append(J_unopt)
            report['offopt_cost'].append(J_offopt)
            report['onopt_cost'].append(J_onopt)
            report['unopt_dist'].append(final_dist_unopt)
            report['offopt_dist'].append(final_dist_offopt)
            report['onopt_dist'].append(final_dist_onopt)
            report['opt_runtime'].append(runtime)

            fig_params = {
                'axes.labelsize': 14,
                'font.size': 14,
                'legend.fontsize': 14,
                'font.family': 'sans-serif',
                'font.sans-serif': 'Helvetica',
                'text.usetex': False,
                'figure.figsize': (12, 8)
                }
            mpl.rcParams.update(fig_params)
            # plt.figure(figsize=(12, 8))
            plt.imshow(heatmap, origin='lower', extent=[1,5,1,6], cmap='viridis', aspect='auto')
            if outcome == 'population':
                plt.colorbar(label="Population Density")
            else:
                plt.colorbar(label="Building Density")
            plt.plot(unopt_traj[:, 0], unopt_traj[:, 1], "-o", label='Trajectory under LQR')
            plt.plot(offopt_traj[:, 0], offopt_traj[:, 1], "-s", label='Trajectory under PRR-offline')
            plt.plot(onopt_traj[:, 0], onopt_traj[:, 1], "-*", label='Trajectory under PRR-online')
            plt.scatter([0], [0], color='red', label='Target', zorder=5, marker='x', s=200)
            plt.xlabel("x position")
            plt.ylabel("y position")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.savefig('{}_{}_{}.png'.format(hazard, outcome, sample_id))
            plt.close()
            break
        
        else:
            # If no hazard, evolve state using null hazard dynamics
            control = lqr_control(state, target, K_null)
            state = A_null @ state + B_null @ control
            xy_point = np.array([state[0], state[1]])
            traj_xy_points.append(xy_point)  # append new xy position

            # Update the zonotope simplex points
            control = lqr_control(cc, target, K_null)
            cc = A_null @ cc + B_null @ control
            control = lqr_control(v1c, target, K_null)
            v1c = A_null @ v1c + B_null @ control
            control = lqr_control(v2c, target, K_null)
            v2c = A_null @ v2c + B_null @ control
            control = lqr_control(v3c, target, K_null)
            v3c = A_null @ v3c + B_null @ control
            control = lqr_control(v4c, target, K_null)
            v4c = A_null @ v4c + B_null @ control

# print(report)
report_df = pd.DataFrame.from_dict(report)
report_df.to_csv('results_{}_{}.csv'.format(hazard, outcome), index=False)