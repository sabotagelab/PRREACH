import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize

np.random.seed(0)

# PARAMS
num_simplices = 4
num_lin_forms = 3
TR = np.array([0.5,1])
c = np.array([5, 5]) + TR 
g1 = np.array([1, 0])
g2 = np.array([0, 1])
v1 = np.array([4, 4]) + TR
v2 = np.array([6, 4]) + TR
v3 = np.array([6, 6]) + TR
v4 = np.array([4, 6]) + TR
E1 = np.array([g1, -g1]).T
E2 = np.array([g1, g2]).T
E3 = np.array([g1, -g2]).T
E4 = np.array([-g1, g2]).T
E5 = np.array([-g1, -g2]).T
E6 = np.array([g2, -g2]).T
ztope_map = {0: E1, 1: E2, 2: E3, 3: E4, 4: E5, 5: E6}
S1 = np.array([v1, c, v2]).T
S2 = np.array([v2, c, v3]).T
S3 = np.array([v3, c, v4]).T
S4 = np.array([v4, c, v1]).T
simplex_map = {0: S1, 1: S2, 2: S3, 3: S4}

SAMPLE_NUMBER = 1000
X = np.random.uniform(c[0]-1, c[0]+1, size=SAMPLE_NUMBER)
Y = np.random.uniform(c[1]-1, c[1]+1, size=SAMPLE_NUMBER)
SAMPLES = np.column_stack([X, Y])

# SKELETON FOR WIND ZONOTOPE
CdA = 0.08  # m^2 drag area for UAV
air_density = 1.225  # kg/m^3
wc = np.array([0, 0])  # 0, 0 wind velocity in x and y direction by default
wg1 = np.array([0.5, 0.5])  # varying wind speed by +-0.5 in x direction
wg2 = np.array([0.5, -0.5])  # varying wind speed by +-0.5 in y direction

EPS = 1e-3

# Hazard Heatmap and Related Polynomial Func
pop_heatmap = np.loadtxt('pop_density.csv', delimiter=',')
build_heatmap = np.loadtxt('Building Density/building_density.csv', delimiter=',')

# p_theta func approximation of the population density map
# 0.0016*(3.4617x + 9.0848y) + 0.6528*(-0.0001y)^2 + 0.0499*(-0.0944x + -0.1049y)^3
c0 = 0.0016
alpha0 = 3.4617
beta0 = 9.0848
c1 = 0.6528
alpha1 = 0.0000
beta1 = -0.0001
c2 = 0.0499
alpha2 = -0.0944
beta2 = -0.1049

p_theta_params = np.array([c0, alpha0, beta0, c1, alpha1, beta1, c2, alpha2, beta2])

# p_theta func approx for building density map
c0_ = 0.0937
alpha0_ = 0.8656
beta0_ = 0.1108
c1_ = 0.5548
alpha1_ = 0.0000
beta1_ = 0.0000
c2_ = 0.1351
alpha2_ = -0.2470
beta2_ = -0.0083

p_building_theta_params = np.array([c0_, alpha0_, beta0_, c1_, alpha1_, beta1_, c2_, alpha2_, beta2_])

p_combined_params = np.array([c0 + c0_, alpha0 + alpha0_, beta0 + beta0_, c1 + c1_, alpha1 + alpha1_, beta1 + beta1_, c2 + c2_, alpha2 + alpha2_, beta2 + beta2_])



################### Functions for the Polynomial Model ##########
def model(params, x1, x2):
    degree = int(len(params)/3.0)
    params = params.reshape((degree,3))
    result = 0
    for d in range(degree):
        c, alpha, beta = params[d,:]
        result += c * (alpha*x1 + beta*x2)**(d+1)
    return result

def dmodel(params, x1, x2):
    degree = int(len(params)/3.0)
    params = params.reshape((degree,3))
    result = 0
    for d in range(degree):
        c, alpha, beta = params[d,:]
        result += c * (d+1) * (alpha*x1 + beta*x2)**d
    return result

def risk(p, params=p_theta_params):
    x1 = p[0]
    x2 = p[1]
    if not ((1 <= x1 <= 5) and (1 <= x2 <= 6)):
        return 0
    return np.maximum(model(params, x1, x2), 0.0)


################### Computing Constraint 
def rho(k, D, simplex_map=simplex_map, wind_params=None, model_params=p_theta_params, return_ztope=False, dyn_b=None):
    if wind_params is not None:
        # Compute the zonotope directly using closed loop system with controller given by matrix D 
        # and the constant wind zonotope given by the wind params
        
        # fully compute reachset at time k with wind disturbance
        wx = wind_params['x']
        wy = wind_params['y']
        wind_var_x = wind_params['var_x']
        wind_var_y = wind_params['var_y']
        if D.shape[0] == 4:
            wind_zono = {'c': np.array([wx, wy]), 'G': np.array([[wind_var_x, wind_var_y], [wind_var_y, -wind_var_x]])}
            X0_extend = {'c': np.array([c[0], c[1], 0, 0]), 'G': np.array([[1, 0], [0, 1], [0, 0], [0, 0]])}
        elif D.shape[0] == 12:
            # x= [x,y,z, roll,pitch,yaw, vx,vy,vz, roll_dot,pitch_dot,yaw_dot]
            # wind_zono = {'c': np.array([wx, wy]), 'G': np.array([[wind_var_x, wind_var_y], [wind_var_y, -wind_var_x]])}
            wind_disturbance_center = get_wind_vector_center(wx, wy)
            wind_zono = {'c': wind_disturbance_center, 'G': np.array([[wind_disturbance_center[2], 0], [0, wind_disturbance_center[3]], [0, 0], [0, 0]])}
            X0_extend = {'c': np.array([c[0], c[1], 0, 0,0,0, 0,0,0, 0,0,0]), 
                         'G': np.array([[1, 0], [0, 1], [0, 0], 
                                        [0, 0], [0, 0], [0, 0],
                                        [0, 0], [0, 0], [0, 0],
                                        [0, 0], [0, 0], [0, 0]])}
        k_zono = X0_extend
        wind_zono = zono_matmul(dyn_b, wind_zono)
        cumul_disturb_zono = wind_zono
        for i in range(1, k-1):
            Di = np.linalg.matrix_power(D, i)
            wind_zono_step = zono_matmul(Di, wind_zono)
            cumul_disturb_zono = zonotope_add(cumul_disturb_zono, wind_zono_step)  # apply disturbance
        Dk = np.linalg.matrix_power(D, k)
        k_zono = zono_matmul(Dk, k_zono) # Apply control at this step
        k_zono = zonotope_add(k_zono, cumul_disturb_zono)  # apply disturbance

        # project down to 2d
        k_zono_proj = zonotope_proj(k_zono)
        k_zono_approx = zonotope_overapprox(k_zono_proj)
        k_zono_simplex_map = zonotope_decompose(k_zono_approx)

        # get volume of this new zonotope
        g1 = k_zono_approx['G'][:, 0]
        g2 = k_zono_approx['G'][:, 1]
        E1 = np.array([g1, -g1]).T
        E2 = np.array([g1, g2]).T
        E3 = np.array([g1, -g2]).T
        E4 = np.array([-g1, g2]).T
        E5 = np.array([-g1, -g2]).T
        E6 = np.array([g2, -g2]).T
        k_ztope_map = {0: E1, 1: E2, 2: E3, 3: E4, 4: E5, 5: E6}

        numer = FD(k, D, simplex_map=k_zono_simplex_map, model_params=model_params)
        vol = GD(k, D, ztope_sub_mat=k_ztope_map)
        if return_ztope:
            return numer / (vol + EPS), k_zono_approx 
    else:
        numer = FD(k, D, simplex_map=simplex_map, model_params=model_params) 
        vol = GD(k, D)
    return numer / (vol + EPS)


################### Computing Denominator of the Constraint and its Derivative
def GD(k, D, ztope_sub_mat=None):
    # Computes Volume of the Zonotope

    if ztope_sub_mat is not None:
        # If zonotope is given directly as set of generators compute it directly
        result = 0 
        if isinstance(ztope_sub_mat, dict):
            for i, E in ztope_map.items():
                result += np.abs(np.linalg.det(E))
        else:
            for m in ztope_sub_mat:
                result += np.abs(np.linalg.det(m))
        return result

    # assumes that disturbance has not external (ie change to inital square zonotope X0 is only matrix multiplication)
    # then volume of new zonotope is det(scaling matrix D) and the original volume of X0
    Dk = np.linalg.matrix_power(D, k)  
    if D.shape[0] == 12:
        P1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # only get x,y position
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    elif D.shape[0] == 13:
        P1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # only get x,y position
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    elif D.shape[0] == 4:
        P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    elif D.shape[0] == 5:
        P1 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
    else:
        P1 = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = np.matmul(P1, np.matmul(Dk, P2))
    result = 0
    for i, E in ztope_map.items():
        result += np.abs(np.linalg.det(E))
    return result * np.abs(np.linalg.det(Q))



################### Computing the non-simplified Numerator of the Constraint and its Derivative

def FD(k, D, simplex_map=simplex_map, simplices=None, model_params=p_theta_params):
    if simplices is not None:
        res = 0
        for i in range(len(simplices)):
            for m in range(num_lin_forms):
                res += fk(k, D, i, m, simp=simplices[i], model_params=model_params)    
        return res
    res = 0
    for i in range(num_simplices):
        for m in range(num_lin_forms):
            res += fk(k, D, i, m, simplex_map=simplex_map, model_params=model_params)
    return res

def fk(k, D, i, m, simplex_map=simplex_map, simp=None, model_params=p_theta_params):
    if simp is not None:
        DkS = simp
    else:
        S = simplex_map[i]
        
        Dk = np.linalg.matrix_power(D, k)
        if D.shape[0] == 12:
            P1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # only get x,y position
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        elif D.shape[0] == 13:
            P1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # only get x,y position
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        elif D.shape[0] == 4:
            P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        elif D.shape[0] == 5:
            P1 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
            P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
        else:
            P1 = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
            P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
        Q = np.matmul(P1, np.matmul(Dk, P2))
        DkS = np.matmul(Q, S)

    degree = int(len(model_params)/3.0)
    params = model_params.reshape((degree,3))
    c, alpha, beta = params[m,:]
    lm = np.array([c*alpha, c*beta])

    if m == 0: # degree 1
        ps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif m == 1: # degree 2
        ps = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    elif m == 2: # degree 3
        ps = [[3, 0, 0], [0, 3, 0], [0, 3, 0], 
              [2, 1, 0], [2, 0, 1], 
              [1, 2, 0], [1, 0, 2], 
              [0, 1, 2], [0, 2, 1], [1, 1, 1]]
    else: # degree 4
        ps = [[4, 0, 0], [0, 4, 0], [0, 0, 4],
              [3, 1, 0], [3, 0, 1], [0, 3, 1], [1, 3, 0], [1, 0, 3], [0, 1, 3], 
              [2, 2, 0], [2, 0, 2], [0, 2, 2]]
        
    sum_over_combos = 0
    for p in ps:
        sum_over_combos += (np.dot(lm, DkS[:, 0])**p[0] * np.dot(lm, DkS[:, 1])**p[1] * np.dot(lm, DkS[:, 2])**p[2])

    return 2 * vol_simp(k, D, i) * math.factorial(m+1)/math.factorial(m+3) * sum_over_combos


def vol_simp(k, D, i, simplex_map=simplex_map, simp=None):
    if simp is not None:
        Si = np.vstack([simp, np.ones(3)])
        return 0.5 * np.abs(np.linalg.det(Si))
    S = simplex_map[i]
    Dk = np.linalg.matrix_power(D, k)
    if D.shape[0] == 12:
        P1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # only get x,y position
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    elif D.shape[0] == 13:
        P1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # only get x,y position
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    elif D.shape[0] == 4:
        P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    elif D.shape[0] == 5:
        P1 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
    else:
        P1 = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        P2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = np.matmul(P1, np.matmul(Dk, P2))
    Si = np.vstack([S, np.ones(3)])
    return 0.5 * np.abs(np.linalg.det(Q)) *np.abs(np.linalg.det(Si))


################### Other Helper Functions
def zono_matmul(D, Z):
    gs = []
    for g in Z['G'].T:
        gs.append(D @ g)
    return {'c': D @ Z['c'], 'G': np.array(gs).T}

def zonotope_add(z1, z2):
    c1 = z1['c']
    c2 = z2['c']
    G1 = z1['G']
    G2 = z2['G']
    Z_c = c1 + c2
    Z_G = np.block([G1, G2])

    return {'c': Z_c, 'G': Z_G}

def zonotope_proj(Z):
    # project zonotope into 2d
    proj_c = Z['c'][0:2]
    proj_G = Z['G'][0:2, ]
    return {'c': np.copy(proj_c), 'G': np.copy(proj_G)}

def zonotope_overapprox(Z):
    # overapproximate into square
    G = Z['G']
    
    adj_x = np.sum(G[0, 2:])  # sum across latter columns of first row
    adj_y = np.sum(G[1, 2:])  # sum across latter columns of second row
    adj = np.maximum(adj_x, adj_y)
    new_G = np.copy(G[:,0:2])  # keep only first two columns
    new_G[0, 0] += adj
    new_G[1, 1] += adj
    return {'c': Z['c'], 'G': new_G}

def zonotope_decompose(Z):
    # decompose into simplices
    center = Z['c']
    G = Z['G']
    g1 = G[:, 0]
    g2 = G[:, 1]
    v1 = center - g1 - g2
    v2 = center + g1 - g2
    v3 = center + g1 + g2
    v4 = center - g1 + g2
    
    S1 = np.array([v1, c, v2]).T
    S2 = np.array([v2, c, v3]).T
    S3 = np.array([v3, c, v4]).T
    S4 = np.array([v4, c, v1]).T
    simplex_map = {0: S1, 1: S2, 2: S3, 3: S4}
    return simplex_map

def ddet(matrix):
    """Calculates the adjugate of a matrix."""

    cofactor_matrix = np.linalg.inv(matrix).T * np.linalg.det(matrix)
    return cofactor_matrix.T

def get_wind_vector_center(x_rel_vel_wind, y_rel_vel_wind):
    # Inputs: xy components of wind relative to drone
    
    # get forces on drone
    fx = -0.5 * air_density * CdA * x_rel_vel_wind * np.abs(x_rel_vel_wind)
    fy = -0.5 * air_density * CdA * y_rel_vel_wind * np.abs(y_rel_vel_wind)

    # get torques on drone
    tx = 0.1 * -0.5 * air_density * CdA * np.abs(x_rel_vel_wind)**2
    ty = 0.1 * -0.5 * air_density * CdA * np.abs(y_rel_vel_wind)**2

    return np.array([fx, fy, tx, ty])

################### Control Helper Functions
def lqr_control(state, target, K, dt=0.1):
    return -K @ (state - target)


def LQR(A, B, Q, R):    
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A

    return np.asarray(K)

def run_single_simulation(start_state, target, A, B, K, steps, Q, R, dt=0.02, cl=None, wind_vals=None):
    state = start_state
    states = [state]
    controls = []
    point_risk = []
    J = 0  ## Controller Cost
    
    for t in range(steps):
        # Calculate control input
        control = lqr_control(state, target, K, dt=dt)
        
        if cl is not None:
            prev_state = np.copy(state)
            state = cl @ state  ## closed loop control
            u = np.matmul(np.linalg.pinv(B), (A @ prev_state - state))
            J += controller_cost(prev_state, u, Q, R)
            if wind_vals is not None:
                dyn_b = wind_vals['dyn_b']
                wu_x = wind_vals['x'][t]
                wu_y = wind_vals['y'][t]
                if A.shape[0] == 4:    
                    state += B @ np.array([wu_x, wu_y])
                else:
                    wu_p = wind_vals['p']
                    wu_q = wind_vals['q']
                    state += dyn_b @ np.array([wu_x, wu_y, wu_p, wu_q])
        else:    
            J += controller_cost(state, control, Q, R)
            state = A @ state + B @ control  ## open loop control based on reference target point
            if wind_vals is not None:
                dyn_b = wind_vals['dyn_b']
                wu_x = wind_vals['x'][t]
                wu_y = wind_vals['y'][t]
                if A.shape[0] == 4:    
                    state += B @ np.array([wu_x, wu_y])
                else:
                    wu_p = wind_vals['p']
                    wu_q = wind_vals['q']
                    state += dyn_b @ np.array([wu_x, wu_y, wu_p, wu_q])

        # Record state and control
        states.append(state)
        controls.append(control)
        
        # record point risk
        point_risk.append(1 - risk(state))

    J += state.T @ Q @ state
    # Convert results to arrays for plotting
    states = np.array(states)
    controls = np.array(controls)
    
    # compute aggregate risk measure
    risk_measure = 1 - np.prod(point_risk)
    return states, controls, risk_measure, J

def controller_cost(x, u, Q, R):
    return x.T @ Q @ x + u.T @ R @ u

def opt_matrix_inverse(B):
    ## solve optimization to get matrix inverse of the control matrix B
    m, n = B.shape

    # Initial guess: Random matrix of appropriate shape
    X0 = np.random.randn(n, m).flatten()
    
    # Define the loss function (Frobenius norm of A * X * A - A and X * A * X - X)
    def loss(X):
        X = X.reshape((n, m))
        return np.linalg.norm(B @ X @ B - B, 'fro') + np.linalg.norm(X @ B @ X - X, 'fro')
    
    # Minimize the loss function
    result = scipy.optimize.minimize(loss, X0, method='Nelder-Mead')
    
    use_svd = False
    if result.success:
        return result.x.reshape((n, m)), use_svd
    else:
        print(result)
        use_svd = True
        return np.linalg.pinv(B), use_svd


if __name__== "__main__":
    Z1 = {'c': np.array([c[0], c[1], 0, 0]), 'G': np.array([[1, 0], [0, 1], [0, 0], [0, 0]])}
    const_disturb = {'c': np.array([0, 0, wc[0], wc[1]]), 'G': np.array([[0, 0], [0, 0], [0.5, 0.5], [0.5, -0.5]])}
    Z_disturb = zonotope_add(Z1, const_disturb)

    A = np.array([[0, 0, 1, 0],
            [0, 0, 0, 1], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]])
    # Dynamics control matrix B
    B = np.array([[0., 0.],
                [0., 0.],
                [1., 0.],
                [0., 1.]])
    
    # === Discretize the System ===
    dt = 0.02
    Ad = scipy.linalg.expm(A * dt)  # Discretized A using matrix exponential
    Bd = np.zeros_like(B)
    n = 1000 # number of integration steps
    for i in range(1, n + 1):
        tau = i * dt/n
        Bdd = scipy.linalg.expm(A * tau) @ B * dt/n
        Bd += Bdd

    # === Define Cost Matrices for LQR ===    
    Q = np.diag([12, 5, .1, .1])

    
    R = np.diag([1, 1])  # Control cost

    # Solve Discrete-Time Algebraic Riccati Equation (DARE)
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

    # Compute Discrete-Time LQR Gain: K = (B^T P B + R)^(-1) B^T P A
    Kd = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    A_cl = Ad - np.matmul(Bd, Kd)

    print('step 1')
    constraint_val = rho(1, A_cl)
    print(constraint_val)
    # constraint_val = rho(1, A_cl, wind_params={'x': 0, 'y': 0, 'var_x': 0.01, 'var_y': 0.01})
    # print(constraint_val)
    constraint_val = rho(1, A_cl, wind_params={'x': 5, 'y': 5, 'var_x': 1, 'var_y': 1}, dyn_b=B)
    print(constraint_val)

    print('step 2')
    constraint_val = rho(2, A_cl)
    print(constraint_val)
    # constraint_val = rho(2, A_cl, wind_params={'x': 0, 'y': 0, 'var_x': 0.01, 'var_y': 0.01})
    # print(constraint_val)
    constraint_val = rho(2, A_cl, wind_params={'x': 5, 'y': 5, 'var_x': 1, 'var_y': 1}, dyn_b=B)
    print(constraint_val)

    print('step 10')
    constraint_val = rho(10, A_cl)
    print(constraint_val)
    # constraint_val = rho(10, A_cl, wind_params={'x': 0, 'y': 0, 'var_x': 0.01, 'var_y': 0.01})
    # print(constraint_val)
    constraint_val = rho(10, A_cl, wind_params={'x': 5, 'y': 5, 'var_x': 1, 'var_y': 1}, dyn_b=B)
    print(constraint_val)

    print('step 20')
    constraint_val = rho(20, A_cl)
    print(constraint_val)
    # constraint_val = rho(20, A_cl, wind_params={'x': 0, 'y': 0, 'var_x': 0.01, 'var_y': 0.01})
    # print(constraint_val)
    constraint_val = rho(20, A_cl, wind_params={'x': 5, 'y': 5, 'var_x': 1, 'var_y': 1}, dyn_b=B)
    print(constraint_val)