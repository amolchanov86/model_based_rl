#!/usr/bin/env python

"""
Model Predictive Path Integral: example of a simple affine quadrotor model
See MPPI paper: https://arc.aiaa.org/doi/pdf/10.2514/1.G001921
Calculations are based on Section III.c that assumes a special form of dynamical systems:
dx = f(x,t)*dt + g(x,t)*(u + du)*dt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import copy

EPS = 1e-6
s_dim =  3 #state dimensions
# GRAV = 9.81
GRAV = 0.

def rollout(x_0, f_fn, G_fn, u_seq, delta_t):
    N = u_seq.shape[0] # u_seq.shape = (N,u_dim), where N = timesteps
    x_dim = x_0.shape[0]
    x_out = np.zeros((N, x_dim)) #trajectory, i.e. (timesteps, x_dim)
    x = 0 + x_0 #???
    for i, u in enumerate(u_seq):
        x_out[i] = x
        # print("shapes: f,G,u,x", f_fn(x).shape, G_fn(x).shape, u.shape, x.shape)
        x += delta_t * (f_fn(x) + G_fn(x) @ u)
    return x_out


def rollout_real(x_0, f_fn, G_fn, u_seq, delta_t, u_min=-11, u_max=11):
    u_seq_clipped = np.clip(u_seq, a_min=u_min, a_max=u_max)
    N = u_seq.shape[0] # u_seq.shape = (N,u_dim), where N = timesteps
    x_dim = x_0.shape[0]
    x_out = np.zeros((N, x_dim)) #trajectory, i.e. (timesteps, x_dim)
    x = 0 + x_0 #???
    for i, u in enumerate(u_seq_clipped):
        # print("U:", u)
        x_out[i] = x
        # print("shapes: f,G,u,x", f_fn(x).shape, G_fn(x).shape, u.shape, x.shape)
        x += delta_t * (f_fn(x) + G_fn(x) @ u)
    return x_out

def q_tilde(q_fn, x, u, du, R, nu):
    """
    Modified Running cost, i.e. cost for the current state and action 
    This is a simplification from III.c section (see the end)
    The assumption: special case of dynamical systems
    du = 1/sqrt(ro) * eps /sqrt(ro) - added perturbation noise
    R - positive definite matrix of control costs
    nu - exploration weight (i.e. penalty for exploration)
    """
    return (
        q_fn(x) +
        (0.5 - 0.5 / nu) * du.T @ R @ du +
        u.T @ R @ du +
        0.5 * u.T @ R @ u
    )

# TODO make faster all-steps-at-once matrix version of q_tilde
def S_tilde(q_fn, x_seq, u_seq, du_seq):
    """
    Cost-to-go for each time step
    NOTE: in the paper cost-to-go is computed as actually cost-so-far (accumulated cost):
    S_tilde(t+1) = S_tilde(t) + q_tilde
    which is strange. Same thing happens in the rally paper: 
    https://arxiv.org/pdf/1707.04540.pdf
    """
    N = len(x_seq) # num of timesteps

    # Computing running costs for each timestep
    Qs = np.array([q_fn(x, u, du)
        for x, u, du in zip(x_seq, u_seq, du_seq)])
    # cumsum() == cumulative sum, i.e. sum of all elements up until the current element
    # cost2go = Qs[::-1].cumsum()[::-1]
    cost2go = Qs.cumsum()
    return cost2go


def mppi_step(f_fn, G_fn, q_fn, R, rho, nu, lamd, delta_t, x_0, u_seq, K):

    N, u_dim = u_seq.shape #N = timesteps

    # Generatinig random control variations
    epsilons = np.random.normal(size=(K, N, u_dim)) #K = number of trajectories
    dus = (1.0 / np.sqrt(rho)) * (1.0 / np.sqrt(delta_t)) * epsilons #Eq.42
    S = np.zeros((K, N))

    def q_tilde_fn(x, u, du):
        return q_tilde(q_fn, x, u, du, R, nu)

    def q_raw_fn(x, u, du):
        return q_fn(x)

    # Iterating through trajectories
    traj = []
    for i in range(K):
        u_perturbed = u_seq + dus[i]
        ## Clipping controls
        # u_perturbed = np.clip(u_perturbed, a_min=0, a_max=1.0)
        # dus[i] = u_perturbed - u_seq
        # print("u_perturbed:", u_perturbed[0])
        # print("dus:", dus[i])

        x_seq = rollout(x_0, f_fn, G_fn, u_perturbed, delta_t)
        #Computing costs-to-go for each trajectory
        S[i] = S_tilde(q_tilde_fn, x_seq, u_seq, dus[i]) #S.shape == (K,N)
        traj.append(x_seq)

    # Computing weights for trajectories
    expS = np.exp((-1.0/lamd) * S) + EPS # (K, N) = (traj_num,timelen)
    denom = np.sum(expS, axis=0) # (N) = (timelen)
    
    # Weighting trajectories to find control updates
    du_weighted = expS[:,:,None] * dus # (K, N, udim)
    u_change_unscaled = np.sum(du_weighted, axis=0) # (N, udim): averaging among traj.
    u_change = u_change_unscaled / denom[:,None] # (N, udim)

    return u_seq + u_change, traj, S

w_x = .25
w_y = 1.
w_z = .5
def collision_check(s):
    return False
    # wall_high = np.array([w_x, w_y, w_z])[:s_dim]
    # wall_low = -wall_high
    # return np.all(s[:2] < wall_high) and np.all(s[:2] > wall_low)
   

# draw rect
def draw_wall(ax, x_range, y_range):
    # TODO: refactor this to use an iterotor
    rect_w = x_range[1] - x_range[0]
    rect_h = y_range[1] - y_range[0]
    rect = patches.Rectangle((x_range[0], y_range[0]), rect_w, rect_h, facecolor=(0, 0, 0))
    ax.add_patch(rect)

# draw rect
def draw_quad(ax, xyz, rot, scale):
    ax.scatter([xyz[0]],[xyz[1]], s=25, c="r", marker="^")
    x_axis = rot[:,0]
    y_axis = rot[:,1]
    ax.plot(
        [xyz[0], xyz[0] + scale*x_axis[0]], 
        [xyz[1], xyz[1] + scale*x_axis[1]], 
        color='red', alpha=0.8, lw=3)
    ax.plot(
        [xyz[0], xyz[0] + scale*y_axis[0]], 
        [xyz[1], xyz[1] + scale*y_axis[1]], 
        color='green', alpha=0.8, lw=3)


class AffineQuadrotorDynamics2D(object):
    """
    State: [xyz,Vxyz,Euler,Omega] = 12-d
      Where: Omega - angular rates in the body frame
    Damping:
      - linear on velocity
      - quadratic on rotational acceleration
      - inertialess rotor dynamics
    """
    def __init__(self):
        ###############################
        ## PARAMETERS
        self.mass = 0.5
        self.arm_length = 0.33 / 2.0
        self.inertia = self.mass * 0.01

        self.thrust_to_weight = 2.0
        self.vel_damp = 0.001
        self.damp_omega = 0.015
        # self.damp_omega = 0.0
        self.torque_to_thrust=0.05

        self.Fmax = 9.81 * self.mass * self.thrust_to_weight / 2.0

        ###############################
        # Auxiliarry matrices
        self.thrust_sum_mx = np.zeros([2,2]) # [0,0,F_sum].T
        self.thrust_sum_mx[1,:] = 1# [0,0,F_sum].T

        FA = self.arm_length*self.Fmax
        self.M_omega = np.array([
            [ -FA,   FA]])
        self.M_omega = (1.0 / self.inertia) * self.M_omega

        self.zero2x2 = np.zeros([2,2])
        self.zero1x2 = np.zeros([1,2])

        self.s_dim = 6
        self.N_u = 2
        self.pos_dim = 2


    @staticmethod
    def R(s):
        tilt = s[4]
        rot = np.array([
            [np.cos(tilt), -np.sin(tilt)],
            [np.sin(tilt),  np.cos(tilt)]])
        return rot

    # unforced dynamics (integrator, damping_deceleration)
    def F(self, s):
        xyz  = s[0:2]
        Vxyz = s[2:4]
        euler = s[4]
        omega = s[5]

        ###############################
        ## Linear position change
        dx = copy.deepcopy(Vxyz)

        ###############################
        ## Linear velocity change
        dV = -self.vel_damp * Vxyz + np.array([0, -GRAV])

        ###############################
        ## Euler angles change
        dE = copy.deepcopy(omega)

        ###############################
        ## Angular rate change
        omega_damp_quadratic = np.clip(self.damp_omega * omega ** 2, a_min=0.0, a_max=1.0)
        dOmega = - omega_damp_quadratic * omega

        # print(dx,dV,dE,dOmega)

        return np.concatenate([dx, dV, [dE], [dOmega]])


    # control affine dynamics (controlling acceleration only)
    # Output: [3,4]
    def G(self, s):
        xyz  = s[0:2]
        Vxyz = s[2:4]
        euler = s[4]
        omega = s[5]

        ###############################
        ## Rotation matrix
        rot = self.R(s)

        ###############################
        ## dx, dV, dE
        dx = self.zero2x2
        dV = (rot / self.mass) @ (self.Fmax * self.thrust_sum_mx)
        dE = self.zero1x2
        
        ###############################
        ## Angular acceleration
        # omega_damp_quadratic = np.clip(self.damp_omega * omega ** 2, a_min=0.0, a_max=1.0)
        dOmega = self.M_omega
        
        # print(dx.shape,dV.shape,dE.shape,dOmega.shape)
        return np.concatenate([dx, dV, dE, dOmega], axis=0)


"""
Scenario: guide point mass under gravity with a wall obstacle
Control cost is identity
Reward is distance from goal squared
State: [x, y, z, Vx, Vy, Vz]
"""
def mppi_test():
    grav_force=np.array([0.,0.,-GRAV])

    dynamics = AffineQuadrotorDynamics2D()

    goal = np.zeros(3)
    init = np.zeros(dynamics.s_dim)
    goal[0] = 1.
    init[0] = -1.
    init[4] = 45./180.*np.pi

    # arbitrary state-dependent cost, i.e.
    # no assumptions on the cost
    def q(x):
        pos = x[:s_dim]
        vel = x[s_dim:]
        delta = pos - goal
        dist2 = delta.T @ delta
        dist = np.sqrt(dist2)
        collision_wall = 100.0 * collision_check(x)

        soft_negativeness = 0.0* np.exp(-5* pos)
        collision_soft = np.prod(soft_negativeness)
        qq = dist2 + collision_wall + collision_soft
        return 1e0 * qq
        #return dist2# + 1 * vel.T @ vel

    # inverse variance of noise relative to control
    # if large, we generate small noise to system
    rho = 1000.
    # exploration weight. nu == 1.0 - no penalty for exploration
    # exploration cost = 0.5*(1-1/nu) * du^T * R * du
    nu = 1.0

    # temperature -
    # if large, we don't really care about reward that much
    # if small, we sharpen
    lamd = 4e0

    # integration step
    delta_t = 1.0 / 100.0

    # initial state
    x = copy.deepcopy(init)

    # time horizon
    N = 40

    N_u = dynamics.N_u #Fx,Fy,Fz
    u_min,u_max=-1,1
    # initial control sequence
    u_seq = np.zeros((N, N_u))

    # PSD quadratic form matrix of control cost
    #R = 1e-1 * np.eye(2)
    R = np.zeros((N_u,N_u))

    # number of trajectories to sample
    K = 100

    x_history = []

    np.random.seed(0)

    #import pdb; pdb.set_trace()

    np.seterr(all="raise")

    traj_fig_id = 1

    plot_figures = True
    plot_xyzlim = 2
    plot_every = 2
    if plot_figures:
        fig = plt.figure(traj_fig_id, figsize=(10, 10))
        ax = fig.add_subplot(111) 
        plt.show(block=False)

    t = 0
    while True:

        x_history.append(0 + x)

        u_seq, mppi_traj, traj_costs = mppi_step(dynamics.F, dynamics.G, q, R, rho, nu, lamd, delta_t, x, u_seq, K)

        ## Real-system trajectory prediction
        x_horizon = rollout_real(x, dynamics.F, dynamics.G, u_seq, delta_t, u_min=u_min, u_max=u_max)

        ## Real-system step
        u = u_seq[0,:]
        # u = np.array([1.,1.])
        u_clipped = np.clip(u, a_min=u_min, a_max=u_max)
        dx = dynamics.F(x) + dynamics.G(x) @ u_clipped
        x += delta_t * dx
        print("Thrust: ", dynamics.Fmax * dynamics.thrust_sum_mx @ u_clipped, "Fmax: ", dynamics.Fmax, "u:", u_clipped)

        x_past = np.stack(x_history)

        u_seq[:-1,:] = u_seq[1:,:]
        u_seq[-1,:] = 0.0


        # Plotting predicted trajectory
        if plot_figures and t % plot_every == 0:
            fig = plt.figure(traj_fig_id)
            plt.cla()
            # Plotting all sampled trajectories
            max_cost = np.max(traj_costs[:,0])
            min_cost = np.min(traj_costs[:,0])
            for i in range(K):
                # The higher the cost the thinner the line
                ax.plot(mppi_traj[i][:,0], mppi_traj[i][:,1], 
                    linewidth=(1 - traj_costs[i,0]/max_cost + EPS), 
                    color="grey")
            ax.set_title("Min/Max costs: %.3f / %.3f" % (min_cost,max_cost))

            # Plotting the optimal trajectory
            ax.set_xlim([-plot_xyzlim,plot_xyzlim])
            ax.set_ylim([-plot_xyzlim,plot_xyzlim])
            # ax.scatter(x[-3],x[-2],x[-1], s=25, c="g", marker="o")
            ## Plot goal
            ax.scatter(goal[0],goal[1], s=25, c="g", marker="o")
            ## Plot quadrotor
            draw_quad(ax, x, dynamics.R(x), scale=0.1)
            ax.plot(x_horizon[:,0], x_horizon[:,1])
            ## Plot taken trajectory
            ax.plot(x_past[:,0], x_past[:,1])

            # Drawing the wall
            draw_wall(ax=ax, x_range=[-w_x,w_x], y_range=[-w_y,w_y])

            plt.draw()
            plt.pause(0.05)

        # Breaking if collided
        if collision_check(x):
            print("collided with obstacle!")
            break

        # Breaking if we reached the goal
        if np.linalg.norm(x[:s_dim]) < 0.05 and np.linalg.norm(x[s_dim:(s_dim*2)]) < 0.1:
            print("Reached the Goal! t=", t)
            break
        # Breaking if the env is over
        # if done: 
        #     print("Timed out! t=", t)
        #     break
        t += 1

    plt.show()




mppi_test()
