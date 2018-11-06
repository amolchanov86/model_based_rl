#!/usr/bin/env python

import os, sys
import argparse
import logging
import numpy as np
from numpy.linalg import norm
from copy import deepcopy
import operator
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from mpl_toolkits.mplot3d import Axes3D

import gym
from gym import spaces
from gym.utils import seeding
import gym.envs.registration as gym_reg

import gym_art.quadrotor.rendering3d as r3d
from gym_art.quadrotor.quadrotor_control import *
from gym_art.quadrotor.quadrotor_modular import *

#!/usr/bin/env python

"""
Model Predictive Path Integral: example of a guided point mass
See MPPI paper: https://arc.aiaa.org/doi/pdf/10.2514/1.G001921
Calculations are based on Section III.c that assumes a special form of dynamical systems:
dx = f(x,t)*dt + g(x,t)*(u + du)*dt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

GRAV = 9.81 #m/s^2

## TODO:
# - quadrotor parameters should be synchronized (including time sted delta_t)
# - put functions f() and G() inside the env
# - re-normalize predictions
# - normalize actions


def rollout(x_0, f_fn, G_fn, u_seq, delta_t):
    N = u_seq.shape[0] # u_seq.shape = (N,u_dim), where N = timesteps
    x_dim = x_0.shape[0]
    x_out = np.zeros((N, x_dim)) #trajectory, i.e. (timesteps, x_dim)
    x = 0 + x_0 #???
    for i, u in enumerate(u_seq):
        x_out[i] = x
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
    cost2go = Qs[::-1].cumsum()[::-1]
    # cost2go = Qs.cumsum()
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
    for i in range(K):
        u_perturbed = u_seq + dus[i]
        x_seq = rollout(x_0, f_fn, G_fn, u_perturbed, delta_t)
        #Computing costs-to-go for each trajectory
        S[i] = S_tilde(q_tilde_fn, x_seq, u_seq, dus[i]) #S.shape == (K,N)

    # Computing weights for trajectories
    expS = np.exp((-1.0/lamd) * S) # (K, N) = (traj_num,timelen)
    denom = np.sum(expS, axis=0) # (N) = (timelen)
    
    # Weighting trajectories to find control updates
    du_weighted = expS[:,:,None] * dus # (K, N, udim)
    u_change_unscaled = np.sum(du_weighted, axis=0) # (N, udim): averaging among traj.
    u_change = u_change_unscaled / denom[:,None] # (N, udim)

    return u_seq + u_change



#######################################
## QUADROTOR DYNAMICS
# assuming state [xyz, V, R, Omeage, xyz_goal]

# unforced dynamics (integrator, damping_deceleration)
def f(s, dt=0.01):
    xyz  = s[0:3]
    Vxyz = s[3:6]
    rot = s[6:15].reshape([3,3])
    omega = s[15:18]
    goal = s[18:21]

    mass = 0.5
    arm_length = 0.33 / 2.0
    inertia = mass * npa(0.01, 0.01, 0.02)
    thrust_to_weight = 2.0

    ###############################
    ## Linear position change
    dx = copy.deepcopy(Vxyz)

    ###############################
    ## Linear velocity change
    vel_damp = 0.999
    dV = vel_damp / dt * Vxyz + np.array([0, 0, -GRAV])

    ###############################
    ## Angular orientation change
    damp_omega = 0.015

    omega_vec = np.matmul(rot, omega) # Change from body2world frame
    wx, wy, wz = omega_vec
    omega_mat_deriv = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

    # ROtation matrix derivative
    dR = np.matmul(omega_mat_deriv, rot).flatten()

    ###############################
    ## Angular rate change
    C_omega = (1.0 / inertia) * (cross(-omega, inertia * omega))
    omega_damp_quadratic = np.clip(damp_omega * omega ** 2, a_min=0.0, a_max=1.0)
    dOmega = (1.0 - omega_damp_quadratic) * C_omega

    ###############################
    ## Goal change
    dgoal = np.zeros_like(goal)

    return np.concatenate([dx, dV, dR, dOmega, dgoal])


# control affine dynamics (controlling acceleration only)
def G(s):
    xyz  = s[0:3]
    Vxyz = s[3:6]
    rot = s[6:15].reshape([3,3])
    omega = s[15:18]
    goal = s[18:21]

    damp_omega = 0.015
    mass = 0.5
    arm_length = 0.33 / 2.0
    inertia = mass * npa(0.01, 0.01, 0.02) # [3,1]
    thrust_to_weight = 2.0
    thrust_max = GRAV * mass * thrust_to_weight / 4.0
    torque_to_thrust=0.05
    torque_max = torque_to_thrust * thrust_max 
    
    ###############################
    # Auxiliarry matrices
    thrust_sum_mx = np.zeros([3,4]) # [0,0,F_sum].T
    thrust_sum_mx[2,:] = 1# [0,0,F_sum].T
    scl = arm_length / norm([1.,1.,0.])
    # Unscaled (normalized) propeller positions
    prop_pos = scl * np.array([
        [1.,  1., -1., -1.],
        [1., -1., -1.,  1.],
        [0.,  0.,  0.,  0.]]).T # row-wise easier with np
    # unit: meters^2 ??? maybe wrong
    prop_crossproducts = np.cross(prop_pos, [0., 0., 1.]).T
    # 1 for props turning CCW, -1 for CW
    prop_ccw = np.array([1., -1., 1., -1.]) # Rotations directions
    prop_ccw_mx = np.zeros([3,4])[2,:] = prop_ccw # Matrix allows using matrix multiplication 

    ###############################
    ## dx, dV, dR, dgoal
    dx = np.zeros([3,4])
    dV = (rot / mass ) @ (thrust_max * thrust_sum_mx)
    dR = np.zeros([9,4])
    dgoal = np.zeros([3,4])
    
    ###############################
    ## Angular acceleration
    #Prop crossproduct give torque directions
    C_thrust = thrust_max * prop_crossproducts # [3,4] @ [4,1]

    # additional torques along z-axis caused by propeller rotations
    C_prop = torque_max * prop_ccw_mx  # [3,4] @ [4,1] = [3,1]

    omega_damp_quadratic = np.clip(damp_omega * omega ** 2, a_min=0.0, a_max=1.0)
    dOmega = ((1.0 - omega_damp_quadratic) * (1.0 / inertia))[:,None] * (C_thrust + C_prop)
    
    return np.concatenate([dx, dV, dR, dOmega, dgoal], axis=0)


"""
Scenario: 3D quadrotor control
Initial position randomized
Controls are in the range [0 .. 1]
Goal position is (0, 0)
Control cost is identity
Reward is distance from goal squared
State: [x, y, dx, dy]
"""

def dynamics_test():

    env = QuadrotorEnv(raw_control=False, raw_control_zero_middle=False, dim_mode='3D', tf_control=False, sim_steps=1)
    s = env.reset()

    # integration step
    delta_t = env.dt

    # time horizon
    ep_len = env.ep_len
    u_seq = np.zeros([ep_len, env.action_space.shape[0]])

    np.random.seed(0)

    np.seterr(all="raise")

    s_real = []

    render = True
    render_each = 1

    traj_fig_id = 1
    plot_figures = True
    fig_lim = 5
    if plot_figures:
        fig = plt.figure(traj_fig_id)
        ax = fig.add_subplot(111, projection='3d') 

        

    t = 0
    while True:
        s_real.append(s)
        if render and (t % render_each == 0): env.render()

        # Running env step
        s, r, done, info = env.step(np.array([0.,0.,0.,0.]))
        # Computing control sequence
        # print("action: ", env.controller.action, "u_seq: ", u_seq.shape)

        # Breaking if the env is over
        if done: break
        u_seq[t,:] = env.controller.action
        t += 1


    # Running prediction
    s_pred = rollout(s_real[0], f, G, u_seq, delta_t)
    s_real = np.array(s_real)


    # Plotting trajectories
    if plot_figures:
        fig = plt.figure(traj_fig_id)
        plt.cla()
        plt.scatter([0],[0],[0], c="g", marker="o")
        plt.plot(s_real[:,0], s_real[:,1], s_real[:,2], "g")
        plt.plot(s_pred[:,0], s_pred[:,1], s_pred[:,2], "r")
        ax.set_xlim([-fig_lim,fig_lim])
        ax.set_ylim([-fig_lim,fig_lim])
        ax.set_zlim([-fig_lim,fig_lim])
        plt.pause(0.05)
        plt.show()



def mppi_test():

    # arbitrary state-dependent cost, i.e.
    # no assumptions on the cost
    def q(x):
        xyz = x[0:3]
        goal = x[-3:]
        dist_cost = np.linalg.norm(xyz - goal)
        return dist_cost 

    # inverse variance of noise relative to control
    # if large, we generate small noise to system
    # rho = 1e-1
    rho = 1.

    # PSD quadratic form matrix of control cost
    #R = 1e-1 * np.eye(2)
    R = np.zeros((4,4))

    # exploration weight. nu == 1.0 - no penalty for exploration
    # exploration cost = 0.5*(1-1/nu) * du^T * R * du
    nu = 1.0

    # temperature -
    # if large, we don't really care about reward that much
    # if small, we sharpen
    lamd = 4e0

    # integration step
    delta_t = 1.0 / 100.0

    # time horizon
    N = 20

    # initial control sequence
    u_seq = 0.5 * np.ones((N, 4))

    # number of trajectories to sample
    K = 100

    np.random.seed(0)

    #import pdb; pdb.set_trace()

    np.seterr(all="raise")

    env = QuadrotorEnv(raw_control=True, raw_control_zero_middle=False, dim_mode='3D', tf_control=False, sim_steps=1)
    env.max_episode_steps = 10000
    s = env.reset()
    s_history = []
    t = 0
    render = True
    render_each = 1
    traj_fig_id = 1

    plot_figures = True
    if plot_figures:
        fig = plt.figure(traj_fig_id)
        ax = fig.add_subplot(111, projection='3d') 
        plt.show(block=False)

    while True:
        s_history.append(s)
        if render and (t % render_each == 0): env.render()

        # Computing control sequence
        u_seq = mppi_step(f, G, q, R, rho, nu, lamd, delta_t, s, u_seq, K)

        # Running env step
        s, r, done, info = env.step(u_seq[0,:])

        # Running prediction
        s_horizon = rollout(s, f, G, u_seq, delta_t)

        # Plotting predicted trajectory
        if plot_figures:
            fig = plt.figure(traj_fig_id)
            plt.cla()
            ax.set_xlim([-10,10])
            ax.set_ylim([-10,10])
            ax.set_zlim([-10,10])
            plt.scatter([0],[0],[0], c="g", marker="o")
            plt.scatter([s[0]],[s[1]],[s[2]], c="r", marker="^")
            plt.plot(s_horizon[:,0], s_horizon[:,1], s_horizon[:,2])

            plt.draw()
            plt.pause(0.05)

        # Breaking if we reached the goal
        if np.linalg.norm(s[:3]) < 0.05 and np.linalg.norm(s[3:6]) < 0.1:
            print("Reached the Goal! t=", t)
            break
        # Breaking if the env is over
        if done: 
            print("Timed out! t=", t)
            break
        t += 1


def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m',"--mode",
        type=int,
        default=1,
        help="Test mode: "
             "0 - MPPI"
             # "1 - Test dynamics model"
    )
    args = parser.parse_args()

    if args.mode == 0:
        mppi_test()
    if args.mode == 1:
        dynamics_test()

if __name__ == '__main__':
    main(sys.argv)
