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


"""
Scenario: guide point mass around corner.
Initial position is (1, -1)
Goal position is (-1, 1)
Negative quadrant is obstacle (x < 0 && y < 0)
Control cost is identity
Reward is distance from goal squared
State: [x, y, dx, dy]
"""
def mppi_test():

    # unforced dynamics (integrator, damping_deceleration)
    def f(x):
        vel = x[2:]
        return np.concatenate([vel, -0.02*vel])

    # control affine dynamics (controlling acceleration only)
    def G(x):
        return np.vstack([np.zeros((2,2)), np.eye(2)])

    goal = np.array([-1, 0.2])
    init = np.array([0.2, -1])

    # arbitrary state-dependent cost, i.e.
    # no assumptions on the cost
    def q(x):
        pos = x[:2]
        vel = x[2:]
        delta = pos - goal
        dist2 = delta.T @ delta
        dist = np.sqrt(dist2)
        collision_wall = 100.0 * np.all(pos < 0)
        soft_negativeness = 0.0* np.exp(-5* pos)
        collision_soft = np.prod(soft_negativeness)
        qq = dist2 + collision_wall + collision_soft
        return 1e0 * qq
        #return dist2# + 1 * vel.T @ vel

    # inverse variance of noise relative to control
    # if large, we generate small noise to system
    rho = 1e-1

    # PSD quadratic form matrix of control cost
    #R = 1e-1 * np.eye(2)
    R = np.zeros((2,2))

    # exploration weight. nu == 1.0 - no penalty for exploration
    # exploration cost = 0.5*(1-1/nu) * du^T * R * du
    nu = 1.0

    # temperature -
    # if large, we don't really care about reward that much
    # if small, we sharpen
    lamd = 4e0

    # integration step
    delta_t = 1.0 / 20.0

    # initial state
    x = np.concatenate([init, np.zeros(2)])

    # time horizon
    N = 20

    # initial control sequence
    u_seq = np.zeros((N, 2))
    u_seq[:,0] = -1.0
    u_seq[:,1] = 1.0

    # number of trajectories to sample
    K = 100

    x_history = []

    np.random.seed(0)

    #import pdb; pdb.set_trace()

    np.seterr(all="raise")

    while True:

        x_history.append(0 + x)

        if q(x) < 0.01 and np.linalg.norm(x[2:]) < 0.1:
            print("reached goal!")
            break

        if np.all(x[:2] < 0):
            print("collided with obstacle!")
            break

        u_seq = mppi_step(f, G, q, R, rho, nu, lamd, delta_t, x, u_seq, K)

        x_horizon = rollout(x, f, G, u_seq, delta_t)

        u = u_seq[0,:]
        #u[u < -1] = -1
        #u[u > 1] = 1
        dx = f(x) + G(x) @ u
        x += delta_t * dx

        x_past = np.stack(x_history)

        u_seq[:-1,:] = u_seq[1:,:]
        u_seq[-1,:] = 0.0

        plt.clf()
        plt.hold(True)
        rect = patches.Rectangle((-2, -2), 2, 2, facecolor=(0, 0, 0))
        plt.gca().add_patch(rect)
        plt.plot(x_past[:,0], x_past[:,1], color=(.8, .2, .2), linestyle='-.')
        plt.plot(x_horizon[:,0], x_horizon[:,1], color=(0, 0, 1), linestyle='-.', linewidth=3)
        plt.axis("equal")
        #plt.xlim([-2, 2])
        #plt.ylim([-2, 2])

        plt.show(block=False)
        plt.pause(0.01)

    plt.show()




mppi_test()
