#!/usr/bin/env python

"""
Model Predictive Path Integral: example of a guided point mass
See MPPI paper: https://arc.aiaa.org/doi/pdf/10.2514/1.G001921
Calculations are based on Section III.c that assumes a special form of dynamical systems:
dx = f(x,t)*dt + g(x,t)*(u + du)*dt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

EPS = 1e-6
s_dim =  3 #state dimensions

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
    expS = np.exp((-1.0/lamd) * S) # (K, N) = (traj_num,timelen)
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
    w_x = .25
    w_y = 1.
    w_z = .5
    wall_high = np.array([w_x, w_y, w_z])[:s_dim]
    wall_low = -wall_high
    return np.all(s[:s_dim] < wall_high) and np.all(s[:s_dim] > wall_low)
   

# draw rect
def draw_wall(ax, x_range, y_range, z_range):
    # TODO: refactor this to use an iterotor
    xx, yy = np.meshgrid(x_range, y_range)
    z_min = z_range[0] * np.ones_like(xx)
    z_max = z_range[1] * np.ones_like(xx)
    ax.plot_wireframe(xx, yy, z_min, color="r")
    ax.plot_surface(xx, yy, z_min, color="r", alpha=0.2)
    ax.plot_wireframe(xx, yy, z_max, color="r")
    ax.plot_surface(xx, yy, z_max, color="r", alpha=0.2)

    yy, zz = np.meshgrid(y_range, z_range)
    x_min = x_range[0] * np.ones_like(yy)
    x_max = x_range[1] * np.ones_like(yy)
    ax.plot_wireframe(x_min, yy, zz, color="r")
    ax.plot_surface(x_min, yy, zz, color="r", alpha=0.2)
    ax.plot_wireframe(x_max, yy, zz, color="r")
    ax.plot_surface(x_max, yy, zz, color="r", alpha=0.2)


    xx, zz = np.meshgrid(x_range, z_range)
    y_min = y_range[0] * np.ones_like(yy)
    y_max = y_range[1] * np.ones_like(yy)
    ax.plot_wireframe(xx, y_min, zz, color="r")
    ax.plot_surface(xx, y_min, zz, color="r", alpha=0.2)
    ax.plot_wireframe(xx, y_max, zz, color="r")
    ax.plot_surface(xx, y_max, zz, color="r", alpha=0.2)

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
        vel = x[s_dim:]
        return np.concatenate([vel, -0.02*vel])

    # control affine dynamics (controlling acceleration only)
    def G(x):
        return np.vstack([np.zeros((s_dim,s_dim)), np.eye(s_dim)])

    goal = np.array([1., 0., 0.])
    init = np.array([-1., 0., 0.])

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
    rho = 1e-1

    # PSD quadratic form matrix of control cost
    #R = 1e-1 * np.eye(2)
    R = np.zeros((s_dim,s_dim))

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
    x = np.concatenate([init, np.zeros(s_dim)])

    # time horizon
    N = 10

    N_u = 3 #Fx,Fy,Fz
    # initial control sequence
    u_seq = np.zeros((N, N_u))

    # number of trajectories to sample
    K = 100

    x_history = []

    np.random.seed(0)

    #import pdb; pdb.set_trace()

    np.seterr(all="raise")

    traj_fig_id = 1

    plot_figures = True
    plot_xyzlim = 2
    plot_every = 5
    if plot_figures:
        fig = plt.figure(traj_fig_id)
        ax = fig.add_subplot(111, projection='3d') 
        plt.show(block=False)

    t = 0
    while True:

        x_history.append(0 + x)

        u_seq, mppi_traj, traj_costs = mppi_step(f, G, q, R, rho, nu, lamd, delta_t, x, u_seq, K)

        x_horizon = rollout(x, f, G, u_seq, delta_t)

        u = u_seq[0,:]
        #u[u < -1] = -1
        #u[u > 1] = 1
        dx = f(x) + G(x) @ u
        x += delta_t * dx

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
                ax.plot(mppi_traj[i][:,0], mppi_traj[i][:,1], mppi_traj[i][:,2], 
                    linewidth=(1 - traj_costs[i,0]/max_cost + EPS), 
                    color="grey")
            ax.set_title("Min/Max costs: %.3f / %.3f" % (min_cost,max_cost))

            # Plotting the optimal trajectory
            ax.set_xlim([-plot_xyzlim,plot_xyzlim])
            ax.set_ylim([-plot_xyzlim,plot_xyzlim])
            ax.set_zlim([0,2*plot_xyzlim])
            # ax.scatter(x[-3],x[-2],x[-1], s=25, c="g", marker="o")
            ## Plot goal
            ax.scatter(goal[0],goal[1],goal[2], s=25, c="g", marker="o")
            ## Plot traj prediction
            ax.scatter([x[0]],[x[1]],[x[2]], s=25, c="r", marker="^")
            ax.plot(x_horizon[:,0], x_horizon[:,1], x_horizon[:,2])
            ## Plot taken trajectory
            ax.plot(x_past[:,0], x_past[:,1], x_past[:,2])

            # Drawing the wall
            draw_wall(ax=ax, x_range=[-w_x,w_x], y_range=[-w_y,w_y], z_range=[-w_z,w_z])

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
