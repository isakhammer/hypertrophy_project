import numpy as np
from src import model

class Particle:
    best = None
    best_value = None
    pos = None
    vel = None
    particle_id = None


def particle_swarm_optimization(f0: np.ndarray,
                                f_d: np.ndarray,
                                f_max: np.ndarray,
                                pars: dict):

    # options
    n = 100
    sr_max = 30
    N_ex = 4
    n_particles = 10

    c_vel = 0.1
    c_best = 0.5
    c_global  = 0.2

    # general options
    wo_opt = pars["wo_opt"]
    N_t = int(wo_opt["t_horizon"]/wo_opt["t_freq"])
    t = np.linspace(0, wo_opt["t_horizon"], N_t)

    # initalize particles
    particles = []
    for i in range(n_particles):
        p = Particle()
        p.pos = np.random.uniform(0 , sr_max, N_t*N_ex)
        p.best = p.pos
        p.best_value = np.inf
        p.vel = p.pos/10
        p.particle_id = i
        particles.append(p)

    # inital sr_log
    sr_log = np.zeros((N_t, N_ex + 1))
    sr_log[:, 0] = t

    # initial global values
    global_best = np.zeros(N_t*N_ex)
    global_best_value = np.inf
    for i in range(n):

        for p in particles:
            sr_log[:, 1:] = p.pos.reshape(N_t, N_ex)
            sr_mg, f, f_avg= model.compute_model(   sr_log=sr_log,
                                                    f0=None,
                                                    pars=pars)
            value =  np.linalg.norm(f_avg[:,1:] - f_d)

            if value < p.best_value:
                p.best_value = value
                p.best = p.pos
            if value < global_best_value:
                global_best_value = value
                global_best = p.pos

        # iterate particles
        for p in particles:
            p.vel = c_vel*p.vel + c_best*(p.best - p.pos) + c_global*(global_best - p.pos)
            p.pos += p.vel

        if (i%10 == 0):
            print(i, global_best_value)


    sr_log[:, 1:] = global_best.reshape(N_t, N_ex)

    return sr_log
