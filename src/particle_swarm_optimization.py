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

    """
    Input:
        f0_d:       [sr_ex0, sr_ex1, ... ]          (N_ex, )
        f_d:        [f_d_mg0, f_d_mg1, ... ]        (N_mg, )
        f_max:      [f_d_mg0, f_d_mg1, ... ]        (N_mg, )
        pars: dict of all parameters
    Out:
        sr_ex_log:  [t, sr_ex0, sr_ex1, ... ]       (N_t, N_ex + 1)
    """

    # options
    n = 50
    sr_max = 20
    N_ex = len(pars["ex_names"])
    n_particles = 30

    c_vel = 0.02
    c_noise = 0.1
    c_best = 1.6
    c_global = 0.2

    # general options
    wo_opt = pars["wo_opt"]
    N_t = int(wo_opt["t_horizon"]/wo_opt["t_freq"])
    t = np.linspace(0, wo_opt["t_horizon"], N_t)

    # initalize particles
    particles = []
    for i in range(n_particles):
        p = Particle()
        p.pos = np.random.uniform(0 , sr_max, N_t*N_ex)
        p.vel = np.random.uniform(-1 , 1, N_t*N_ex)
        p.best_value = np.inf
        p.vel = p.pos/10
        p.particle_id = i
        particles.append(p)

    # initalize array of stimulated exercise log
    sr_ex_log = np.zeros((N_t, N_ex + 1))
    sr_ex_log[:, 0] = t

    # initial global values
    global_best = np.zeros(N_t*N_ex)
    global_best_value = np.inf
    for i in range(n):

        for p in particles:
            sr_ex_log[:, 1:] = p.pos.reshape(N_t, N_ex)
            sr_mg, f, f_avg= model.compute_model(   sr_ex_log=sr_ex_log,
                                                    f0=f0,
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
            p.vel = c_vel*p.vel + c_best*(p.best - p.pos) + c_global*(global_best - p.pos) + np.random.normal( 0, c_noise, N_t*N_ex)
            p.pos += p.vel
            if p.particle_id == 0:
                print("vel ", np.linalg.norm(p.vel)," best: " , p.best_value," noise: ", np.random.normal( 0, c_noise, 1))


        if (i%10 == 0):
            print("\nglobal ", i, global_best_value)


    sr_ex_log[:, 1:] = global_best.reshape(N_t, N_ex)

    return sr_ex_log
