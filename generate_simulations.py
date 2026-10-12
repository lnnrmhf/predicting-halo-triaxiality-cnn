import os
import argparse
import sys
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
from jax import config
jax.config.update("jax_threefry_partitionable", True)
import numpy as np
from discodj import DiscoDJ



def generate_samples(n_samples, boxsize, res, start_idx, dim, name, device, precision, cosmo):
    #base_dir = "/home/lena/datascience/DataScience/notebooks/Final/simulations_test"
    base_dir = "../../../../../../mnt/data/lena/simulations_test"

    for i in range(start_idx, start_idx + n_samples):
        print(f"Generating sample {i}...")

        sim_folder = os.path.join(base_dir, f"sim_{i}")
        os.makedirs(sim_folder, exist_ok=True)

        #define DISCO-DJ object
        dj = DiscoDJ(dim=dim, res=res, name=name, device=device, precision=precision, boxsize=boxsize, cosmo=cosmo)

        #simulation set up
        stepper = "bullfrog"
        method = "pm"
        res_pm = 4 * dj.res
        time_var = "D"
        antialias = 0
        grad_kernel_order = 4
        laplace_kernel_order = 0
        worder = 2
        n_resample = 1
        deconvolve = False
        nlpt_order_ics = 2
        chunk_size = res ** dim // 16   #mass assignment in smaller chunks due to memory issues 

        numsteps = 100  # number of steps to be performed

        a_ini = 0.02  # initial scale factor of the simulation (where it is initialized with LPT)
        a_end = 1.0  # final scale factor of the simulation
        dj = dj.with_timetables()
        dj = dj.with_linear_ps(transfer_function="Eisenstein-Hu")
        white_noise_field = dj.get_ngenic_noise(seed=i * 1000 + 42)
        dj = dj.with_ics(white_noise_space="real", white_noise_field=white_noise_field, convert_to_numpy=False)

        #save initial density field 
        delta_ini = np.array(dj.delta_ini).copy()
        density_path = os.path.join(sim_folder, f"density_{i}.npy")
        np.save(density_path, delta_ini)

        print(f"Sim {i}: white noise std = {np.array(white_noise_field).std():.4f}")

        n_order = 2
        dj = dj.with_lpt(n_order=n_order, convert_to_numpy=False)

        #run n-body simulation 
        X_sim, P_sim, a_sim = dj.run_nbody(a_ini=a_ini, a_end=a_end, n_steps=numsteps, res_pm=res_pm,
                                        time_var=time_var, stepper=stepper, method=method, antialias=antialias,
                                        grad_kernel_order=grad_kernel_order, laplace_kernel_order=laplace_kernel_order,
                                        nlpt_order_ics=nlpt_order_ics, n_resample=n_resample,
                                        deconvolve=deconvolve, return_displacement=False, chunk_size=chunk_size,
                                        convert_to_numpy=False)  # convert_to_numpy: same as above for the ICs and LPT

        #save the positions, velocities and IDs of the particles in an ascii file to be processed by ROCKSTAR 
        pos = np.array(X_sim).reshape(-1, 3)
        vel = np.array(P_sim).reshape(-1, 3)
        N = len(pos)
        ids = np.arange(N, dtype=np.float32).reshape(-1, 1)
        data = np.column_stack([pos, vel, ids])

        ascii_path = os.path.join(sim_folder, f"particles_{i}.ascii")
        with open(ascii_path, "w") as f:
            f.write(f"# Number of Particles: {N}\n") 
            np.savetxt(f, data, fmt=["%.6f","%.6f","%.6f","%.6f","%.6f","%.6f","%d"])

        position_path = os.path.join(sim_folder, f"particles_xsim_{i}.npy")
        np.save(position_path, np.array(X_sim))
        
        print(f"Saved sample {i} to {base_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulations using DiscoDJ")
    
    parser.add_argument("--n_samples", type=int, required=True,
                        help="Number of simulations")
    parser.add_argument("--res", type=int, required=True,                        
                        help="Particle resolution")
    parser.add_argument("--boxsize", type=float, required=True,
                        help="Boxsize in Mpc/h")
    parser.add_argument("--start_idx", type=int, required=True,
                        help="Index to start generating simulations from") 
    
    args = parser.parse_args()

    #detect if a GPU is available
    devices = jax.devices()
    device = "gpu" if np.any([d.platform == "gpu" for d in devices]) else "cpu"
    #set the parameters
    name = "sample_generation"
    dim = 3

    precision = "single"

    cosmo = dict(Omega_c=0.259605,  # cold dark matter content
             Omega_b=0.0488911,  # baryonic content (note: Disco-DJ so far only performs N-body simulations, no hydro; this is only used for the linear power spectrum!)
             h=0.67742,  # dimensionless Hubble constant
             n_s=0.96822,  # scalar spectral index
             sigma8=0.808992  # amplitude of matter density fluctuations at a scale of 8 Mpc/h
             )
    
    generate_samples(args.n_samples, args.boxsize, args.res, args.start_idx, dim, name, device, precision, cosmo)
    print("Done.")

if __name__ == "__main__":
    main()