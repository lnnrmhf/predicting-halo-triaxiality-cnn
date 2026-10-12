import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches
import argparse
import sys


def generate_labels(i, res, boxsize):

    print(f"Generating labels for simulation {i}...")

    home_dir = "/home/lena/datascience/DataScience/notebooks/Final"
    data_dir = "../../../../../../mnt/data/lena"
    sim_folder = os.path.join(data_dir, f"simulations_test/sim_{i}")
    halo_folder = os.path.join(home_dir, f"rockstar_halos")

    #read in all the data 
    particle_path = os.path.join(sim_folder, f"particles_xsim_{i}.npy")
    particle_pos = np.load(particle_path)
    particle_pos = particle_pos.reshape(-1, 3)

    pos_x = particle_pos[:, 0]
    pos_y = particle_pos[:, 1]
    pos_z = particle_pos[:, 2]

    halo_path = os.path.join(halo_folder, f"halos_{i}.0.ascii")
    halos = pd.read_csv(halo_path, comment="#", sep=r"\s+", header=None)

    halo_mvir = halos.iloc[:,2].values
    halo_rvir_mpc = halos.iloc[:, 4].values / 1000.0 
    halo_x = halos.iloc[:, 8].values
    halo_y = halos.iloc[:, 9].values
    halo_z = halos.iloc[:, 10].values
    halo_b_a = halos.iloc[:, 33].values
    halo_c_a = halos.iloc[:, 34].values
    parent_id = halos.iloc[:, 49].values

    particle_table_path = os.path.join(halo_folder, f"halos_{i}.0.particles")
    particle_table = pd.read_csv(particle_table_path,
                        comment="#", sep=r"\s+", header=None,
                        names=["x","y","z","vx","vy","vz", "particle_id","assigned_halo","internal_halo","external_halo"])

    # Eulerian particle positions
    pos_euler = np.column_stack([pos_x, pos_y, pos_z])  

    # Lagrangian particle positions
    coords = np.linspace(0, boxsize, res, endpoint=False)
    qx, qy, qz = np.meshgrid(coords, coords, coords, indexing='ij')
    pos_lagrange = np.column_stack([qx.ravel(), qy.ravel(), qz.ravel()])

    ix = (pos_lagrange[:, 0] / boxsize * res).astype(int) % res
    iy = (pos_lagrange[:, 1] / boxsize * res).astype(int) % res
    iz = (pos_lagrange[:, 2] / boxsize * res).astype(int) % res

    #define halos of interest: mass threshold and no subhalos -> only host halos of a certain mass and above 
    mass_threshold = 4e12
    use_halo = (parent_id == -1) & (halo_mvir > mass_threshold) 

    #filter to host halos only
    halo_x_host = halo_x[use_halo]
    halo_y_host = halo_y[use_halo]
    halo_z_host  = halo_z[use_halo]
    halo_rvir_host = halo_rvir_mpc[use_halo]
    halo_b_a_host = halo_b_a[use_halo]
    halo_c_a_host = halo_c_a[use_halo]

    halo_pos_host = np.column_stack([halo_x_host, halo_y_host, halo_z_host])

    #mapping of external ID to axis ratios
    external_to_ba = dict(zip(range(len(halos)), halo_b_a))
    external_to_ca = dict(zip(range(len(halos)), halo_c_a))
    #host halo ids
    host_halo_ids = set(np.where(use_halo)[0])
    #particles in host halos 
    host_particles = particle_table[particle_table["assigned_halo"] == particle_table["internal_halo"]]

    #assign particles using Rockstar's FOF ids
    particle_halo = np.full(len(pos_euler), -1, dtype=int)
    lagrange_ba_grid  = np.full((res, res, res), np.nan, dtype=np.float32)
    lagrange_ca_grid  = np.full((res, res, res), np.nan, dtype=np.float32)

    #map particle_id -> halo index in host catalog
    for row in range(len(host_particles)):
        pid = int(host_particles["particle_id"].iloc[row])
        ext_id = host_particles["external_halo"].iloc[row]
        
        if ext_id == -1 or ext_id not in host_halo_ids:
            continue

        particle_halo[pid] = ext_id
        lagrange_ba_grid[ix[pid], iy[pid], iz[pid]] = external_to_ba[ext_id]
        lagrange_ca_grid[ix[pid], iy[pid], iz[pid]] = external_to_ca[ext_id]

    #use virial radius kNN for remaining unassigned particles
    unassigned_mask = particle_halo == -1
    unassigned_pos = pos_euler[unassigned_mask]
    unassigned_indices = np.where(unassigned_mask)[0]

    tree = cKDTree(halo_pos_host)
    k = 5
    distances, halo_indices_k = tree.query(unassigned_pos, k=k)
    rvir_candidates = halo_rvir_host[halo_indices_k]
    within_rvir = distances < rvir_candidates

    for j in range(k):
        #only update still unassigned particles
        still_unassigned = particle_halo[unassigned_indices] == -1
        within = within_rvir[:, j]
        update = still_unassigned & within

        update_indices = unassigned_indices[update]
        #assign halo
        assigned_halo_idx = halo_indices_k[update, j]
        particle_halo[unassigned_indices[update]] =assigned_halo_idx

        #assign axis ratio values
        lagrange_ba_grid[ix[update_indices], iy[update_indices], iz[update_indices]] = halo_b_a_host[assigned_halo_idx]
        lagrange_ca_grid[ix[update_indices], iy[update_indices], iz[update_indices]] = halo_c_a_host[assigned_halo_idx]

    #save labels
    label_ba_path =  os.path.join(sim_folder, f"label_ba_{i}.npy")
    np.save(label_ba_path, lagrange_ba_grid)

    label_ca_path =  os.path.join(sim_folder, f"label_ca_{i}.npy")
    np.save(label_ca_path, lagrange_ca_grid)



def main():
    parser = argparse.ArgumentParser(
        description="Generate labels for the particles")
    
    parser.add_argument("--simulation_number", type=int, required=True,                        
                        help="Number of simulation to be processed")
    parser.add_argument("--res", type=int, required=True,                        
                        help="Particle resolution")
    parser.add_argument("--boxsize", type=float, required=True,
                        help="Boxsize in Mpc/h")
    
    args = parser.parse_args()

    generate_labels(args.simulation_number, args.res, args.boxsize)
    print("Done.")

if __name__ == "__main__":
    main()