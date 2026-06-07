import numpy as np
import h5py
from scipy.interpolate import CubicSpline
from numpy.polynomial.legendre import leggauss
from datetime import datetime

# ==========================================
# Parameters & Grid Setup
# ==========================================
Na = 33
Nb = 33

log_a_list = np.linspace(-1.0, 1.0, Na)
log_b_list = np.linspace(-1.0, 1.0, Nb)

# Resolution of the search grids (viewing angles)
Nt_sparse, Np_sparse = 17, 17
Nt_fine, Np_fine = 33, 33

# Resolution for surface integration 
# (reduced to 128x128 during sparse search for significant speedup)
N_THETA_sparse, N_PHI_sparse = 128, 128
N_THETA_fine, N_PHI_fine = 512, 512

# Dynamically calculate the search interval for the fine step
# (We search within +/- one sparse step around the coarse peak to guarantee finding the true maximum)
theta_sparse_step = (np.pi / 2) / (Nt_sparse - 1)
phi_sparse_step   = (np.pi / 2) / (Np_sparse - 1)

# ==========================================
# Interpolation Tables (External Data)
# ==========================================
try:
    table_data = np.loadtxt("chandrasekhar_1960.csv", delimiter=",", skiprows=1)
except FileNotFoundError:
    raise FileNotFoundError("Please ensure 'chandrasekhar_1960.csv' is in the same directory.")

mu_table = table_data[:, 0]
I_l_table = table_data[:, 1]
I_r_table = table_data[:, 2]

I_l_spline = CubicSpline(mu_table, I_l_table)
I_r_spline = CubicSpline(mu_table, I_r_table)

# ==========================================
# Helper Functions for Integration Grid
# ==========================================

def setup_integration_grid(N_T, N_P):
    """Generate a spherical coordinate grid for surface integration."""
    dPhi = 2 * np.pi / N_P
    u, w_Theta = leggauss(N_T)
    Theta_list = np.arccos(u)
    Phi_list = (np.arange(N_P) + 0.5) * dPhi
    Theta_grid, Phi_grid = np.meshgrid(Theta_list, Phi_list, indexing='ij')

    grid = {
        'dPhi': dPhi,
        'w_Theta_array': np.tile(w_Theta[:, np.newaxis], (1, N_P)),
        'sin_Theta': np.sin(Theta_grid),
        'cos_Theta': np.cos(Theta_grid),
        'sin_Phi': np.sin(Phi_grid),
        'cos_Phi': np.cos(Phi_grid),
    }
    
    # Cartesian coordinates on the unit sphere
    grid['x_s'] = grid['sin_Theta'] * grid['cos_Phi']
    grid['y_s'] = grid['sin_Theta'] * grid['sin_Phi']
    grid['z_s'] = grid['cos_Theta']
    return grid

def compute_ellipsoid_geometry(a, b, grid):
    """Calculate normal vectors and inverse squared distances."""
    nx = grid['sin_Theta'] * grid['cos_Phi'] / a**2
    ny = grid['sin_Theta'] * grid['sin_Phi'] / b**2
    nz = grid['cos_Theta']
    n_array = np.stack([nx, ny, nz], axis=0)
    
    norm = np.linalg.norm(n_array, axis=0, keepdims=True)
    epsilon = 1e-12
    n_array = n_array / np.maximum(norm, epsilon)
    
    s = grid['sin_Theta'] * grid['cos_Phi'] / a
    t = grid['sin_Theta'] * grid['sin_Phi'] / b
    r2_array = 1.0 / (s**2 + t**2 + grid['cos_Theta']**2)
    
    return n_array, r2_array

# ==========================================
# Core Physics Functions
# ==========================================
def I_prime(mu): return I_l_spline(mu) + I_r_spline(mu)
def Q_prime(mu, cos_2psi): return (I_l_spline(mu) - I_r_spline(mu)) * cos_2psi
def U_prime(mu, sin_2psi): return (I_l_spline(mu) - I_r_spline(mu)) * sin_2psi

def get_x_prime(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
def get_y_prime(phi):
    return np.array([-np.sin(phi), np.cos(phi), 0.0])
def get_z_prime(theta, phi):
    return np.array([-np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)])

def integrate_all_parameters(a, b, theta, phi, n_array, r2_array, grid):
    """Perform surface integration to obtain Stokes parameters."""
    x_prime = get_x_prime(theta, phi)
    y_prime = get_y_prime(phi)
    z_prime = get_z_prime(theta, phi)
    
    mu_array = x_prime[0]*n_array[0] + x_prime[1]*n_array[1] + x_prime[2]*n_array[2]
    mu_masked = np.maximum(mu_array, 0.0)
    
    nr_array = n_array[0]*grid['x_s'] + n_array[1]*grid['y_s'] + n_array[2]*grid['z_s']
    w_array = r2_array / np.maximum(np.abs(nr_array), 1e-12)
    w_p_array = mu_masked * w_array
    
    ny_prime = y_prime[0]*n_array[0] + y_prime[1]*n_array[1] + y_prime[2]*n_array[2]
    nz_prime = z_prime[0]*n_array[0] + z_prime[1]*n_array[1] + z_prime[2]*n_array[2]
    den = nz_prime**2 + ny_prime**2
    mask = (den > 1e-12)
    
    cos_2psi = np.zeros_like(den)
    sin_2psi = np.zeros_like(den)
    cos_2psi[mask] = (nz_prime[mask]**2 - ny_prime[mask]**2) / den[mask]
    sin_2psi[mask] = 2 * ny_prime[mask] * nz_prime[mask] / den[mask]
    
    int_weight = w_p_array * grid['w_Theta_array'] * grid['dPhi']
    
    I_net = np.sum(int_weight * I_prime(mu_masked))
    Q_net = np.sum(int_weight * Q_prime(mu_masked, cos_2psi))
    U_net = np.sum(int_weight * U_prime(mu_masked, sin_2psi))
    
    return I_net, Q_net, U_net

def get_Pi(I_net, Q_net, U_net): return np.sqrt(Q_net**2 + U_net**2) / I_net * 100
def get_chi(Q_net, U_net): return 0.5 * np.arctan2(U_net, Q_net)

def find_maximum_in_grid(a, b, theta_array, phi_array, n_array, r2_array, grid):
    """Helper function to find the maximum polarization degree."""
    Pi_max = -np.inf
    theta_max, phi_max, chi_max = 0.0, 0.0, 0.0
    
    for theta_val in theta_array:
        for phi_val in phi_array:
            I, Q, U = integrate_all_parameters(
                a, b, theta_val, phi_val, n_array, r2_array, grid
            )
            Pi = get_Pi(I, Q, U)
            
            if Pi > Pi_max:
                Pi_max = Pi
                theta_max = theta_val
                phi_max = phi_val
                chi_max = get_chi(Q, U)
                
    return Pi_max, theta_max, phi_max, chi_max

# ==========================================
# Main Execution & Coarse-to-Fine Search
# ==========================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"ellipsoid_polarization_maximum_db_{timestamp}.h5"
    
    print(f"Starting Fast Coarse-to-Fine Search. Saving to: {output_filename}")
    
    grid_sparse = setup_integration_grid(N_THETA_sparse, N_PHI_sparse)
    grid_fine   = setup_integration_grid(N_THETA_fine, N_PHI_fine)

    with h5py.File(output_filename, "w") as f:
        # Save grid metadata
        g = f.create_group("grid")
        g.create_dataset("log_a", data=log_a_list)
        g.create_dataset("log_b", data=log_b_list)
        
        Pi_max_ds    = f.create_dataset("Pi_max/value", shape=(Na, Nb), dtype="f4")
        theta_max_ds = f.create_dataset("Pi_max/theta", shape=(Na, Nb), dtype="f4")
        phi_max_ds   = f.create_dataset("Pi_max/phi",   shape=(Na, Nb), dtype="f4")
        chi_max_ds   = f.create_dataset("Pi_max/chi",   shape=(Na, Nb), dtype="f4")
        
        # Sparse viewing angle arrays
        theta_sparse_arr = np.linspace(0, np.pi/2, Nt_sparse)
        phi_sparse_arr   = np.linspace(0, np.pi/2, Np_sparse)

        for ia, log_a in enumerate(log_a_list):
            a = 10**log_a
            for ib, log_b in enumerate(log_b_list):
                b = 10**log_b
                
                print(f"Processing ia={ia:02d}, ib={ib:02d} (a={a:.2e}, b={b:.2e}) ...", end=" ", flush=True)
                
                if ia == ib:
                    current_phi_sparse = np.array([0.0])
                else:
                    current_phi_sparse = phi_sparse_arr

                # --------------------------------------------------
                # Step 1: SPARSE SEARCH
                # --------------------------------------------------
                n_array_sp, r2_array_sp = compute_ellipsoid_geometry(a, b, grid_sparse)
                _, theta_0, phi_0, _ = find_maximum_in_grid(
                    a, b, theta_sparse_arr, current_phi_sparse, 
                    n_array_sp, r2_array_sp, grid_sparse
                )
                
                # --------------------------------------------------
                # Step 2: FINE SEARCH
                # --------------------------------------------------
                # Create a dynamic fine grid using the parameterized step sizes
                theta_fine_arr = np.clip(
                    np.linspace(theta_0 - theta_sparse_step, theta_0 + theta_sparse_step, Nt_fine), 
                    0.0, np.pi/2
                )
                
                if ia == ib:
                    current_phi_fine = np.array([0.0])
                else:
                    current_phi_fine = np.clip(
                        np.linspace(phi_0 - phi_sparse_step, phi_0 + phi_sparse_step, Np_fine), 
                        0.0, np.pi/2
                    )
                
                n_array_fi, r2_array_fi = compute_ellipsoid_geometry(a, b, grid_fine)
                Pi_max, theta_max, phi_max, chi_max = find_maximum_in_grid(
                    a, b, theta_fine_arr, current_phi_fine, 
                    n_array_fi, r2_array_fi, grid_fine
                )
                
                # --------------------------------------------------
                # Step 3: SAVE AND FLUSH
                # --------------------------------------------------
                Pi_max_ds[ia, ib]    = Pi_max
                theta_max_ds[ia, ib] = theta_max
                phi_max_ds[ia, ib]   = phi_max
                chi_max_ds[ia, ib]   = chi_max
                
                f.flush()
                
                print(f"Done! (Pi_max = {Pi_max:.2f}%)")
