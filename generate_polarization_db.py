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
Nt = 65
Np = 65

# Axis ratio grids
log_a_list = np.linspace(-1.0, 1.0, Na)
log_b_list = np.linspace(-1.0, 1.0, Nb)

# Viewing angle grids
theta_list = np.linspace(0, np.pi/2, Nt)
phi_list   = np.linspace(0, np.pi/2, Np)

# Integration grids
N_THETA = 512
N_PHI = 512

dPhi = 2 * np.pi / N_PHI
u, w_Theta = leggauss(N_THETA)

Theta_list = np.arccos(u)
Phi_list = (np.arange(N_PHI) + 0.5) * dPhi
Theta_grid, Phi_grid = np.meshgrid(Theta_list, Phi_list, indexing='ij')

sin_Theta = np.sin(Theta_grid)
cos_Theta = np.cos(Theta_grid)
sin_Phi = np.sin(Phi_grid)
cos_Phi = np.cos(Phi_grid)

# Cartesian coordinates on the unit sphere
x_sphere = sin_Theta * cos_Phi
y_sphere = sin_Theta * sin_Phi
z_sphere = cos_Theta

w_Theta_array = np.tile(w_Theta[:, np.newaxis], (1, N_PHI))

# ==========================================
# Interpolation Tables (External Data)
# ==========================================
# Read Chandrasekhar (1960) tables from an external CSV file
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
# Helper Functions
# ==========================================

def I_prime(mu):
    """
    Calculate the total specific intensity (I_l + I_r) for a given cosine of emission angle.
    """
    return I_l_spline(mu) + I_r_spline(mu)

def Q_prime(mu, cos_2psi):
    """
    Calculate the Stokes Q parameter component before integration.
    """
    return (I_l_spline(mu) - I_r_spline(mu)) * cos_2psi

def U_prime(mu, sin_2psi):
    """
    Calculate the Stokes U parameter component before integration.
    """
    return (I_l_spline(mu) - I_r_spline(mu)) * sin_2psi

def get_x_prime(theta, phi):
    """
    Compute the x-prime basis vector (line of sight) given viewing angles.
    """
    xx = np.sin(theta) * np.cos(phi)
    xy = np.sin(theta) * np.sin(phi)
    xz = np.cos(theta)
    return np.array([xx, xy, xz])

def get_y_prime(phi):
    """
    Compute the y-prime basis vector given the azimuthal viewing angle.
    """
    yx = -np.sin(phi)
    yy = np.cos(phi)
    yz = 0.0
    return np.array([yx, yy, yz])

def get_z_prime(theta, phi):
    """
    Compute the z-prime basis vector given viewing angles.
    """
    zx = -np.cos(theta) * np.cos(phi)
    zy = -np.cos(theta) * np.sin(phi)
    zz = np.sin(theta)
    return np.array([zx, zy, zz])

def get_mu_array(n_array, x_prime):
    """
    Calculate the cosine of the angle between the local normal and the viewing direction.
    """
    xx, xy, xz = x_prime
    nx, ny, nz = n_array
    return xx*nx + xy*ny + xz*nz

def get_r2_array(a, b, sin_T, cos_T, sin_P, cos_P):
    """
    Calculate the inverse squared distance to the origin on the ellipsoid surface.
    """
    s = sin_T * cos_P / a
    t = sin_T * sin_P / b
    u = cos_T
    return 1.0 / (s**2 + t**2 + u**2)

def get_w_array(r2_array, n_array, x_s, y_s, z_s):
    """
    Calculate the projected area weight for surface integration.
    """
    nx, ny, nz = n_array
    nr_array = nx*x_s + ny*y_s + nz*z_s
    epsilon = 1e-12
    return r2_array / np.maximum(np.abs(nr_array), epsilon)

def get_psi_array(n_array, y_prime, z_prime):
    """
    Calculate trigonometric components (cos(2*psi), sin(2*psi)) of the polarization rotation angle.
    """
    yx, yy, yz = y_prime
    zx, zy, zz = z_prime
    nx, ny, nz = n_array
    
    ny_prime = yx*nx + yy*ny + yz*nz
    nz_prime = zx*nx + zy*ny + zz*nz
    den = nz_prime**2 + ny_prime**2
    epsilon = 1e-12
    
    mask = (den > epsilon)
    cos_2phi = np.zeros_like(den)
    sin_2phi = np.zeros_like(den)
    cos_2phi[mask] = (nz_prime[mask]**2 - ny_prime[mask]**2) / den[mask]
    sin_2phi[mask] = 2 * ny_prime[mask] * nz_prime[mask] / den[mask]
    return cos_2phi, sin_2phi

def integrate_all_parameters(a, b, theta, phi, n_array, r2_array, w_Theta_arr, d_Phi, x_s, y_s, z_s):
    """
    Perform numerical integration over the visible, illuminated ellipsoid surface 
    to compute net Stokes parameters (I, Q, U).
    """
    x_prime = get_x_prime(theta, phi)
    y_prime = get_y_prime(phi)
    z_prime = get_z_prime(theta, phi)
    
    mu_array = get_mu_array(n_array, x_prime)
    mu_masked = np.maximum(mu_array, 0.0)
    w_array = get_w_array(r2_array, n_array, x_s, y_s, z_s)
    w_p_array = mu_masked * w_array
    cos_2psi, sin_2psi = get_psi_array(n_array, y_prime, z_prime)
    
    # Common integration weight factor
    int_weight = w_p_array * w_Theta_arr * d_Phi
    
    I_net = np.sum(int_weight * I_prime(mu_masked))
    Q_net = np.sum(int_weight * Q_prime(mu_masked, cos_2psi))
    U_net = np.sum(int_weight * U_prime(mu_masked, sin_2psi))
    
    return I_net, Q_net, U_net

def get_Pi(I_net, Q_net, U_net):
    """
    Calculate the degree of polarization in percentage.
    """
    return np.sqrt(Q_net**2 + U_net**2) / I_net * 100

def get_chi(Q_net, U_net):
    """
    Calculate the polarization angle (position angle) in radians.
    """
    return 0.5 * np.arctan2(U_net, Q_net)

# ==========================================
# Main Execution & HDF5 Output
# ==========================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"ellipsoid_polarization_db_{timestamp}.h5"
    
    print(f"Saving database to: {output_filename}")
    
    with h5py.File(output_filename, "w") as f:
        
        # Grid information
        g = f.create_group("grid")
        g.create_dataset("log_a", data=log_a_list)
        g.create_dataset("log_b", data=log_b_list)
        g.create_dataset("theta", data=theta_list)
        g.create_dataset("phi", data=phi_list)

        # 4D Dataset configurations
        ds_kwargs = {
            "shape": (Na, Nb, Nt, Np),
            "dtype": "f4",
            "chunks": (1, 1, Nt, Np),
            "compression": "gzip"
        }
        
        I_ds   = f.create_dataset("I/values", **ds_kwargs)
        Q_ds   = f.create_dataset("Q/values", **ds_kwargs)
        U_ds   = f.create_dataset("U/values", **ds_kwargs)
        Pi_ds  = f.create_dataset("Pi/values", **ds_kwargs)
        chi_ds = f.create_dataset("chi/values", **ds_kwargs)
        
        # 2D Maximum Datasets
        Pi_max_ds       = f.create_dataset("Pi_max/value", shape=(Na, Nb), dtype="f4")
        Pi_max_theta_ds = f.create_dataset("Pi_max/theta", shape=(Na, Nb), dtype="f4")
        Pi_max_phi_ds   = f.create_dataset("Pi_max/phi",   shape=(Na, Nb), dtype="f4")
        Pi_max_chi_ds   = f.create_dataset("Pi_max/chi",   shape=(Na, Nb), dtype="f4")

        # Metadata
        Pi_ds.attrs["unit"] = "percent"
        Pi_ds.attrs["definition"] = "sqrt(Q^2+U^2)/I * 100"
        chi_ds.attrs["unit"] = "radian"
        chi_ds.attrs["definition"] = "0.5 * atan2(U, Q)"
        
        # Computation
        for ia, log_a in enumerate(log_a_list):
            a = 10**log_a
            for ib, log_b in enumerate(log_b_list):
                b = 10**log_b
                
                print(f"Computing: a={a:.3e} (ia={ia}), b={b:.3e} (ib={ib})")
                
                nx = sin_Theta * cos_Phi / a**2
                ny = sin_Theta * sin_Phi / b**2
                nz = cos_Theta
                n_array = np.stack([nx, ny, nz], axis=0)
                
                norm = np.linalg.norm(n_array, axis=0, keepdims=True)
                epsilon = 1e-12
                n_array = n_array / np.maximum(norm, epsilon)
                r2_array = get_r2_array(a, b, sin_Theta, cos_Theta, sin_Phi, cos_Phi)
                
                for it, theta_val in enumerate(theta_list):
                    for ip, phi_val in enumerate(phi_list):
                        I, Q, U = integrate_all_parameters(
                            a, b, theta_val, phi_val, n_array, r2_array,
                            w_Theta_array, dPhi, x_sphere, y_sphere, z_sphere
                        )
                        
                        I_ds[ia, ib, it, ip] = I
                        Q_ds[ia, ib, it, ip] = Q
                        U_ds[ia, ib, it, ip] = U
                        
                        Pi_ds[ia, ib, it, ip] = get_Pi(I, Q, U)
                        chi_ds[ia, ib, it, ip] = get_chi(Q, U)
                
                # Extract maxima
                Pi_slice = Pi_ds[ia, ib, :, :]
                chi_slice = chi_ds[ia, ib, :, :]
                idx = np.argmax(Pi_slice)
                it_max, ip_max = np.unravel_index(idx, Pi_slice.shape)

                Pi_max_ds[ia, ib] = Pi_slice[it_max, ip_max]
                Pi_max_theta_ds[ia, ib] = theta_list[it_max] 
                Pi_max_phi_ds[ia, ib] = phi_list[ip_max]
                Pi_max_chi_ds[ia, ib] = chi_slice[it_max, ip_max]
        
                f.flush()