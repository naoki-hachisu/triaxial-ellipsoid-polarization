# triaxial-ellipsoid-polarization
Numerical calculation of polarization properties of Thomson-scattering triaxial ellipsoids.

This repository is divided into two main parts: **Data Preparation** and **Visualization**.

## 1. Data Preparation (01-04)

* **`01_generate_spheroid_map_oblate.py`**
  Calculates the expected degree of polarization for an oblate spheroid given a viewing angle $\theta$ and an axis ratio $1/a$. The result is saved as `spheroid_oblate_map.csv` in the `data/` directory. This is the source data for **Figure 4 (Oblate spheroids)**.

* **`02_generate_spheroid_map_prolate.py`**
  Calculates the expected degree of polarization for a prolate spheroid given a viewing angle $\theta$ and an axis ratio $a$. The result is saved as `spheroid_prolate_map.csv` in the `data/` directory. This is the source data for **Figure 4 (Prolate spheroids)**.

* **`03_generate_polarization_database.py`**
  Generates a comprehensive database containing the degree of polarization ($\Pi$), polarization angle ($\chi$), and Stokes parameters ($I, Q, U$) over a 4-dimensional parameter space: shape parameters ($\log a, \log b$) and viewing angles ($\theta, \phi$). It also provides a rough estimation of the maximum degree of polarization ($\Pi_{\rm max}$) and its corresponding viewing direction ($\theta_{\rm max}, \phi_{\rm max}$) and angle ($\chi_{\rm max}$). 
  A reduced version ($N_a=N_b=5$) is provided as `ellipsoid_polarization_database_small.h5` in the `data/` directory, which is used for **Figures 5 and 7**. The high-resolution database ($N_a=N_b=33$, `ellipsoid_polarization_database.h5`) is too large for GitHub and will be publicly available on Zenodo.

* **`04_generate_polarization_maximum_database.py`**
  Performs a detailed calculation to find the maximum degree of polarization ($\Pi_{\rm max}$) and its exact viewing direction ($\theta_{\rm max}, \phi_{\rm max}$) and angle ($\chi_{\rm max}$) for given shape parameters ($\log a, \log b$). The output is saved as `ellipsoid_polarization_maximum_database.h5` in the `data/` directory. This is the source data for **Figure 8**.

## 2. Visualization (05)

* **`05_generate_figures.ipynb`**
  A Jupyter Notebook that generates all the figures presented in the paper using the data prepared in steps 01-04. Note that this calculation requires the tabulated radiative transfer solutions $I(\mu)$ and $Q(\mu)$ from Chandrasekhar (1960), which are provided as `chandrasekhar_1960.csv` in the `data/` directory.
