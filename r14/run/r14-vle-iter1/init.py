import signac
import numpy as np
import unyt as u
import sys

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.r14 import R14Constants

def init_project():

    # Initialize project
    project = signac.init_project()

    # Define temps
    temps = [
        130.0 * u.K,
        150.0 * u.K,
        170.0 * u.K,
        190.0 * u.K,
        210.0 * u.K
    ]

    # Run at vapor pressure
    press = {
        130: (0.30392 * u.bar),
        150: (1.412 * u.bar),
        170: (4.4054 * u.bar),
        190: (10.663 * u.bar),
        210: (21.864 * u.bar),
    }

    n_vap = 160 # number of molecules in vapor phase
    n_liq = 640

    # Experimental density
    R14 = R14Constants()

    # Load samples from Latin hypercube
    lh_samples = np.genfromtxt("../../analysis/csv/r14-vle-iter1-params.csv",delimiter=",",skip_header=1,)[:, 1:]

    # Define bounds on sigma/epsilon
    bounds_sigma = np.asarray([[2.0, 4.0], [2.5, 3.5]])  # C1 # F1 

    bounds_epsilon = np.asarray(
        [[10.0, 75.0], [15.0, 50.0]])  # C1 # F1

    bounds = np.vstack((bounds_sigma, bounds_epsilon))

    # Convert scaled latin hypercube samples to physical values
    scaled_params = values_scaled_to_real(lh_samples, bounds)
    for temp in temps:
        for sample in scaled_params:
            # Unpack the sample
            (sigma_C1, sigma_F1, epsilon_C1, epsilon_F1) = sample

            # Define the state point
            state_point = {
                "T": float(temp.in_units(u.K).value),
                "P": float(press[int(temp.in_units(u.K).value)].in_units(u.bar).value),
                "sigma_C1": float((sigma_C1 * u.Angstrom).in_units(u.nm).value),
                "sigma_F1": float((sigma_F1 * u.Angstrom).in_units(u.nm).value),
                "epsilon_C1": float((epsilon_C1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_F1": float((epsilon_F1 * u.K * u.kb).in_units("kJ/mol").value),
                "N_vap": n_vap,
                "N_liq": n_liq,
                "expt_liq_density": R14.expt_liq_density[
                    int(temp.in_units(u.K).value)
                ],
                "nsteps_liqeq": 5000,
                "nsteps_eq": 10000,
                "nsteps_prod": 100000,
            }            

            job = project.open_job(state_point)
            job.init()


if __name__ == "__main__":
    init_project()
