import signac
import numpy as np
import unyt as u
import sys

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.r170 import R170Constants

def init_project():

    # Initialize project
    project = signac.init_project("r170-vle-iter1")
    
    # Define temps
    temps = [
        210.0 * u.K,
        230.0 * u.K,
        250.0 * u.K,
        270.0 * u.K,
        290.0 * u.K
    ]

    # Run at vapor pressure
    press = {
        210: (3.338 * u.bar),
        230: (7.0018 * u.bar),
        250: (13.008 * u.bar),
        270: (22.1 * u.bar),
        290: (35.159 * u.bar),
    }

    n_vap = 160 # number of molecules in vapor phase
    n_liq = 640

    # Experimental density
    R170 = R170Constants()

    # Load samples from Latin hypercube
    lh_samples = np.genfromtxt("../../analysis/csv/r170-vle-iter1-params.csv",delimiter=",",skip_header=1,)[:, 1:]
    
    # Define bounds on sigma/epsilon
    bounds_sigma = np.asarray([[3.0, 4.0], [1.5, 3.0]])  # C1 # H

    bounds_epsilon = np.asarray(
        [[20.0, 75.0], [2.0, 10.0]])  # C1 # H

    bounds = np.vstack((bounds_sigma, bounds_epsilon))

    # Convert scaled latin hypercube samples to physical values
    scaled_params = values_scaled_to_real(lh_samples, bounds)

    for temp in temps:
        for sample in scaled_params:

            # Unpack the sample
            (sigma_C1, sigma_H1, epsilon_C1, epsilon_H1) = sample

            # Define the state point
            state_point = {
                "T": float(temp.in_units(u.K).value),
                "P": float(press[int(temp.in_units(u.K).value)].in_units(u.bar).value),
                "sigma_C1": float((sigma_C1 * u.Angstrom).in_units(u.nm).value),
                "sigma_H1": float((sigma_H1 * u.Angstrom).in_units(u.nm).value),
                "epsilon_C1": float((epsilon_C1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_H1": float((epsilon_H1 * u.K * u.kb).in_units("kJ/mol").value),
                "N_vap": n_vap,
                "N_liq": n_liq,
                "expt_liq_density": R170.expt_liq_density[
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
