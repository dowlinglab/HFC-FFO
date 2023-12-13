import signac
import numpy as np
import unyt as u
import sys

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.r50 import R50Constants

def init_project():

    # Initialize project
    project = signac.init_project()
     # Define temps
    temps = [
        130.0 * u.K,
        140.0 * u.K,
        150.0 * u.K,
        160.0 * u.K,
        170.0 * u.K
    ]

    # Run at vapor pressure
    press = {
        130: (3.6732 * u.bar),
        140: (6.4118 * u.bar),
        150: (10.4 * u.bar),
        160: (15.921 * u.bar),
        170: (23.283 * u.bar),
    }

    n_vap = 160 # number of molecules in vapor phase
    n_liq = 640

    # Experimental density
    R50 = R50Constants()

    # Load samples from Latin hypercube
    lh_samples = np.genfromtxt("../../analysis/csv/r50-vle-iter3-params.csv",delimiter=",",skip_header=1,)[:, 1:]

    # Define bounds on sigma/epsilon
    bounds_sigma = np.asarray([[3.0, 4.0], [1.5, 3.0]])  # C1 # H1 

    bounds_epsilon = np.asarray(
        [[20.0, 75.0], [2.0, 10.0]])  # C1 # H1

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
                "expt_liq_density": R50.expt_liq_density[
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
