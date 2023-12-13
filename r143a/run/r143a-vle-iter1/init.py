import signac
import numpy as np
import unyt as u
import sys

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.r143a import R143aConstants

def init_project():

    # Initialize project
    project = signac.init_project()

    # Define temps
    temps = [
        240.0 * u.K,
        260.0 * u.K,
        280.0 * u.K,
        300.0 * u.K,
        320.0 * u.K
    ]

    # Run at vapor pressure
    press = {
        240: (1.8911 * u.bar),
        260: (4.0263 * u.bar),
        280: (7.6320 * u.bar),
        300: (13.2446 * u.bar),
        320: (21.4971 * u.bar),
    }

    n_vap = 160 # number of molecules in vapor phase
    n_liq = 640

    # Experimental density
    R143a = R143aConstants()

    # Load samples from Latin hypercube
    lh_samples = np.genfromtxt("../../analysis/csv/r143a-vle-iter1-params.csv",delimiter=",",skip_header=1,)[:, 1:]

    # Define bounds on sigma/epsilon
    bounds_sigma = np.asarray([[3.0, 4.0], [3.0, 4.0], [2.0, 4.0], [1.5, 3.0],])  # C1 # C2  # F  # H

    bounds_epsilon = np.asarray(
        [[20.0, 70.0], [20.0, 70.0], [15.0, 40.0], [2.0, 10.0],]  # C1 # C2 # F  # H
    )

    bounds = np.vstack((bounds_sigma, bounds_epsilon))

    # Convert scaled latin hypercube samples to physical values
    scaled_params = values_scaled_to_real(lh_samples, bounds)

    for temp in temps:
        for sample in scaled_params:

            # Unpack the sample
            (sigma_C1, sigma_C2, sigma_F1, sigma_H1, epsilon_C1, epsilon_C2, epsilon_F1, epsilon_H1) = sample

            # Define the state point
            state_point = {
                "T": float(temp.in_units(u.K).value),
                "P": float(press[int(temp.in_units(u.K).value)].in_units(u.bar).value),
                "sigma_C1": float((sigma_C1 * u.Angstrom).in_units(u.nm).value),
                "sigma_C2": float((sigma_C2 * u.Angstrom).in_units(u.nm).value),
                "sigma_F1": float((sigma_F1 * u.Angstrom).in_units(u.nm).value),
                "sigma_H1": float((sigma_H1 * u.Angstrom).in_units(u.nm).value),
                "epsilon_C1": float((epsilon_C1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_C2": float((epsilon_C2 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_F1": float((epsilon_F1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_H1": float((epsilon_H1 * u.K * u.kb).in_units("kJ/mol").value),
                "N_vap": n_vap,
                "N_liq": n_liq,
                "expt_liq_density": R143a.expt_liq_density[
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
