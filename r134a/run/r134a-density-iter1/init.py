import signac
import numpy as np
import unyt as u

from fffit.utils import values_scaled_to_real


def init_project():

    # Initialize project
    project = signac.init_project("r134a-density-iter1")

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
        240: (0.72481 * u.bar),
        260: (1.7684 * u.bar),
        280: (3.7271 * u.bar),
        300: (7.0282 * u.bar),
        320: (12.166 * u.bar),
    }

    # Run for 2.5 ns (1 fs timestep)
    nstepseq = 500000
    nstepsprod = 2500000

    # Load samples from Latin hypercube
    lh_samples = np.genfromtxt("data/LHS_200_x_10.csv",delimiter=",")
    # Define bounds on sigma/epsilon
    bounds_sigma = np.asarray([[3.0, 4.0], [3.0, 4.0], [2.5, 3.5], [2.5, 3.5], [1.5, 3.0]])  # C1 # C2  # F1 F2  # H

    bounds_epsilon = np.asarray(
        [[20.0, 75.0], [20.0, 75.0], [15.0, 40.0], [15.0, 40.0], [2.0, 10.0]])  # C1 # C2 # F1 F2  # H
    

    bounds = np.vstack((bounds_sigma, bounds_epsilon))
    # Convert scaled latin hypercube samples to physical values
    scaled_params = values_scaled_to_real(lh_samples, bounds)
    
    for temp in temps:
        for sample in scaled_params:

            # Unpack the sample
            (sigma_C1, sigma_C2, sigma_F1, sigma_F2, sigma_H1, epsilon_C1, epsilon_C2, epsilon_F1, epsilon_F2, epsilon_H1) = sample

            # Define the state point
            state_point = {
                "T": float(temp.in_units(u.K).value),
                "P": float(press[int(temp.in_units(u.K).value)].in_units(u.bar).value),
                "nstepseq": nstepseq,
                "nstepsprod": nstepsprod,
                "sigma_C1": float((sigma_C1 * u.Angstrom).in_units(u.nm).value),
                "sigma_C2": float((sigma_C2 * u.Angstrom).in_units(u.nm).value),
                "sigma_F1": float((sigma_F1 * u.Angstrom).in_units(u.nm).value),
                "sigma_F2": float((sigma_F2 * u.Angstrom).in_units(u.nm).value),
                "sigma_H1": float((sigma_H1 * u.Angstrom).in_units(u.nm).value),
                "epsilon_C1": float((epsilon_C1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_C2": float((epsilon_C2 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_F1": float((epsilon_F1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_F2": float((epsilon_F2 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_H1": float((epsilon_H1 * u.K * u.kb).in_units("kJ/mol").value),
            }

            job = project.open_job(state_point)
            job.init()


if __name__ == "__main__":
    init_project()
