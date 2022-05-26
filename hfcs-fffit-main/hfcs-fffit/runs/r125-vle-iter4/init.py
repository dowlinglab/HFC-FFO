import sys
import signac
import numpy as np
import unyt as u

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.r125 import R125Constants


def init_project():

    # Initialize project
    project = signac.init_project("r125-vle-iter4")

    # Define temps
    temps = [
        229.0 * u.K,
        249.0 * u.K,
        269.0 * u.K,
        289.0 * u.K,
        309.0 * u.K,
    ]

    # Run at vapor pressure
    press = {
        229: (123.65 * u.kPa),
        249: (290.76 * u.kPa),
        269: (592.27 * u.kPa),
        289: (1082.84 * u.kPa),
        309: (1824.93 * u.kPa),
    }

    n_vap = 160
    n_liq = 640

    # Experimental density
    R125 = R125Constants()

    # Load samples from Latin hypercube
    lh_samples = np.genfromtxt(
        "../../analysis/csv/r125-vle-iter4-params.csv",
        delimiter=",",
        skip_header=1,
    )[:, 1:]

    # Define bounds on sigma/epsilon
    bounds_sigma = np.asarray(
        [
            [3.0, 4.0],  # C
            [3.0, 4.0],  # C
            [2.5, 3.5],  # F
            [2.5, 3.5],  # F
            [1.7, 2.7],  # H
        ]
    )

    bounds_epsilon = np.asarray(
        [
            [20.0, 60.0],  # C
            [20.0, 60.0],  # C
            [15.0, 40.0],  # F
            [15.0, 40.0],  # F
            [2.0, 10.0],  # H
        ]
    )

    bounds = np.vstack((bounds_sigma, bounds_epsilon))

    # Convert scaled latin hypercube samples to physical values
    scaled_params = values_scaled_to_real(lh_samples, bounds)

    for temp in temps:
        for sample in scaled_params:

            # Unpack the sample
            (
                sigma_C1,
                sigma_C2,
                sigma_F1,
                sigma_F2,
                sigma_H1,
                epsilon_C1,
                epsilon_C2,
                epsilon_F1,
                epsilon_F2,
                epsilon_H1,
            ) = sample

            # Define the state point
            state_point = {
                "T": float(temp.in_units(u.K).value),
                "P": float(
                    press[int(temp.in_units(u.K).value)].in_units(u.bar).value
                ),
                "sigma_C1": float(
                    (sigma_C1 * u.Angstrom).in_units(u.nm).value
                ),
                "sigma_C2": float(
                    (sigma_C2 * u.Angstrom).in_units(u.nm).value
                ),
                "sigma_F1": float(
                    (sigma_F1 * u.Angstrom).in_units(u.nm).value
                ),
                "sigma_F2": float(
                    (sigma_F2 * u.Angstrom).in_units(u.nm).value
                ),
                "sigma_H1": float(
                    (sigma_H1 * u.Angstrom).in_units(u.nm).value
                ),
                "epsilon_C1": float(
                    (epsilon_C1 * u.K * u.kb).in_units("kJ/mol").value
                ),
                "epsilon_C2": float(
                    (epsilon_C2 * u.K * u.kb).in_units("kJ/mol").value
                ),
                "epsilon_F1": float(
                    (epsilon_F1 * u.K * u.kb).in_units("kJ/mol").value
                ),
                "epsilon_F2": float(
                    (epsilon_F2 * u.K * u.kb).in_units("kJ/mol").value
                ),
                "epsilon_H1": float(
                    (epsilon_H1 * u.K * u.kb).in_units("kJ/mol").value
                ),
                "N_vap": n_vap,
                "N_liq": n_liq,
                "expt_liq_density": R125.expt_liq_density[
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
