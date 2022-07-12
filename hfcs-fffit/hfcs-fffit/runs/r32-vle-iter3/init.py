import sys
import signac
import numpy as np
import unyt as u

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.r32 import R32Constants


def init_project():

    # Initialize project
    project = signac.init_project("r32-vle-iter3")

    # Define temps
    temps = [241.0 * u.K, 261.0 * u.K, 281.0 * u.K, 301.0 * u.K, 321.0 * u.K]

    # Run at vapor pressure
    press = {
        241: (2.5159 * u.bar),
        261: (5.4327 * u.bar),
        281: (10.426 * u.bar),
        301: (18.295 * u.bar),
        321: (29.989 * u.bar),
    }

    n_vap = 200
    n_liq = 800

    # Experimental density
    R32 = R32Constants()

    # Load samples
    lh_samples = np.genfromtxt(
        "../../analysis/csv/r32-vle-iter3-params.csv",
        delimiter=",",
        skip_header=1,
    )[:, 1:]

    # Define bounds on sigma/epsilon
    bounds_sigma = np.asarray(
        [[3.0, 4.0], [2.5, 3.5], [1.7, 2.7],]
    )  # C  # F  # H

    bounds_epsilon = np.asarray(
        [[20.0, 60.0], [15.0, 40.0], [2.0, 10.0],]  # C  # F  # H
    )

    bounds = np.vstack((bounds_sigma, bounds_epsilon))

    # Convert scaled latin hypercube samples to physical values
    scaled_params = values_scaled_to_real(lh_samples, bounds)

    for temp in temps:
        for sample in scaled_params:

            # Unpack the sample
            (
                sigma_C,
                sigma_F,
                sigma_H,
                epsilon_C,
                epsilon_F,
                epsilon_H,
            ) = sample

            # Define the state point
            state_point = {
                "T": float(temp.in_units(u.K).value),
                "P": float(
                    press[int(temp.in_units(u.K).value)].in_units(u.bar).value
                ),
                "sigma_C": float((sigma_C * u.Angstrom).in_units(u.nm).value),
                "sigma_F": float((sigma_F * u.Angstrom).in_units(u.nm).value),
                "sigma_H": float((sigma_H * u.Angstrom).in_units(u.nm).value),
                "epsilon_C": float(
                    (epsilon_C * u.K * u.kb).in_units("kJ/mol").value
                ),
                "epsilon_F": float(
                    (epsilon_F * u.K * u.kb).in_units("kJ/mol").value
                ),
                "epsilon_H": float(
                    (epsilon_H * u.K * u.kb).in_units("kJ/mol").value
                ),
                "N_vap": n_vap,
                "N_liq": n_liq,
                "expt_liq_density": R32.expt_liq_density[
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
