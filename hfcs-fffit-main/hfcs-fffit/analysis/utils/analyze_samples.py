import numpy as np
import pandas as pd
from scipy.stats import linregress

from fffit.utils import values_scaled_to_real

def prepare_df_density_errors(df, molecule):
    """Create a dataframe with mean square error (mse) and mean absolute
    percent error (mape) for each unique parameter set.

    Parameters
    ----------
    df : pandas.Dataframe
        per simulation results
    molecule : R32, R125
        molecule class with bounds/experimental data

    Returns
    -------
    df_new : pandas.Dataframe
        dataframe with one row per parameter set and including
        the MSE and MAPE for liq_density
    """
    new_data = []
    for group, values in df.groupby(list(molecule.param_names)):

        # Temperatures
        temps = values_scaled_to_real(
            values["temperature"], molecule.temperature_bounds
        )

        # Liquid density
        sim_liq_density = values_scaled_to_real(
            values["md_density"], molecule.liq_density_bounds
        )
        expt_liq_density = values_scaled_to_real(
            values["expt_density"], molecule.liq_density_bounds
        )
        mse_liq_density = np.mean((sim_liq_density - expt_liq_density) ** 2)
        mape_liq_density = (
            np.mean(
                np.abs((sim_liq_density - expt_liq_density) / expt_liq_density)
            )
            * 100.0
        )
        properties = {
            f"sim_liq_density_{float(temp):.0f}K": float(liq_density)
            for temp, liq_density in zip(temps, sim_liq_density)
        }

        new_quantities = {
            **properties,
            "mse_liq_density": mse_liq_density,
            "mape_liq_density": mape_liq_density,
        }

        new_data.append(list(group) + list(new_quantities.values()))

    columns = list(molecule.param_names) + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)

    return new_df

def prepare_df_vle_errors(df, molecule):
    """Create a dataframe with mean square error (mse) and mean absolute
    percent error (mape) for each unique parameter set. The critical
    temperature and density are also evaluated.

    Parameters
    ----------
    df : pandas.Dataframe
        per simulation results
    molecule : R32, R125
        molecule class with bounds/experimental data

    Returns
    -------
    df_new : pandas.Dataframe
        dataframe with one row per parameter set and including
        the MSE and MAPE for liq_density, vap_density, pvap, hvap,
        critical temperature, critical density
    """
    new_data = []
    for group, values in df.groupby(list(molecule.param_names)):

        # Temperatures
        temps = values_scaled_to_real(
            values["temperature"], molecule.temperature_bounds
        )

        # Liquid density
        sim_liq_density = values_scaled_to_real(
            values["sim_liq_density"], molecule.liq_density_bounds
        )
        expt_liq_density = values_scaled_to_real(
            values["expt_liq_density"], molecule.liq_density_bounds
        )
        mse_liq_density = np.mean((sim_liq_density - expt_liq_density) ** 2)
        mape_liq_density = (
            np.mean(
                np.abs((sim_liq_density - expt_liq_density) / expt_liq_density)
            )
            * 100.0
        )
        properties = {
            f"sim_liq_density_{float(temp):.0f}K": float(liq_density)
            for temp, liq_density in zip(temps, sim_liq_density)
        }
        # Vapor density
        sim_vap_density = values_scaled_to_real(
            values["sim_vap_density"], molecule.vap_density_bounds
        )
        expt_vap_density = values_scaled_to_real(
            values["expt_vap_density"], molecule.vap_density_bounds
        )
        mse_vap_density = np.mean((sim_vap_density - expt_vap_density) ** 2)
        mape_vap_density = (
            np.mean(
                np.abs((sim_vap_density - expt_vap_density) / expt_vap_density)
            )
            * 100.0
        )
        properties.update(
            {
                f"sim_vap_density_{float(temp):.0f}K": float(vap_density)
                for temp, vap_density in zip(temps, sim_vap_density)
            }
        )

        # Vapor pressure
        sim_Pvap = values_scaled_to_real(
            values["sim_Pvap"], molecule.Pvap_bounds
        )
        expt_Pvap = values_scaled_to_real(
            values["expt_Pvap"], molecule.Pvap_bounds
        )
        mse_Pvap = np.mean((sim_Pvap - expt_Pvap) ** 2)
        mape_Pvap = np.mean(np.abs((sim_Pvap - expt_Pvap) / expt_Pvap)) * 100.0
        properties.update(
            {
                f"sim_Pvap_{float(temp):.0f}K": float(Pvap)
                for temp, Pvap in zip(temps, sim_Pvap)
            }
        )

        # Enthalpy of vaporization
        sim_Hvap = values_scaled_to_real(
            values["sim_Hvap"], molecule.Hvap_bounds
        )
        expt_Hvap = values_scaled_to_real(
            values["expt_Hvap"], molecule.Hvap_bounds
        )
        mse_Hvap = np.mean((sim_Hvap - expt_Hvap) ** 2)
        mape_Hvap = np.mean(np.abs((sim_Hvap - expt_Hvap) / expt_Hvap)) * 100.0
        properties.update(
            {
                f"sim_Hvap_{float(temp):.0f}K": float(Hvap)
                for temp, Hvap in zip(temps, sim_Hvap)
            }
        )

        # Critical Point (Law of rectilinear diameters)
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
            temps.flatten(),
            ((sim_liq_density + sim_vap_density) / 2.0).flatten(),
        )

        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
            temps.flatten(),
            ((sim_liq_density - sim_vap_density) ** (1 / 0.32)).flatten(),
        )

        Tc = np.abs(intercept2 / slope2)
        mse_Tc = (Tc - molecule.expt_Tc) ** 2
        mape_Tc = np.abs((Tc - molecule.expt_Tc) / molecule.expt_Tc) * 100.0
        properties.update({"sim_Tc": Tc})

        rhoc = intercept1 + slope1 * Tc
        mse_rhoc = (rhoc - molecule.expt_rhoc) ** 2
        mape_rhoc = (
            np.abs((rhoc - molecule.expt_rhoc) / molecule.expt_rhoc) * 100.0
        )
        properties.update({"sim_rhoc": rhoc})

        new_quantities = {
            **properties,
            "mse_liq_density": mse_liq_density,
            "mse_vap_density": mse_vap_density,
            "mse_Pvap": mse_Pvap,
            "mse_Hvap": mse_Hvap,
            "mse_Tc": mse_Tc,
            "mse_rhoc": mse_rhoc,
            "mape_liq_density": mape_liq_density,
            "mape_vap_density": mape_vap_density,
            "mape_Pvap": mape_Pvap,
            "mape_Hvap": mape_Hvap,
            "mape_Tc": mape_Tc,
            "mape_rhoc": mape_rhoc,
        }

        new_data.append(list(group) + list(new_quantities.values()))

    columns = list(molecule.param_names) + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)

    return new_df
