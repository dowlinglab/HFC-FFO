import numpy as np
import pandas as pd

from fffit.utils import values_real_to_scaled, values_scaled_to_real


def prepare_df_density(df_csv, molecule, liquid_density_threshold):
    """Prepare a pandas dataframe for fitting a GP model to density data

    Performs the following actions:
       - Renames "density" to "md_density"
       - Adds "expt_density"
       - Adds "is_liquid"
       - Converts all values from physical values to scaled values

    Parameters
    ----------
    df_csv : pd.DataFrame
        The dataframe as loaded from a CSV file with the signac results
    molecule : R32Constants, R125Constants
        An instance of a molecule constants class
    liquid_density_threshold : float
        Density threshold (kg/m^3) for distinguishing liquid and vapor

    Returns
    -------
    df_all : pd.DataFrame
        The dataframe with scaled parameters, temperature, density, and is_liquid
    df_liquid : pd.DataFrame
        `df_all` where `is_liquid` is True
    df_vapor : pd.DataFrame
        `df_all` where `is_liquid` is False
    """
    if "density" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'density'")
    if "temperature" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'temperature'")
    for param in list(molecule.param_names):
        if param not in df_csv.columns:
            raise ValueError(
                f"df_csv must contain a column for parameter: '{param}'"
            )

    # Add expt density and is_liquid
    df_all = df_csv.rename(columns={"density": "md_density"})
    df_all["expt_density"] = df_all["temperature"].apply(
        lambda temp: molecule.expt_liq_density[int(temp)]
    )
    df_all["is_liquid"] = df_all["md_density"].apply(
        lambda x: x > liquid_density_threshold
    )

    # Scale all values
    scaled_param_values = values_real_to_scaled(
        df_all[list(molecule.param_names)], molecule.param_bounds
    )
    scaled_temperature = values_real_to_scaled(
        df_all["temperature"], molecule.temperature_bounds
    )
    scaled_md_density = values_real_to_scaled(
        df_all["md_density"], molecule.liq_density_bounds
    )
    scaled_expt_density = values_real_to_scaled(
        df_all["expt_density"], molecule.liq_density_bounds
    )
    df_all[list(molecule.param_names)] = scaled_param_values
    df_all["temperature"] = scaled_temperature
    df_all["md_density"] = scaled_md_density
    df_all["expt_density"] = scaled_expt_density

    # Split out vapor and liquid samples
    df_liquid = df_all[df_all["is_liquid"] == True]
    df_vapor = df_all[df_all["is_liquid"] == False]

    return df_all, df_liquid, df_vapor


def prepare_df_vle(df_csv, molecule):
    """Prepare a pandas dataframe for fitting a GP model to density data

    Performs the following actions:
       - Renames "liq_density" to "sim_liq_density"
       - Renames "vap_density" to "sim_vap_density"
       - Renames "Pvap" to "sim_Pvap"
       - Removes "liq_enthalpy" and "vap_enthalpy" and adds "sim_Hvap"
       - Adds "expt_liq_density"
       - Adds "expt_vap_density"
       - Adds "expt_Pvap"
       - Adds "expt_Hvap"
       - Adds "is_liquid"
       - Converts all values from physical values to scaled values

    Parameters
    ----------
    df_csv : pd.DataFrame
        The dataframe as loaded from a CSV file with the signac results
    molecule : R32Constants, R125Constants
        An instance of a molecule constants class
    n_molecules : int
        The number of molecules in the simulation

    Returns
    -------
    df_all : pd.DataFrame
        The dataframe with scaled parameters and MD/expt. properties
    """
    if "liq_density" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'liq_density'")
    if "vap_density" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'vap_density'")
    if "Pvap" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'Pvap'")
    if "Hvap" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'Hvap'")
    if "liq_enthalpy" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'liq_enthalpy'")
    if "vap_enthalpy" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'vap_enthalpy'")
    if "temperature" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'temperature'")
    for param in list(molecule.param_names):
        if param not in df_csv.columns:
            raise ValueError(
                f"df_csv must contain a column for parameter: '{param}'"
            )

    # Rename properties to MD, calculate Hvap, add expt properties
    df_all = df_csv.rename(columns={"liq_density": "sim_liq_density"})
    df_all = df_all.rename(columns={"vap_density": "sim_vap_density"})
    df_all = df_all.rename(columns={"Pvap": "sim_Pvap"})
    df_all = df_all.rename(columns={"Hvap": "sim_Hvap"})
    df_all.drop(columns="vap_enthalpy", inplace=True)
    df_all.drop(columns="liq_enthalpy", inplace=True)

    # Convert Hvap to kJ/kg
    df_all["sim_Hvap"] = (
        df_all["sim_Hvap"] / molecule.molecular_weight * 1000.0
    )

    df_all["expt_liq_density"] = df_all["temperature"].apply(
        lambda temp: molecule.expt_liq_density[int(temp)]
    )
    df_all["expt_vap_density"] = df_all["temperature"].apply(
        lambda temp: molecule.expt_vap_density[int(temp)]
    )
    df_all["expt_Pvap"] = df_all["temperature"].apply(
        lambda temp: molecule.expt_Pvap[int(temp)]
    )
    df_all["expt_Hvap"] = df_all["temperature"].apply(
        lambda temp: molecule.expt_Hvap[int(temp)]
    )

    # Scale all values
    scaled_param_values = values_real_to_scaled(
        df_all[list(molecule.param_names)], molecule.param_bounds
    )
    scaled_temperature = values_real_to_scaled(
        df_all["temperature"], molecule.temperature_bounds
    )
    scaled_sim_liq_density = values_real_to_scaled(
        df_all["sim_liq_density"], molecule.liq_density_bounds
    )
    scaled_sim_vap_density = values_real_to_scaled(
        df_all["sim_vap_density"], molecule.vap_density_bounds
    )
    scaled_sim_Pvap = values_real_to_scaled(
        df_all["sim_Pvap"], molecule.Pvap_bounds
    )
    scaled_sim_Hvap = values_real_to_scaled(
        df_all["sim_Hvap"], molecule.Hvap_bounds
    )
    scaled_expt_liq_density = values_real_to_scaled(
        df_all["expt_liq_density"], molecule.liq_density_bounds
    )
    scaled_expt_vap_density = values_real_to_scaled(
        df_all["expt_vap_density"], molecule.vap_density_bounds
    )
    scaled_expt_Pvap = values_real_to_scaled(
        df_all["expt_Pvap"], molecule.Pvap_bounds
    )
    scaled_expt_Hvap = values_real_to_scaled(
        df_all["expt_Hvap"], molecule.Hvap_bounds
    )
    df_all[list(molecule.param_names)] = scaled_param_values
    df_all["temperature"] = scaled_temperature
    df_all["sim_liq_density"] = scaled_sim_liq_density
    df_all["sim_vap_density"] = scaled_sim_vap_density
    df_all["sim_Pvap"] = scaled_sim_Pvap
    df_all["sim_Hvap"] = scaled_sim_Hvap
    df_all["expt_liq_density"] = scaled_expt_liq_density
    df_all["expt_vap_density"] = scaled_expt_vap_density
    df_all["expt_Pvap"] = scaled_expt_Pvap
    df_all["expt_Hvap"] = scaled_expt_Hvap

    return df_all


def classify_samples(samples, classifier):
    """Evaulate the classifer and return predicted liquid and vapor samples

    Parameters
    ----------
    samples : np.ndarray, shape=(n_samples, n_params)
        Samples to rank
    classifier : sklearn.svm.SVC
        Classifier to distinguish between liquid and vapor

    Returns
    -------
    liquid_samples : np.ndarray, shape=(n_liquid, n_params)
        Samples classified as liquid
    vapor_samples : np.ndarray, shape=(n_liquid, n_params)
        Samples classified as liquid

    """

    # Classification performed at the highest temperature
    # Append highest temperature (1.0) to LH samples
    samples_temperature = np.hstack(
        (samples, np.tile(1.0, (samples.shape[0], 1)))
    )

    # Apply clasifier
    pred = classifier.predict(samples_temperature)

    # Separate LH samples into predicted liquid and predicted vapor
    liquid_samples = samples[np.where(pred == 1)]
    vapor_samples = samples[np.where(pred == 0)]
    print("Shape of samples to classify:", samples.shape)
    print("Shape of the predicted liquid samples:", liquid_samples.shape)
    print("Shape of the predicted vapor samples:", vapor_samples.shape)

    return liquid_samples, vapor_samples


def rank_samples(samples, gp_model, molecule, property_name, property_offset=0.0):
    """Evalulate the GP model for a samples and return ranked results
    from lowest to highest MSE with experiment across the temperature range

    Parameters
    ----------
    samples : np.ndarray, shape=(n_samples, n_params)
        Samples to rank
    gp_model : gpflow.model
        GP model to predict the property_name of each sample
    molecule : R32Constants, R125Constants
        An instance of a molecule constants class
    property_name : string
        The name of the property of interest. Valid options are
        "liq_density", "vap_density", "Pvap", "Hvap"
    property_offset : float
        Adjust the value predicted by the gp model by this amount.
        Quantity specified in physical units

    Returns
    -------
    ranked_samples : pd.DataFrame
        Samples sorted by MSE
    """

    valid_property_names = [
        "sim_liq_density",
        "sim_vap_density",
        "sim_Pvap",
        "sim_Hvap",
    ]

    if property_name not in valid_property_names:
        raise ValueError(
            "Invalid property_name {}. Supported property_names are "
            "{}".format(property_name, valid_property_names)
        )

    temperature_bounds = molecule.temperature_bounds
    if property_name == "sim_liq_density":
        expt_property = molecule.expt_liq_density
        property_bounds = molecule.liq_density_bounds
    elif property_name == "sim_vap_density":
        expt_property = molecule.expt_vap_density
        property_bounds = molecule.vap_density_bounds
    elif property_name == "sim_Pvap":
        expt_property = molecule.expt_Pvap
        property_bounds = molecule.Pvap_bounds
    elif property_name == "sim_Hvap":
        expt_property = molecule.expt_Hvap
        property_bounds = molecule.Hvap_bounds

    # Apply GP model and calculate mean squared errors (MSE) between
    # GP model predictions and experimental data for all parameter samples
    mse = _calc_gp_mse(
        gp_model, samples, expt_property, property_bounds, temperature_bounds, property_offset
    )
    # Make pandas dataframes, rank, and return
    samples_mse = np.hstack((samples, mse.reshape(-1, 1)))
    samples_mse = pd.DataFrame(
        samples_mse, columns=list(molecule.param_names) + ["mse"]
    )
    ranked_samples = samples_mse.sort_values("mse")

    return ranked_samples


def _calc_gp_mse(
    gp_model, samples, expt_property, property_bounds, temperature_bounds, property_offset=0.0
):
    """Calculate the MSE between the GP model and experiment for samples"""

    all_errs = np.empty(shape=(samples.shape[0], len(expt_property.keys())))
    col_idx = 0
    for (temp, density) in expt_property.items():
        scaled_temp = values_real_to_scaled(temp, temperature_bounds)
        xx = np.hstack((samples, np.tile(scaled_temp, (samples.shape[0], 1))))
        means_scaled, vars_scaled = gp_model.predict_f(xx)
        means = values_scaled_to_real(means_scaled, property_bounds)
        means = means + property_offset
        err = means - density
        all_errs[:, col_idx] = err[:, 0]
        col_idx += 1

    return np.mean(all_errs ** 2, axis=1)
