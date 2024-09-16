import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn import svm
import scipy.optimize as optimize
import unyt as u

from sklearn.model_selection import train_test_split

from fffit.utils import (
    shuffle_and_split,
    values_real_to_scaled,
    values_scaled_to_real,
    variances_scaled_to_real,
)

from fffit.plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

from fffit.models import run_gpflow_scipy
from skmultilearn.model_selection import iterative_train_test_split

sys.path.append("../")

from utils.r41 import R41Constants
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R41 = R41Constants()

iternum = 2
liquid_density_threshold = 400 # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "../csv/"
in_csv_name = "r41-density-iter" + str(iternum) + "-results.csv" 

# Read file
df_csv = pd.read_csv(csv_path + in_csv_name, index_col=0)

def values_real_to_pref(theta_guess):
    """
    Scales real units (nm, kJ/mol) to preferred units (Angstrom and Kelvin)

    Parameters
    ----------
    theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

    Returns
    -------
    theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in K)
    """
    assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"

    if len(theta_guess.shape) == 1:
        theta_guess = theta_guess.reshape(1,-1)

    midpoint = theta_guess.shape[1] // 2
    sigmas = (theta_guess[:, :midpoint]) * float((1.0*u.nm).in_units(u.Angstrom).value)
    epsilons = (theta_guess[:, midpoint:]) / float((1.0 * u.K * u.kb).in_units("kJ/mol"))
    theta_guess = np.hstack((sigmas, epsilons))
    # sigmas = [float((x * u.nm).in_units(u.Angstrom).value) for x in theta_guess[:midpoint]]
    # epsilons = [float(x / (u.K * u.kb).in_units("kJ/mol")) for x in theta_guess[midpoint:]]
    # theta_guess = np.array(sigmas + epsilons)

    if theta_guess.shape[0] == 1:
        theta_guess = theta_guess.flatten()

    return theta_guess

molecule = R41
# Add expt density and is_liquid
df_all = df_csv.rename(columns={"density": "md_density"})
df_all["expt_density"] = df_all["temperature"].apply(
    lambda temp: molecule.expt_liq_density[int(temp)]
)
df_all["is_liquid"] = df_all["md_density"].apply(
    lambda x: x > liquid_density_threshold
)

# df_liquid2 = df_all[df_all["is_liquid"] == True]
# scaled_param_values = values_real_to_scaled(
#         df_liquid2[list(molecule.param_names)], molecule.param_bounds
#     )
# df_liquid2[list(molecule.param_names)] = scaled_param_values
# column_names = list(R41.param_names)
# #Drop the last four columns
# df_liquid2 = df_liquid2.drop(columns=["md_density", "expt_density", "is_liquid", "temperature"])
# #Remove duplocate rows
# df_liquid2 = df_liquid2.drop_duplicates()
# g = seaborn.pairplot(df_liquid2)
# g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
# g.savefig("liq_samples_scl.pdf")

#Scale sigmas and epsilons to preferred units
# Extract the columns to be transformed
theta_guess = df_all[list(R41.param_names)].to_numpy()

# Apply the transformation
transformed_values = values_real_to_pref(theta_guess)

# Update the DataFrame with the transformed values
df_all[list(R41.param_names)] = transformed_values

# Split out vapor and liquid samples
df_liquid = df_all[df_all["is_liquid"] == True]
df_vapor = df_all[df_all["is_liquid"] == False]

# print(df_liquid.head())
out_csv_name = "r41-density-iter" + str(iternum) + "-liq_samp.csv"
df_liquid.to_csv(csv_path + out_csv_name)

#Drop the last four columns
df_liquid = df_liquid.drop(columns=["md_density", "expt_density", "is_liquid", "temperature"])
#Remove duplocate rows
df_liquid = df_liquid.drop_duplicates()
column_names = list(R41.param_names)
g = seaborn.pairplot(df_liquid)
# g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("liq_samples.pdf")


#Drop the last four columns
df_vapor = df_vapor.drop(columns=["md_density", "expt_density", "is_liquid", "temperature"])
#Remove duplocate rows
df_vapor = df_vapor.drop_duplicates()
g = seaborn.pairplot(df_vapor)
# g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("vap_samples.pdf")
