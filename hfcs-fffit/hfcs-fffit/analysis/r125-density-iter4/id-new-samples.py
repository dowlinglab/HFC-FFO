#!/usr/bin/env python
# coding: utf-8

import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes

from sklearn import svm

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import silhouette_score


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

sys.path.append("../")

from utils.r125 import R125Constants
from utils.id_new_samples import prepare_df_density

R125 = R125Constants()

liquid_density_threshold = 500  # kg/m^3

iternum = 4

csv_path = "/scratch365/rdefever/hfcs-fffit/hfcs-fffit/analysis/csv/"


# Load in all parameter csvs and result csvs
param_csv_names = [
    "r125-density-iter" + str(i) + "-params.csv" for i in range(1, iternum + 1)
]
result_csv_names = [
    "r125-density-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]
df_params = [
    pd.read_csv(csv_path + param_csv_name, index_col=0)
    for param_csv_name in param_csv_names
]
df_results = [
    pd.read_csv(csv_path + result_csv_name, index_col=0)
    for result_csv_name in result_csv_names
]

# Concatenate all parameter sets and results
df_params = pd.concat(df_params).reset_index(drop=True)
df_results = pd.concat(df_results).reset_index(drop=True)

# Create a df with the MSE for each parameter set
# and add the parameter set idx
df_results["expt_density"] = df_results["temperature"].apply(
    lambda x: R125.expt_liq_density[int(x)]
)
df_results["sq_err"] = (df_results["density"] - df_results["expt_density"]) ** 2
df_mse = (
    df_results.groupby(list(R125.param_names))["sq_err"].mean().reset_index(name="mse")
)

scaled_param_values = values_real_to_scaled(
    df_mse[list(R125.param_names)], R125.param_bounds
)
param_idxs = []
param_vals = []
for params1 in scaled_param_values:
    for idx, params2 in enumerate(df_params[list(R125.param_names)].values):
        if np.allclose(params1, params2):
            param_idxs.append(idx)
            param_vals.append(params2)
            break
df_mse["param_idx"] = param_idxs
df_mse[list(R125.param_names)] = param_vals

# Plot all with MSE < 625
g = seaborn.pairplot(
    pd.DataFrame(df_mse[df_mse["mse"] < 625.0], columns=list(R125.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R125-all-MSE.lt.625.pdf")
# Plot all with MSE < 100
g = seaborn.pairplot(
    pd.DataFrame(df_mse[df_mse["mse"] < 100.0], columns=list(R125.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R125-all-MSE.lt.100.pdf")

top_param_set = df_mse[df_mse["mse"] < 625]
# Greedy search to ID top params
distance = 2.3
final_param_set = pd.DataFrame(columns=top_param_set.columns)
not_final_param_set = pd.DataFrame(columns=top_param_set.columns)

while len(top_param_set > 0):
    top_param_set = top_param_set.sort_values("mse")
    final_param_set = final_param_set.append(top_param_set.iloc[[0]])
    # Remove anything within distance
    l1_norm = np.sum(
        np.abs(
            top_param_set[list(R125.param_names)].values
            - final_param_set[list(R125.param_names)].iloc[[-1]].values
        ),
        axis=1,
    )
    points_to_remove = np.where(l1_norm < distance)[0]
    not_final_param_set = not_final_param_set.append(
        top_param_set.iloc[points_to_remove]
    )
    top_param_set.drop(index=top_param_set.index[points_to_remove], inplace=True)
final_param_set_mse625 = final_param_set

top_param_set = df_mse[df_mse["mse"] < 100]
# Greedy search to ID top params
distance = 2.13
final_param_set = pd.DataFrame(columns=top_param_set.columns)
not_final_param_set = pd.DataFrame(columns=top_param_set.columns)

while len(top_param_set > 0):
    top_param_set = top_param_set.sort_values("mse")
    final_param_set = final_param_set.append(top_param_set.iloc[[0]])
    # Remove anything within distance
    l1_norm = np.sum(
        np.abs(
            top_param_set[list(R125.param_names)].values
            - final_param_set[list(R125.param_names)].iloc[[-1]].values
        ),
        axis=1,
    )
    points_to_remove = np.where(l1_norm < distance)[0]
    not_final_param_set = not_final_param_set.append(
        top_param_set.iloc[points_to_remove]
    )
    top_param_set.drop(index=top_param_set.index[points_to_remove], inplace=True)
final_param_set_mse100 = final_param_set

g = seaborn.pairplot(
    pd.DataFrame(final_param_set_mse625, columns=list(R125.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R125-greedy-MSE.lt.625.pdf")

g = seaborn.pairplot(
    pd.DataFrame(final_param_set_mse100, columns=list(R125.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R125-greedy-MSE.lt.100.pdf")

final_param_set_mse100.drop(columns=["mse"], inplace=True)
final_param_set_mse100.drop(columns=["param_idx"], inplace=True)
final_param_set_mse100.to_csv(csv_path + "r125-vle-iter1-params.csv")
