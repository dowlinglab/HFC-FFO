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

from fffit.plot import(
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

from fffit.models import run_gpflow_scipy

sys.path.append('../')

from utils.r125 import R125Constants
from utils.id_new_samples import prepare_df_density

R125 = R125Constants()

liquid_density_threshold = 500 # kg/m^3

iternum=4

csv_path = "/scratch365/rdefever/hfcs-fffit/hfcs-fffit/analysis/csv/"


# Load in all parameter csvs and result csvs
param_csv_names = ["r125-density-iter" + str(i) + "-params.csv" for i in range(1, iternum+1)]
result_csv_names = ["r125-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
df_params = [pd.read_csv(csv_path + param_csv_name, index_col=0) for param_csv_name in param_csv_names]
df_results = [pd.read_csv(csv_path + result_csv_name, index_col=0) for result_csv_name in result_csv_names]

# Create a df with the MSE for each parameter set
# and add the parameter set idx
df_mses = []
for iter_ in range(1,iternum+1):
    df_param = df_params[iter_ - 1]
    df_result = df_results[iter_ - 1]

    df_result["expt_density"] = df_result["temperature"].apply(lambda x: R125.expt_liq_density[int(x)])
    df_result["sq_err"] = (df_result["density"] - df_result["expt_density"])**2
    df_mse = df_result.groupby(list(R125.param_names))["sq_err"].mean().reset_index(name="mse")

    scaled_param_values = values_real_to_scaled(df_mse[list(R125.param_names)], R125.param_bounds)
    param_idxs = []
    for params1 in scaled_param_values:
        for idx, params2 in enumerate(df_param[list(R125.param_names)].values):
            if np.allclose(params1, params2):
                param_idxs.append(idx)
                break
    df_mse["param_idx"] = param_idxs
    df_mses.append(df_mse)

# Plot all results sorted by MSE
for i, df_mse in enumerate(df_mses):
    plt.plot(
        range(200),
        df_mse.sort_values("mse")["mse"],
        label=f"Round {i+1}"
    )

plt.xlabel("Sorted Index")
plt.ylabel("MSE (kg$^2$/m$^6$)")
plt.legend()
plt.tight_layout()
plt.savefig("R125-density-MSE-sorted-all.pdf")
plt.close()

# Plot results according to parameter set idx
for i, df_mse in enumerate(df_mses):
    fig, parent_ax = plt.subplots()
    inset_ax = inset_axes(
        parent_ax, width="55%", height="40%", loc="upper right",
         bbox_to_anchor=(-0.03,-0.03,1,1), bbox_transform=parent_ax.transAxes
    )
    parent_ax.plot(
        df_mse.sort_values("param_idx")["param_idx"],
        df_mse.sort_values("param_idx")["mse"],
        label=f"Round {i+1}",
        color=f"C{i+1}"
    )
    inset_ax.plot(
        df_mse.sort_values("param_idx")["param_idx"],
        df_mse.sort_values("param_idx")["mse"],
        label=f"Round {i+1}",
        color=f"C{i+1}"
    )
    parent_ax.set_xlabel("Parameter set index", fontsize=16)
    parent_ax.set_ylabel("MSE (kg$^2$/m$^6$)", fontsize=16)

    inset_ax.set_ylim(-200,1000)
    parent_ax.set_ylim(0,1100000)
    parent_ax.legend(loc="upper left")

    fig.savefig(f"R125-density-MSE-unsorted-round-{i+1}.pdf")
    plt.close()

# Plot results according to parameter set idx
for i, df_mse in enumerate(df_mses):
    plt.plot(
        df_mse.sort_values("param_idx")["param_idx"],
        df_mse.sort_values("param_idx")["mse"],
        label=f"Round {i+1}",
        color=f"C{i+1}"
    )
    plt.xlabel("Parameter set index", fontsize=16)
    plt.ylabel("MSE (kg$^2$/m$^6$)", fontsize=16)
    plt.ylim(0,1100000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"R125-density-MSE-unsorted-round-{i+1}.pdf")
    plt.close()

# Plot results sorted by MSE
# Separate into liquid and vapor
for i, df_mse in enumerate(df_mses):
    prev_plt = plt.plot(
        range(100),
        df_mse.sort_values("param_idx")[:100].sort_values("mse")["mse"],
        label=f"Round {i+1}"
    )
    plt.plot(
        range(100),
        df_mse.sort_values("param_idx")[100:].sort_values("mse")["mse"],
        '--',
        color=prev_plt[0]._color,
    )

plt.xlabel("Sorted Index")
plt.ylabel("MSE (kg$^2$/m$^6$)")
plt.legend()
plt.tight_layout()
plt.savefig("R125-density-MSE-sorted.pdf")
plt.close()

# Zoom in more
for i, df_mse in enumerate(df_mses):
    prev_plt = plt.plot(
        range(100),
        df_mse.sort_values("param_idx")[:100].sort_values("mse")["mse"],
        label=f"Round {i+1}"
    )
    plt.plot(
        range(100),
        df_mse.sort_values("param_idx")[100:].sort_values("mse")["mse"],
        '--',
        color=prev_plt[0]._color,
    )

plt.xlabel("Sorted Index")
plt.ylabel("MSE (kg$^2$/m$^6$)")
plt.ylim(-200,1000)
plt.legend()
plt.tight_layout()
plt.savefig("R125-density-MSE-sorted-inset.pdf")
plt.close()
