#!/usr/bin/env python
# coding: utf-8

import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import scipy.optimize as optimize
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

from utils.r143a import R143aConstants
from utils.id_new_samples import prepare_df_density

R143a = R143aConstants()

'''def opt_dist(distance, top_samples, constants, target_num, rand_seed = None, eval = False):
    """
    Calculates the distance between points such that exactly a target number of points are chosen for the next iteration
    
    Parameters:
    -----------
        distance: float, The allowable minimum distance between points
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r143a.R143aConstants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default
        eval: bool, Determines whether error is calculated or new_points is returned
    
    Returns:
        error: float, The squared error between the target value and number of new_points
        OR
        new_points: pandas data frame, a pandas data frame containing the number of points to be used 
    """
    if rand_seed != None:
        np.random.seed(rand_seed)
    new_points = pd.DataFrame()
    discarded_points = pd.DataFrame(columns=top_samples.columns)
    while len(top_samples > 0):
        # Shuffle the pareto points
        top_samples = top_samples.sample(frac=1)
        new_points = new_points.append(top_samples.iloc[[0]])
        # Remove anything within distance
        l1_norm = np.sum(
            np.abs(
                top_samples[list(constants.param_names)].values
                - new_points[list(constants.param_names)].iloc[[-1]].values
            ),
            axis=1,
        )
        points_to_remove = np.where(l1_norm < distance)[0]
        discarded_points = discarded_points.append(
            top_samples.iloc[points_to_remove]
        )
        top_samples.drop(
            index=top_samples.index[points_to_remove], inplace=True
        )
        error = (target_num - len(new_points))**2
    if eval == True:
        return new_points
    else:
        return error'''

def opt_dist(distance, top_samples, constants, target_num, rand_seed = None, eval = False):
    """
    Calculates the distance between points such that exactly a target number of points are chosen for the next iteration
    
    Parameters:
    -----------
        distance: float, The allowable minimum distance between points
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r143a.R143aConstants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default
        eval: bool, Determines whether error is calculated or new_points is returned
    
    Returns:
        error: float, The squared error between the target value and number of new_points
        OR
        new_points: pandas data frame, a pandas data frame containing the number of points to be used 
    """
    if len(top_samples) <= target_num:
        print("Trying dist =", distance)
        
    top_samp0 = top_samples
    if rand_seed != None:
        np.random.seed(rand_seed)
    new_points = pd.DataFrame()
    discarded_points = pd.DataFrame(columns=top_samples.columns)
    while len(top_samples > 0):
        # Shuffle the pareto points
        top_samples = top_samples.sample(frac=1)
        new_points = new_points.append(top_samples.iloc[[0]])
        # Remove anything within distance
        l1_norm = np.sum(
            np.abs(
                top_samples[list(constants.param_names)].values
                - new_points[list(constants.param_names)].iloc[[-1]].values
            ),
            axis=1,
        )
        points_to_remove = np.where(l1_norm < distance)[0]
        discarded_points = discarded_points.append(
            top_samples.iloc[points_to_remove]
        )
        top_samples.drop(
            index=top_samples.index[points_to_remove], inplace=True
        )
    error = target_num - len(new_points)
    
#     print("Error = ",error)
#     return error
    if eval == True:
        return new_points
    else:
        return error

liquid_density_threshold = 500  # kg/m^3

iternum = 4

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/analysis/csv/"


# Load in all parameter csvs and result csvs
param_csv_names = [
    "r143a-density-iter" + str(i) + "-params.csv" for i in range(1, iternum + 1)
]
print(param_csv_names)
result_csv_names = [
    "r143a-density-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
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
    lambda x: R143a.expt_liq_density[int(x)]
)
df_results["sq_err"] = (df_results["density"] - df_results["expt_density"]) ** 2
df_mse = (
    df_results.groupby(list(R143a.param_names))["sq_err"].mean().reset_index(name="mse")
)

scaled_param_values = values_real_to_scaled(
    df_mse[list(R143a.param_names)], R143a.param_bounds
)
param_idxs = []
param_vals = []
for params1 in scaled_param_values:
    for idx, params2 in enumerate(df_params[list(R143a.param_names)].values):
        if np.allclose(params1, params2):
            param_idxs.append(idx)
            param_vals.append(params2)
            break
print(df_mse)
print(param_idxs)
df_mse["param_idx"] = param_idxs
df_mse[list(R143a.param_names)] = param_vals

# Plot all with MSE < 625
g = seaborn.pairplot(
    pd.DataFrame(df_mse[df_mse["mse"] < 625.0], columns=list(R143a.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R143a-all-MSE.lt.625.pdf")
# Plot all with MSE < 100
g = seaborn.pairplot(
    pd.DataFrame(df_mse[df_mse["mse"] < 100.0], columns=list(R143a.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R143a-all-MSE.lt.100.pdf")

top_param_set = df_mse[df_mse["mse"] < 100]

dist_guess = 0.9
dist_seed = 10
target_num = 25
args = (top_param_set,R143a, target_num, dist_seed)
solution = optimize.root(opt_dist, dist_guess, args = args, method = "broyden1", options={'maxiter': 50})
dist_opt = solution.x
new_points = opt_dist(dist_opt, top_param_set, R143a, target_num, rand_seed=dist_seed , eval = True)

while int(len(new_points)) != int(target_num):
    dist_guess = dist_opt
    solution = optimize.root(opt_dist, dist_guess, args = args, method = "broyden1", options={'maxiter': 50})
    dist_opt = solution.x
    new_points = opt_dist(dist_opt, top_param_set, R143a, target_num, rand_seed=dist_seed , eval = True)
    dist_seed += 1
    print("Trying seed", dist_seed)
   
print(len(new_points), "final points are left after removing similar points using a distance of", np.round(dist_opt,5))

'''top_param_set = df_mse[df_mse["mse"] < 625]
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
            top_param_set[list(R143a.param_names)].values
            - final_param_set[list(R143a.param_names)].iloc[[-1]].values
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
            top_param_set[list(R143a.param_names)].values
            - final_param_set[list(R143a.param_names)].iloc[[-1]].values
        ),
        axis=1,
    )
    points_to_remove = np.where(l1_norm < distance)[0]
    not_final_param_set = not_final_param_set.append(
        top_param_set.iloc[points_to_remove]
    )
    top_param_set.drop(index=top_param_set.index[points_to_remove], inplace=True)'''

final_param_set_mse100 = new_points
print(final_param_set_mse100)
print("we found",len(final_param_set_mse100),"points")

'''g = seaborn.pairplot(
    pd.DataFrame(final_param_set_mse625, columns=list(R143a.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R143a-greedy-MSE.lt.625.pdf")'''

g = seaborn.pairplot(
    pd.DataFrame(final_param_set_mse100, columns=list(R143a.param_names))
)
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("R143a-greedy-MSE.lt.100.pdf")

final_param_set_mse100.drop(columns=["mse"], inplace=True)
final_param_set_mse100.drop(columns=["param_idx"], inplace=True)
#final_param_set_mse100.to_csv(csv_path + "r143a-vle-iter1-params.csv")
