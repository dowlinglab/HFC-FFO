import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import plot_confusion_matrix

from sklearn import svm
import scipy.optimize as optimize

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

from utils.r50 import R50Constants
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R50 = R50Constants()

def opt_dist(distance, top_samples, constants, target_num, rand_seed = None, eval = False):
    """
    Calculates the distance between points such that exactly a target number of points are chosen for the next iteration
    
    Parameters:
    -----------
        distance: float, The allowable minimum distance between points
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r134a.R134aConstants, contains the infromation for a certain refrigerant
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
        points_to_remove = np.where(l1_norm <= distance)[0] #Changed to <= to get zero bc to work
        discarded_points = discarded_points.append(
            top_samples.iloc[points_to_remove]
        )
        top_samples.drop(
            index=top_samples.index[points_to_remove], inplace=True
        )
#     error = target_num - len(new_points)
    len_new_points = len(new_points)
    
#     print("Error = ",error)
#     return error
    if eval == True:
        return new_points
    else:
#         return error
        return len_new_points


def bisection(lower_bound, upper_bound, error_tol, top_samples, constants, target_num, rand_seed = None, verbose = False):
    """
    approximates a root of a function bounded by lower_bound and upper_bound to within a tolerance 
    
    Parameters:
    -----------
        lower_bound: float, lower bound of the distance, must be > 0
        upper_bound: float, lower bound of the distance, must be > lower_bound
        error_tol: floar, tolerance of error
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r134a.R134aConstants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default
        
    Returns:
    --------
        midpoint: The distance that satisfies the error criteria based on the target number

    """
    assert len(top_samples) > target_num, "Ensure you have more samples than the target number!"
    #Initialize Termination criteria and add assert statements
    assert lower_bound >= 0, "Lower bound must be greater than 0"
    assert lower_bound < upper_bound, "Lower bound must be less than the upper bound"
    
        #Set error of upper and lower bound
#         print("Low B", lower_bound)
#         print("High B", upper_bound)
    eval_lower_bound = opt_dist(lower_bound, top_samples, constants, target_num, rand_seed)
    eval_upper_bound = opt_dist(upper_bound, top_samples, constants, target_num, rand_seed)
#     print("Low Eval",eval_lower_bound )
#     print("High Eval",eval_upper_bound )
    
    #Throw Error if initial guesses are bad
    if not eval_lower_bound > target_num > eval_upper_bound:
        print("Increase Length of Upper Bound. Given bounds do not include the root!")

    terminate = False
    
    #While error > tolerance
    while terminate == False:
        #Find the midpoint and evaluate it    
        midpoint = (lower_bound + upper_bound)/2
#         print("Mid B", midpoint)
        eval_midpoint = opt_dist(midpoint, top_samples, constants, target_num, rand_seed)
#         print("Mid Eval", eval_midpoint)
        error =  target_num - eval_midpoint   
        if verbose == True:
            print('distance = %0.6f and error = %0.6f' % (midpoint, error))
        
        # Set the upper or lower bound depending on sign
        if eval_midpoint == target_num:
            #Terminate loop if error < error_tol
            terminate = abs(eval_midpoint) < error_tol 
            break      
        elif  eval_midpoint < target_num:
            upper_bound = midpoint
        else:
            lower_bound = midpoint
        if (upper_bound - lower_bound) < error_tol:
            print("Bounds have collapsed and the number of points is", eval_midpoint)
            break
        
 
    return midpoint, error
############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 4
cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

##############################################################################
##############################################################################

liquid_density_threshold = 200  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r50/analysis/csv/"

# Load in all parameter csvs and result csvs
param_csv_names = [
    "r50-density-iter" + str(i) + "-params.csv" for i in range(1, iternum + 1)
]
print(param_csv_names)
result_csv_names = [
    "r50-density-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
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
    lambda x: R50.expt_liq_density[int(x)]
)
df_results["sq_err"] = (df_results["density"] - df_results["expt_density"]) ** 2
df_mse = (
    df_results.groupby(list(R50.param_names))["sq_err"].mean().reset_index(name="mse")
)
scaled_param_values = values_real_to_scaled(
    df_mse[list(R50.param_names)], R50.param_bounds
)
param_idxs = []
param_vals = []
for params1 in scaled_param_values:
    for idx, params2 in enumerate(df_params[list(R50.param_names)].values):
        if np.allclose(params1, params2):
            param_idxs.append(idx)
            param_vals.append(params2)
            break
df_mse["param_idx"] = param_idxs
df_mse[list(R50.param_names)] = param_vals

top_param_set = df_mse[df_mse["mse"] < 100]
print(top_param_set)
from numpy.linalg import norm
dist_seed = 115
#target_num_v = 43
target_num_l = 25 

zero_array = np.zeros(top_param_set.shape[1])
one_array = np.ones(top_param_set.shape[1])
ub_array = one_array - zero_array

# lower_bound = 1e-8
lower_bound = 0
#IL norm between the highest high parameter space, and lowest low parameter space value
upper_bound = norm(ub_array, 1) # This number will be 10, the number of dimensions
error_tol = 1e-5


distance_opt_l,number_points_l = bisection(lower_bound, upper_bound, error_tol, top_param_set, R50, target_num_l, dist_seed)
print('\nRequired Distance for liquid is : %0.8f and there are %0.1f points too few' % (distance_opt_l, number_points_l) )

new_points_vle = opt_dist(distance_opt_l, top_param_set, R50, target_num_l, rand_seed=dist_seed , eval = True)
    
print(len(new_points_vle), "top liquid density points are left after removing similar points using a distance of", np.round(distance_opt_l,5))

new_points_vle.drop(columns=["mse"], inplace=True)
new_points_vle.drop(columns=["param_idx"], inplace=True)
new_points_vle.to_csv(csv_path + "r50-vle-iter1-params.csv")

