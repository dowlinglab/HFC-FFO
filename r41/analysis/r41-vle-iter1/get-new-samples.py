import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import scipy.optimize as optimize

from fffit.utils import (
    shuffle_and_split,
    values_scaled_to_real,
)


from fffit.models import run_gpflow_scipy
from fffit.pareto import find_pareto_set, is_pareto_efficient

sys.path.append("../")

from utils.r41 import R41Constants
from utils.id_new_samples import (
    prepare_df_density,
    prepare_df_vle,
    rank_samples,
)

R41 = R41Constants()

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1
##############################################################################
##############################################################################
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
        constants: utils.r143a.R143aConstants, contains the infromation for a certain refrigerant
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

liquid_density_threshold = 500  # kg/m^3


# Read VLE files
csv_path = "/scratch365/mcarlozo/HFC-FFO/r41/analysis/csv/"
out_csv_name = "r41-vle-iter" + str(iternum + 1) + "-params.csv"

vle_mses = pd.read_csv(csv_path + "vle_mses.csv",index_col=0)

# Find new points for next iteration
pareto_points = vle_mses[vle_mses["is_pareto"] == True]
print(f"A total of {len(pareto_points)} pareto efficient points were found.")
new_points = pareto_points.sort_values("mse_liq_density").iloc[[0]]
new_points = new_points.append(
    pareto_points.sort_values("mse_vap_density").iloc[[0]]
)
new_points = new_points.append(pareto_points.sort_values("mse_Hvap").iloc[[0]])
new_points = new_points.append(pareto_points.sort_values("mse_Pvap").iloc[[0]])
pareto_points.drop(index=new_points.index, inplace=True)

from numpy.linalg import norm
dist_seed = 115
target_num = 25

zero_array = np.zeros(pareto_points.shape[1])
one_array = np.ones(pareto_points.shape[1])
ub_array = one_array - zero_array

# lower_bound = 1e-8
lower_bound = 0
#IL norm between the highest high parameter space, and lowest low parameter space value
upper_bound = norm(ub_array, 1) # This number will be 8, the number of dimensions
error_tol = 1e-5

distance_opt, number_points = bisection(lower_bound, upper_bound, error_tol, pareto_points, R41, target_num, dist_seed)
print('\nRequired Distance for vapor is : %0.8f and there are %0.1f points too few' % (distance_opt, number_points) )

new_points = opt_dist(distance_opt, pareto_points, R41, target_num, rand_seed=dist_seed , eval = True)

# Save to CSV
new_points.drop(
    columns=[
        "mse_liq_density",
        "mse_vap_density",
        "mse_Pvap",
        "mse_Hvap",
        "is_pareto",
    ],
    inplace=True,
)
new_points.to_csv(csv_path + out_csv_name)
