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

from utils.r143a import R143aConstants
from utils.id_new_samples import (
    prepare_df_density,
    prepare_df_vle,
    rank_samples,
)

R143a = R143aConstants()

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 2
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

def bisection(lower_bound, upper_bound, error_tol, top_samples, constants, target_num, rand_seed = None): 
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
    assert lower_bound > 0, "Lower bound must be greater than 0"
    assert lower_bound < upper_bound, "Lower bound must be less than the upper bound"
    
    # check if lower_bound and upper_bound bound a root
#     print("Low B", lower_bound)
#     print("High B", upper_bound)
    eval_lower_bound = opt_dist(lower_bound, top_samples, constants, target_num, rand_seed)
#     print("Low Eval",eval_lower_bound )
    eval_upper_bound = opt_dist(upper_bound, top_samples, constants, target_num, rand_seed)
#     print("High Eval",eval_upper_bound )
    if np.sign(eval_lower_bound) == np.sign(eval_upper_bound):
        raise Exception(
         "Increase length of upper bound. Given bounds do not include the root!")
        
    # get midpoint
    midpoint = (lower_bound + upper_bound)/2
    eval_midpoint = opt_dist(midpoint, top_samples, constants, target_num, rand_seed)
#     print("Mid", eval_midpoint)
    
    
    if np.abs(eval_midpoint) < error_tol:
        # stopping condition, report midpoint as root
        return midpoint
    elif np.sign(eval_lower_bound) == np.sign(eval_midpoint):
        # case where midpoint is an improvement on lower_bound. 
        # Make recursive call with lower_bound = midpoint
        return bisection(midpoint, upper_bound, error_tol, top_samples, constants, target_num, rand_seed)
    elif np.sign(eval_upper_bound) == np.sign(eval_midpoint):
        # case where midpoint is an improvement on upper_bound. 
        # Make recursive call with upper_bound = midpoint
        return bisection(lower_bound, midpoint, error_tol, top_samples, constants, target_num, rand_seed)

dist_seed = 115
target_num = 25

lower_bound = 1e-7
#IL norm between the highest high parameter space, and lowest low parameter space value
upper_bound = 5
error_tol = 1e-3

liquid_density_threshold = 500  # kg/m^3


# Read VLE files
csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/analysis/csv/"
out_csv_name = "r143a-vle-iter" + str(iternum + 1) + "-params.csv"

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

distance_opt = bisection(lower_bound, upper_bound, error_tol, pareto_points, R143a, target_num, dist_seed)
print('\nRequired Distance is : %0.8f' % distance_opt)

new_points = opt_dist(distance_opt, pareto_points, R143a, target_num, rand_seed=dist_seed , eval = True)

'''distance = distance_opt
discarded_points = pd.DataFrame(columns=pareto_points.columns)

np.random.seed(dist_seed)

while len(pareto_points > 0):
    # Shuffle the pareto points
    pareto_points = pareto_points.sample(frac=1)
    new_points = new_points.append(pareto_points.iloc[[0]])
    # Remove anything within distance
    l1_norm = np.sum(
        np.abs(
            pareto_points[list(R143a.param_names)].values
            - new_points[list(R143a.param_names)].iloc[[-1]].values
        ),
        axis=1,
    )
    points_to_remove = np.where(l1_norm < distance)[0]
    discarded_points = discarded_points.append(
        pareto_points.iloc[points_to_remove]
    )
    pareto_points.drop(
        index=pareto_points.index[points_to_remove], inplace=True
    )
print(
    f"After removing similar points, we are left with {len(new_points)} pareto efficient points."
)


# Plot new points
g = seaborn.pairplot(next_iter_points, vars=list(R143a.param_names))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("figs/R143a-new-points.pdf")'''

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
