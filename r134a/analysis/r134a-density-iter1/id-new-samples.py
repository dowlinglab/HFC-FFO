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

from utils.r134a import R134aConstants
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R134a = R134aConstants()

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
############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1
cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

##############################################################################
##############################################################################

liquid_density_threshold = 500  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r134a/analysis/csv/"
in_csv_names = ["r134a-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
out_csv_name = "r134a-density-iter" + str(iternum + 1) + "-params.csv"
out_top_liquid_csv_name = "r134a-density-iter" + str(iternum ) + "-liquid-params.csv"
out_top_vapor_csv_name = "r134a-density-iter" + str(iternum ) + "-vapor-params.csv"

# Read file
df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R134a, liquid_density_threshold
)

### Step 2: Fit classifier and GP models

# Create training/test set
param_names = list(R134a.param_names) + ["temperature"]
property_name = "is_liquid"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifer is {test_score*100.0}% accurate on the test set.")
plot_confusion_matrix(classifier, x_test, y_test)  
plt.savefig("classifier.pdf")


### Fit GP Model
# Create training/test set
param_names = list(R134a.param_names) + ["temperature"]
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
)

# Fit model
model = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R134a.n_params + 1)),
)


### Step 3: Find new parameters for MD simulations

# SVM to classify hypercube regions as liquid or vapor
latin_hypercube = np.loadtxt("LHS_500000_x_10.csv", delimiter=",")
liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)
# Find the lowest MSE points from the GP in both sets
ranked_liquid_samples = rank_samples(liquid_samples, model, R134a, "sim_liq_density")
ranked_vapor_samples = rank_samples(vapor_samples, model, R134a, "sim_liq_density")#both l and g compared to liquid density

# Make a set of the lowest MSE parameter sets
top_liquid_samples = ranked_liquid_samples[
    ranked_liquid_samples["mse"] < 625.0
]
top_vapor_samples = ranked_vapor_samples[ranked_vapor_samples["mse"] < 625.0]
print(
    "There are:",
    top_liquid_samples.shape[0],
    "liquid parameter sets which produce densities within 25 kg/m$^2$ of experimental densities",
)
print(
    "There are:",
    top_vapor_samples.shape[0],
    " vapor parameter sets which produce densities within 25 kg/m$^2$ of experimental densities",
)

#### Visualization: Low MSE parameter sets
# Create a pairplot of the top "liquid" parameter values
column_names = list(R134a.param_names)
g = seaborn.pairplot(top_liquid_samples.drop(columns=["mse"]))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("liq_mse_below625.pdf")

# Create a pairplot of the top "vapor" parameter values
column_names = list(R134a.param_names)
g = seaborn.pairplot(top_vapor_samples.drop(columns=["mse"]))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("vap_mse_below625.pdf")

new_liquid_params = [
    top_liquid_samples.drop(columns=["mse"])
]
new_vapor_params = [
    top_vapor_samples.drop(columns=["mse"])
]

# Concatenate into a single dataframe and save to CSV
new_liquid_params = pd.concat(new_liquid_params)
new_liquid_params.to_csv(csv_path + out_top_liquid_csv_name)
new_vapor_params = pd.concat(new_vapor_params)
new_vapor_params.to_csv(csv_path + out_top_vapor_csv_name)
top_liq = pd.read_csv(csv_path + out_top_liquid_csv_name, delimiter = ",", index_col = 0)
top_vap = pd.read_csv(csv_path + out_top_vapor_csv_name, delimiter = ",", index_col = 0)

top_liq = top_liq.reset_index(drop=True)
top_vap = top_vap.reset_index(drop=True)

from numpy.linalg import norm
dist_seed = 115
target_num_v = 43
target_num_l = 157 # just for liquid; vapor less than 100 and use all

zero_array = np.zeros(top_liq.shape[1])
one_array = np.ones(top_liq.shape[1])
ub_array = one_array - zero_array

# lower_bound = 1e-8
lower_bound = 0
#IL norm between the highest high parameter space, and lowest low parameter space value
upper_bound = norm(ub_array, 1) # This number will be 10, the number of dimensions
error_tol = 1e-5

distance_opt_v,number_points_v = bisection(lower_bound, upper_bound, error_tol, top_vap, R134a, target_num_v, dist_seed)
print('\nRequired Distance for vapor is : %0.8f and there are %0.1f points too few' % (distance_opt_v, number_points_v) )

distance_opt_l,number_points_l = bisection(lower_bound, upper_bound, error_tol, top_liq, R134a, target_num_l, dist_seed)
print('\nRequired Distance for liquid is : %0.8f and there are %0.1f points too few' % (distance_opt_l, number_points_l) )

new_points_l = opt_dist(distance_opt_l, top_liq, R134a, target_num_l, rand_seed=dist_seed , eval = True)
    
print(len(new_points_l), "top liquid density points are left after removing similar points using a distance of", np.round(distance_opt_l,5))

new_points_v = opt_dist(distance_opt_v, top_vap, R134a, target_num_v, rand_seed=dist_seed , eval = True)
    
print(len(new_points_v), "top liquid density points are left after removing similar points using a distance of", np.round(distance_opt_v,5))

pd.concat([new_points_l,new_points_v], axis=0).to_csv(csv_path + out_csv_name)
#pd.concat([new_points_l,top_vap], axis=0).to_csv(csv_path + out_csv_name)

