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

from utils.r143a import R143aConstants
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R143a = R143aConstants()

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
        return error

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 4
cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

##############################################################################
##############################################################################

liquid_density_threshold = 500  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/analysis/csv/"
in_csv_names = ["r143a-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
out_csv_name = "r143a-density-iter" + str(iternum + 1) + "-params.csv"
out_top_liquid_csv_name = "r143a-density-iter" + str(iternum ) + "-liquid-params.csv"
out_top_vapor_csv_name = "r143a-density-iter" + str(iternum ) + "-vapor-params.csv"

# Read file
df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R143a, liquid_density_threshold
)

### Step 2: Fit classifier and GP models

# Create training/test set
param_names = list(R143a.param_names) + ["temperature"]
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
param_names = list(R143a.param_names) + ["temperature"]
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
)

# Fit model
model = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R143a.n_params + 1)),
)


### Step 3: Find new parameters for MD simulations

# SVM to classify hypercube regions as liquid or vapor
latin_hypercube = np.loadtxt("LHS_500000_x_8.csv", delimiter=",")
liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)
# Find the lowest MSE points from the GP in both sets
ranked_liquid_samples = rank_samples(liquid_samples, model, R143a, "sim_liq_density")
ranked_vapor_samples = rank_samples(vapor_samples, model, R143a, "sim_liq_density")#both l and g compared to liquid density

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
column_names = list(R143a.param_names)
g = seaborn.pairplot(top_liquid_samples.drop(columns=["mse"]))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("liq_mse_below625.pdf")

# Create a pairplot of the top "vapor" parameter values
column_names = list(R143a.param_names)
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
#top_liq = pd.read_csv("../csv/r143a-density-iter2-liquid-params.csv", delimiter = ",", index_col = 0)
#top_vap = pd.read_csv("../csv/r143a-density-iter2-vapor-params.csv", delimiter = ",", index_col = 0)

top_liq = top_liq.reset_index(drop=True)
top_vap = top_vap.reset_index(drop=True)

dist_guess = 1
dist_seed = 10
bounds = [(0,None)]
target_num = 100
args_v = (top_vap ,R143a, target_num, dist_seed)
solution_v = optimize.minimize(opt_dist, dist_guess, bounds = bounds, args=args_v, method='Nelder-Mead')
dist_opt_v = solution_v.x
new_points_v = opt_dist(dist_opt_v, top_vap, R143a, target_num, rand_seed=dist_seed , eval = True)

while len(new_points_v) != target_num:
    dist_opt_v = solution_v.x
    dist_seed += 1
    new_points_v = opt_dist(dist_opt_v, top_vap, R143a, target_num, rand_seed=dist_seed , eval = True)
    
print(len(new_points_v), "top vapor density points are left after removing similar points using a distance of", np.round(dist_opt_v,5))

args_l = (top_liq ,R143a, target_num, dist_seed)
solution_l = optimize.minimize(opt_dist, dist_guess, bounds = bounds, args=args_l, method='Nelder-Mead')
dist_opt_l = solution_l.x
new_points_l = opt_dist(dist_opt_l, top_liq, R143a, target_num, rand_seed=dist_seed , eval = True)

while len(new_points_l) != target_num:
    dist_opt_l = solution_l.x
    dist_seed += 1
    new_points_l = opt_dist(dist_opt_l, top_liq, R143a, target_num, rand_seed=dist_seed , eval = True)
    
print(len(new_points_l), "top liquid density points are left after removing similar points using a distance of", np.round(dist_opt_l,5))

pd.concat([new_points_l,new_points_v], axis=0).to_csv(csv_path + out_csv_name)

'''# Search to ID well spaced points
# Top Liquid density
distance = 2.14
discarded_points = pd.DataFrame(columns=top_liquid_samples.columns)

np.random.seed(distance_seed)

while len(top_liquid_samples > 0):
    # Shuffle the liquid points
    liquid_points = top_liquid_samples.sample(frac=1)
    new_liquid_points = new_liquid_points.append(liquid_points.iloc[[0]])
    # Remove anything within distance
    l1_norm = np.sum(
        np.abs(
            liquid_points[list(R143a.param_names)].values
            - new_liquid_points[list(R143a.param_names)].iloc[[-1]].values
        ),
        axis=1,
    )
    points_to_remove = np.where(l1_norm < distance)[0]
    discarded_points = discarded_points.append(
        liquid_points.iloc[points_to_remove]
    )
    liquid_points.drop(
        index=liquid_points.index[points_to_remove], inplace=True
    )
print(
    f"After removing similar points, we are left with {len(new_liquid_points)} top liquid samples."
)

# Top Vapor density
distance = 2.14
discarded_points = pd.DataFrame(columns=top_vapor_samples.columns)

np.random.seed(distance_seed)

while len(top_vapor_samples > 0):
    # Shuffle the liquid points
    vapor_points = top_vapor_samples.sample(frac=1)
    new_vapor_points = new_vapor_points.append(vapor_points.iloc[[0]])
    # Remove anything within distance
    l1_norm = np.sum(
        np.abs(
            vapor_points[list(R143a.param_names)].values
            - new_vapor_points[list(R143a.param_names)].iloc[[-1]].values
        ),
        axis=1,
    )
    points_to_remove = np.where(l1_norm < distance)[0]
    discarded_points = discarded_points.append(
        vapor_points.iloc[points_to_remove]
    )
    vapor_points.drop(
        index=vapor_points.index[points_to_remove], inplace=True
    )
print(
    f"After removing similar points, we are left with {len(new_vapor_points)} top vapor samples."
)
# Plot new points
g = seaborn.pairplot(new_points, vars=list(R125.param_names))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("figs/R125-new-points.pdf")

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

#### Combine top 100 lowest MSE for parameter sets predicted as liquid and vapor

new_params = [
    ranked_liquid_samples.drop(columns=["mse"])[:100],
    ranked_vapor_samples.drop(columns=["mse"])[:100],
]

# Concatenate into a single dataframe and save to CSV
new_params = pd.concat(new_params)
new_params.to_csv(csv_path + out_csv_name)

# Create a pairplot of the top 100 "liquid" parameter values
column_names = list(R143a.param_names)
g = seaborn.pairplot(ranked_liquid_samples.drop(columns=["mse"])[:100])
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("liq_mse_top100.pdf")

# Create a pairplot of the top 100 "vapor" parameter values
column_names = list(R143a.param_names)
g = seaborn.pairplot(ranked_vapor_samples.drop(columns=["mse"])[:100])
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("vap_mse_top100.pdf")i'''
