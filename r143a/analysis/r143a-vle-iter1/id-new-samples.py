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
############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1
gp_shuffle_seed = 584745
##############################################################################
##############################################################################

md_gp_shuffle_seed = 1
distance_seed = 10
liquid_density_threshold = 500  # kg/m^3


# Read VLE files
csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/analysis/csv/"
in_csv_names = [
    "r143a-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]
out_csv_name = "r143a-vle-iter" + str(iternum + 1) + "-params.csv"
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_vle = prepare_df_vle(df_csv, R143a)

# Read liquid density files
max_density_iter = 4
in_csv_names = [
    "r143a-density-iter" + str(i) + "-results.csv"
    for i in range(1, max_density_iter + 1)
]
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R143a, liquid_density_threshold
)

### Fit GP models to VLE data
# Create training/test set
param_names = list(R143a.param_names) + ["temperature"]
property_names = ["sim_liq_density", "sim_vap_density", "sim_Pvap", "sim_Hvap"]

vle_models = {}
for property_name in property_names:
    # Get train/test
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_vle, param_names, property_name, shuffle_seed=gp_shuffle_seed
    )

    # Fit model
    vle_models[property_name] = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.RBF(lengthscales=np.ones(R143a.n_params + 1)),
    )

# For vapor density replace with Matern52 kernel
property_name = "sim_vap_density"
# Get train/test
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_vle, param_names, property_name, shuffle_seed=gp_shuffle_seed
)
# Fit model
vle_models[property_name] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R143a.n_params + 1)),
)

### Fit GP models to liquid density data
# Get train/test
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_liquid, param_names, property_name, shuffle_seed=md_gp_shuffle_seed
)

# Fit model
md_model = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R143a.n_params + 1)),
)


# Get difference between GROMACS/Cassandra density
df_test_points = df_vle[
    list(R143a.param_names) + ["temperature", "sim_liq_density"]
]
xx = df_test_points[list(R143a.param_names) + ["temperature"]].values
means, vars_ = md_model.predict_f(xx)
diff = values_scaled_to_real(
    df_test_points["sim_liq_density"].values.reshape(-1, 1),
    R143a.liq_density_bounds,
) - values_scaled_to_real(means, R143a.liq_density_bounds)
print(
    f"The average density difference between Cassandra and GROMACS is {np.mean(diff)} kg/m^3"
)
print(
    f"The minimum density difference between Cassandra and GROMACS is {np.min(diff)} kg/m^3"
)
print(
    f"The maximum density difference between Cassandra and GROMACS is {np.max(diff)} kg/m^3"
)


### Step 3: Find new parameters for simulations
max_mse = 625  # kg^2/m^6
latin_hypercube = np.loadtxt("LHS_500000_x_8.csv", delimiter=",")
ranked_samples = rank_samples( # compare and downselect with MD density results first
    latin_hypercube, md_model, R143a, "sim_liq_density", property_offset=13.5
)
print("Ranking samples complete!")
viable_samples = ranked_samples[ranked_samples["mse"] < max_mse].drop(
    columns="mse"
)
viable_samples = viable_samples.values
print(
    "There are:",
    viable_samples.shape[0],
    "viable parameter sets which are within 25 kg/m$^2$ of GROMACS liquid densities",
)

# Calculate other properties
vle_predicted_mses = {}
for property_name, model in vle_models.items():
    vle_predicted_mses[property_name] = rank_samples(
        viable_samples, model, R143a, property_name
    )
print("Completed calculating other properties!")

# Merge into single DF
vle_mses = vle_predicted_mses["sim_liq_density"].merge(
    vle_predicted_mses["sim_vap_density"], on=R143a.param_names
)
vle_mses = vle_mses.rename(
    {"mse_x": "mse_liq_density", "mse_y": "mse_vap_density"}, axis="columns"
)
vle_mses = vle_mses.merge(vle_predicted_mses["sim_Pvap"], on=R143a.param_names)
vle_mses = vle_mses.merge(vle_predicted_mses["sim_Hvap"], on=R143a.param_names)
vle_mses = vle_mses.rename(
    {"mse_x": "mse_Pvap", "mse_y": "mse_Hvap"}, axis="columns"
)

# Find pareto efficient points
result, pareto_points, dominated_points = find_pareto_set(
    vle_mses.drop(columns=list(R143a.param_names)).values, is_pareto_efficient
)
vle_mses = vle_mses.join(pd.DataFrame(result, columns=["is_pareto"]))

vle_mses.to_csv(csv_path + "vle_mses.csv")

# Plot pareto points vs. MSEs
g = seaborn.pairplot(
    vle_mses,
    vars=["mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap"],
    hue="is_pareto",
)
g.savefig("figs/R143a-pareto-mses.pdf")

# Plot pareto points vs. params
g = seaborn.pairplot(vle_mses, vars=list(R143a.param_names), hue="is_pareto")
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("figs/R143a-pareto-params.pdf")

'''# Find new points for next iteration
pareto_points = vle_mses[vle_mses["is_pareto"] == True]
print(f"A total of {len(pareto_points)} pareto efficient points were found.")
new_points = pareto_points.sort_values("mse_liq_density").iloc[[0]]
new_points = new_points.append(
    pareto_points.sort_values("mse_vap_density").iloc[[0]]
)
new_points = new_points.append(pareto_points.sort_values("mse_Hvap").iloc[[0]])
new_points = new_points.append(pareto_points.sort_values("mse_Pvap").iloc[[0]])
pareto_points.drop(index=new_points.index, inplace=True)

# Search to ID well spaced points
dist_guess = 0.9
dist_seed = 10
target_num = 25
args = (new_points, R143a, target_num, dist_seed)
solution = optimize.root(opt_dist, dist_guess, args = args, method = "broyden1", options={'maxiter': 50})
dist_opt = solution.x
next_iter_points = opt_dist(dist_opt, new_points, R143a, target_num, rand_seed=dist_seed , eval = True)

while int(len(next_iter_points)) != int(target_num):
    dist_guess = dist_opt
    solution = optimize.root(opt_dist, dist_guess, args = args, method = "broyden1", options={'maxiter': 50})
    dist_opt = solution.x
    next_iter_points = opt_dist(dist_opt, new_points, R143a, target_num, rand_seed=dist_seed , eval = True)
    dist_seed += 1
    print("Trying seed", dist_seed)'''
   
'''distance = 2.14
discarded_points = pd.DataFrame(columns=pareto_points.columns)

np.random.seed(distance_seed)

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
    )'''
'''print(
    f"After removing similar points, we are left with {len(next_iter_points)} pareto efficient points."
)


# Plot new points
g = seaborn.pairplot(next_iter_points, vars=list(R143a.param_names))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("figs/R143a-new-points.pdf")

# Save to CSV
next_iter_points.drop(
    columns=[
        "mse_liq_density",
        "mse_vap_density",
        "mse_Pvap",
        "mse_Hvap",
        "is_pareto",
    ],
    inplace=True,
)
next_iter_points.to_csv(csv_path + out_csv_name)'''
