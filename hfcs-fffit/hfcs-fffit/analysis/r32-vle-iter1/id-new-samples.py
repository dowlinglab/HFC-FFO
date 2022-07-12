import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from fffit.utils import (
    shuffle_and_split,
    values_scaled_to_real,
)


from fffit.models import run_gpflow_scipy
from fffit.pareto import find_pareto_set, is_pareto_efficient

sys.path.append("../")

from utils.r32 import R32Constants
from utils.id_new_samples import (
    prepare_df_density,
    prepare_df_vle,
    rank_samples,
)

R32 = R32Constants()

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1
gp_shuffle_seed = 1948589

##############################################################################
##############################################################################

md_gp_shuffle_seed = 1
distance_seed = 10
liquid_density_threshold = 500  # kg/m^3


# Read VLE files
csv_path = "/scratch365/rdefever/hfcs-fffit/hfcs-fffit/analysis/csv/"
in_csv_names = [
    "r32-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]
out_csv_name = "r32-vle-iter" + str(iternum + 1) + "-params.csv"
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_vle = prepare_df_vle(df_csv, R32)

# Read liquid density files
max_density_iter = 4
in_csv_names = [
    "r32-density-iter" + str(i) + "-results.csv"
    for i in range(1, max_density_iter + 1)
]
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R32, liquid_density_threshold
)

### Fit GP models to VLE data
# Create training/test set
param_names = list(R32.param_names) + ["temperature"]
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
        gpflow.kernels.RBF(lengthscales=np.ones(R32.n_params + 1)),
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
    gpflow.kernels.RBF(lengthscales=np.ones(R32.n_params + 1)),
)


# Get difference between GROMACS/Cassandra density
df_test_points = df_vle[
    list(R32.param_names) + ["temperature", "sim_liq_density"]
]
xx = df_test_points[list(R32.param_names) + ["temperature"]].values
means, vars_ = md_model.predict_f(xx)
diff = values_scaled_to_real(
    df_test_points["sim_liq_density"].values.reshape(-1, 1),
    R32.liq_density_bounds,
) - values_scaled_to_real(means, R32.liq_density_bounds)
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
latin_hypercube = np.loadtxt("LHS_1e6x6.csv", delimiter=",")
ranked_samples = rank_samples(
    latin_hypercube, md_model, R32, "sim_liq_density", property_offset=22.8
)
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
        viable_samples, model, R32, property_name
    )


# Merge into single DF
vle_mses = vle_predicted_mses["sim_liq_density"].merge(
    vle_predicted_mses["sim_vap_density"], on=R32.param_names
)
vle_mses = vle_mses.rename(
    {"mse_x": "mse_liq_density", "mse_y": "mse_vap_density"}, axis="columns"
)
vle_mses = vle_mses.merge(vle_predicted_mses["sim_Pvap"], on=R32.param_names)
vle_mses = vle_mses.merge(vle_predicted_mses["sim_Hvap"], on=R32.param_names)
vle_mses = vle_mses.rename(
    {"mse_x": "mse_Pvap", "mse_y": "mse_Hvap"}, axis="columns"
)

# Find pareto efficient points
result, pareto_points, dominated_points = find_pareto_set(
    vle_mses.drop(columns=list(R32.param_names)).values, is_pareto_efficient
)
vle_mses = vle_mses.join(pd.DataFrame(result, columns=["is_pareto"]))


# Plot pareto points vs. MSEs
g = seaborn.pairplot(
    vle_mses,
    vars=["mse_liq_density", "mse_vap_density", "mse_Pvap", "mse_Hvap"],
    hue="is_pareto",
)
g.savefig("figs/R32-pareto-mses.pdf")

# Plot pareto points vs. params
g = seaborn.pairplot(vle_mses, vars=list(R32.param_names), hue="is_pareto")
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("figs/R32-pareto-params.pdf")

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

# Search to ID well spaced points
distance = 0.52
discarded_points = pd.DataFrame(columns=pareto_points.columns)

np.random.seed(distance_seed)

while len(pareto_points > 0):
    # Shuffle the pareto points
    pareto_points = pareto_points.sample(frac=1)
    new_points = new_points.append(pareto_points.iloc[[0]])
    # Remove anything within distance
    l1_norm = np.sum(
        np.abs(
            pareto_points[list(R32.param_names)].values
            - new_points[list(R32.param_names)].iloc[[-1]].values
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
g = seaborn.pairplot(new_points, vars=list(R32.param_names))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("figs/R32-new-points.pdf")

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
