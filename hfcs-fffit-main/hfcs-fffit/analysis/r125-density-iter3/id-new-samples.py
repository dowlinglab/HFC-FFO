import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn import svm

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
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R125 = R125Constants()

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 3
cl_shuffle_seed = 385569949
gp_shuffle_seed = 784759025

##############################################################################
##############################################################################

liquid_density_threshold = 500  # kg/m^3

csv_path = "/scratch365/rdefever/hfcs-fffit/hfcs-fffit/analysis/csv/"
in_csv_names = ["r125-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
out_csv_name = "r125-density-iter" + str(iternum + 1) + "-params.csv"

# Read files
df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R125, liquid_density_threshold
)

### Step 2: Fit classifier and GP models

# Create training/test set
param_names = list(R125.param_names) + ["temperature"]
property_name = "is_liquid"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifer is {test_score*100.0}% accurate on the test set.")


### Fit GP Model
# Create training/test set
param_names = list(R125.param_names) + ["temperature"]
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
)

# Fit model
model = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R125.n_params + 1)),
)


### Step 3: Find new parameters for MD simulations

# SVM to classify hypercube regions as liquid or vapor
latin_hypercube = np.loadtxt("LHS_5e5x10.csv", delimiter=",")
liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)
# Find the lowest MSE points from the GP in both sets
ranked_liquid_samples = rank_samples(liquid_samples, model, R125, "sim_liq_density")
ranked_vapor_samples = rank_samples(vapor_samples, model, R125, "sim_liq_density")

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
column_names = list(R125.param_names)
g = seaborn.pairplot(top_liquid_samples.drop(columns=["mse"]))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("liq_mse_below625.pdf")

# Create a pairplot of the top "vapor" parameter values
column_names = list(R125.param_names)
g = seaborn.pairplot(top_vapor_samples.drop(columns=["mse"]))
g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
g.savefig("vap_mse_below625.pdf")

#### Combine top 100 lowest MSE for parameter sets predicted as liquid and vapor

new_params = [
    ranked_liquid_samples.drop(columns=["mse"])[:100],
    ranked_vapor_samples.drop(columns=["mse"])[:100],
]

# Concatenate into a single dataframe and save to CSV
new_params = pd.concat(new_params)
new_params.to_csv(csv_path + out_csv_name)
