import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn import svm
import scipy.optimize as optimize
import unyt as u

from sklearn.model_selection import train_test_split

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
from skmultilearn.model_selection import iterative_train_test_split

sys.path.append("../")

from utils.r41 import R41Constants
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R41 = R41Constants()

############################# QUANTITIES TO EDIT #############################
##############################################################################
iternum = 2
cl_shuffle_seed = 1 #classifier
gp_shuffle_seed = 42 #GP seed 
liquid_density_threshold = 400  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier
save_fig = False
##############################################################################
##############################################################################


csv_path = "../csv/"
in_csv_names = ["r41-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
out_csv_name = "r41-density-iter" + str(iternum + 1) + "-params.csv"
out_top_liquid_csv_name = "r41-density-iter" + str(iternum ) + "-liquid-params.csv"
out_top_vapor_csv_name = "r41-density-iter" + str(iternum ) + "-vapor-params.csv"

# Read files for next iter
df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R41, liquid_density_threshold
)
### Step 2: Fit classifier and GP models

# Create training/test set
param_names = list(R41.param_names) + ["temperature"]
property_name = "is_liquid"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
#class_weight "balanced" used because there are fewer liquid than vapor samples in the LHS sets
classifier = svm.SVC(kernel="rbf", class_weight="balanced") 
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifer is {test_score*100.0}% accurate on the test set.")
ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test, display_labels = ["Vapor", "Liquid"])  

### Fit GP Model
# Create training/test set
param_names = list(R41.param_names) + ["temperature"]
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
)

# Fit model
model = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R41.n_params + 1)),
)

### Step 3: Find new parameters for MD simulations
# SVM to classify hypercube regions as liquid or vapor
next_samples = np.genfromtxt(csv_path + out_csv_name, delimiter=",",skip_header=1,)[:, 1:]
liquid_samples, vapor_samples = classify_samples(next_samples, classifier)

# Find the lowest MSE points from the GP in both sets
ranked_liquid_samples = rank_samples(liquid_samples, model, R41, "sim_liq_density")
ranked_vapor_samples = rank_samples(vapor_samples, model, R41, "sim_liq_density")#both l and g compared to liquid density
top_liquid_samples2 = ranked_liquid_samples[
    ranked_liquid_samples["mse"] < 100.0
]
top_vapor_samples2 = ranked_vapor_samples[ranked_vapor_samples["mse"] < 100.0]

print(
    "There are:",
    top_liquid_samples2.shape[0],
    "liquid parameter sets which produce densities within 10 kg/m$^3$ of experimental densities",
)
print(
    "There are:",
    top_vapor_samples2.shape[0],
    " vapor parameter sets which produce densities within 10 kg/m$^3$ of experimental densities",
)