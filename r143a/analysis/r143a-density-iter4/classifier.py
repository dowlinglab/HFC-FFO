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

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 4
cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

##############################################################################
##############################################################################

liquid_density_threshold = 500  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r143a/analysis/csv/"
in_csv_name = "r143a-density-iter" + str(iternum) + "-results.csv"

# Read file
in_csv_name_1and2iter = ["r143a-density-iter" + str(i) + "-results.csv" for i in range (1,5)]
df_1and2iters = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_name_1and2iter]
df_1and2iter = pd.concat(df_1and2iters)
df_all_1and2iter, df_liquid_only1iter, df_vapor_only1iter = prepare_df_density(
    df_1and2iter, R143a, liquid_density_threshold
)

df_csv = pd.read_csv(csv_path + in_csv_name, index_col=0)
#df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R143a, liquid_density_threshold
)

### Step 2: Fit classifier and GP models

# Create training/test set
param_names = list(R143a.param_names) + ["temperature"]
property_name = "is_liquid"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all_1and2iter, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifier is {test_score*100.0}% accurate on the original test set.")

x_train_new, y_train_new, x_test_new, y_test_new, = shuffle_and_split(
    df_all, param_names, property_name, fraction_train=1,shuffle_seed=cl_shuffle_seed
)
test_score_new = classifier.score(x_train_new, y_train_new)
print(f"Classifier is {test_score_new*100.0}% accurate on the 1000 data points of iter 4.")
plot_confusion_matrix(classifier, x_train_new, y_train_new)  
plt.savefig("classifier_predict.pdf")


