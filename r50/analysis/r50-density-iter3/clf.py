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

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 3
cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

##############################################################################
##############################################################################

liquid_density_threshold = 200  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r50/analysis/csv/"
in_csv_names = ["r50-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]

'''# Read file
in_csv_name_1 = "r50-density-iter" + str(1) + "-results.csv"
df_1 = pd.read_csv(csv_path + in_csv_name_1, index_col=0) #for in_csv_name in in_csv_name_1and2iter]
df_all_1, df_liquid_1, df_vapor_1 = prepare_df_density(
    df_1, R50, liquid_density_threshold
)'''

df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R50, liquid_density_threshold
)
print("There are ",df_liquid.shape[0], " liquid simulations.")
print("There are ",df_vapor.shape[0]," vapor simulations.")
print("Total number of simulations: ",df_all.shape[0])

# Create training/test set
# Train only on iteration 1
param_names = list(R50.param_names) + ["temperature"]
property_name = "is_liquid"
'''x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all_1, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifier is {test_score*100.0}% accurate on the test set.")
plot_confusion_matrix(classifier, x_test, y_test)  
plt.savefig("classifier_1.pdf")'''

#############################
# Train on all available data
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifier is {test_score*100.0}% accurate on the test set.")
plot_confusion_matrix(classifier, x_test, y_test)  
plt.savefig("classifier.pdf")

