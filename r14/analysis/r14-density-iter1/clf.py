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

from utils.r14 import R14Constants
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)

R14 = R14Constants()


############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum =1
cl_shuffle_seed = 6928457 #classifier

##############################################################################
##############################################################################

liquid_density_threshold = 1100  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier

csv_path = "../csv/"
in_csv_name = "r14-density-iter" + str(iternum) + "-results.csv" 

# Read file
df_csv = pd.read_csv(csv_path + in_csv_name, index_col=0)
#df_csv = pd.concat(df_csvs)
df_all, df_liquid, df_vapor = prepare_df_density(
    df_csv, R14, liquid_density_threshold
)
print("There are ",df_liquid.shape[0], " liquid simulations.")
print("There are ",df_vapor.shape[0]," vapor simulations.")
print("Total number of simulations: ",df_all.shape[0])

### Step 2: Fit classifier and GP models

# Create training/test set
param_names = list(R14.param_names) + ["temperature"]
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

