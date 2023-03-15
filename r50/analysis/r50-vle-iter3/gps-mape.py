import sys
import gpflow
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fffit.utils import (
    shuffle_and_split,
    values_real_to_scaled,
    values_scaled_to_real,
    variances_scaled_to_real,
)

from plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

from fffit.models import run_gpflow_scipy

sys.path.append("../")

from utils.r50 import R50Constants
from utils.id_new_samples import prepare_df_vle

R50 = R50Constants()

pdf = PdfPages('figs/gp_mape.pdf')

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 3
gp_shuffle_seed = 584745

##############################################################################
##############################################################################

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r50/analysis/csv/"
in_csv_names = ["r50-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
out_csv_name = "r50-vle-iter" + str(iternum + 1) + "-params.csv"

# Read files
df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all = prepare_df_vle(df_csv, R50)

### Fit GP Model to liquid density
param_names = list(R50.param_names) + ["temperature"]
property_name = "sim_liq_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R50.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_test, y_test, R50.liq_density_bounds))

### Fit GP Model to vapor density
param_names = list(R50.param_names) + ["temperature"]
property_name = "sim_vap_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R50.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_test, y_test, R50.vap_density_bounds))

### Fit GP Model to Pvap
param_names = list(R50.param_names) + ["temperature"]
property_name = "sim_Pvap"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R50.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_test, y_test, R50.Pvap_bounds))

### Fit GP Model to Hvap
param_names = list(R50.param_names) + ["temperature"]
property_name = "sim_Hvap"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)    

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R50.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_test, y_test, R50.Hvap_bounds))


pdf.close()

