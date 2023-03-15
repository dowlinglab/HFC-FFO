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
from utils.id_new_samples import prepare_df_density

R50 = R50Constants()

pdf = PdfPages('gp.pdf')

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 4
gp_shuffle_seed = 8278573

##############################################################################
##############################################################################

csv_path = "../csv/"
in_csv_names = ["r50-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
out_csv_name = "r50-density-iter" + str(iternum + 1) + "-params.csv"

# Read files

df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all, df_liq, df_vap = prepare_df_density(df_csv, R50,liquid_density_threshold=200)
df_all = df_liq #Set df_all to df_liq to only plot liquid GP parity plots

### Fit GP Model to liquid density
param_names = list(R50.param_names) + ["temperature"]
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

#Find Temperature bounds for plots
plot_bounds = R50.temperature_bounds
plot_bounds[0] -= 10
plot_bounds[1] += 10

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R50.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_test, y_test, R50.liq_density_bounds))
pdf.close()

