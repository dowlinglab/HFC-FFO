import sys
import os
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

from fffit.plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

from fffit.models import run_gpflow_scipy

sys.path.append("../")

from utils.r41 import R41Constants
from utils.id_new_samples import prepare_df_vle

R41 = R41Constants()

os.makedirs("figs", exist_ok=True)
pdf = PdfPages("figs/gp_models_eval.pdf")

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1
gp_shuffle_seed = 42

##############################################################################
##############################################################################

csv_path = "../csv/"
in_csv_names = ["r41-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)]

# Read files
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_all = prepare_df_vle(df_csv, R41)
# print(R41.temperature_bounds)
# print(df_all)
### Fit GP Model to liquid density
param_names = list(R41.param_names) + ["temperature"]
property_name = "sim_liq_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R41.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R41.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R41.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R41.liq_density_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R41.liq_density_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R41.n_params,
    R41.temperature_bounds,
    R41.liq_density_bounds,
    plot_bounds=np.asarray(
        [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
        dtype=np.float32,
    ),
    property_name="Liquid Density [kg/m^3]",
)

for fig in figs:
    pdf.savefig(fig)
del figs

# Plot parameter slices
for param_name in R41.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R41.param_names,
        list(R41.expt_liq_density.keys())[2],
        R41.temperature_bounds,
        R41.liq_density_bounds,
        property_name="Liquid Density [kg/m^3]",
    )
    for fig in figs:
        pdf.savefig(fig)
    del figs

# Loop over test params
for test_params in x_test[:, : R41.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(
        np.where((df_all[list(R41.param_names)] == test_params).all(axis=1))[0]
    )
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if (
            np.where(
                (
                    df_all.values[match, : R41.n_params + 1]
                    == x_test[:, : R41.n_params + 1]
                ).all(axis=1)
            )[0].shape[0]
            == 1
        ):
            test_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )
        # Else append to train points
        else:
            train_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R41.temperature_bounds,
            R41.liq_density_bounds,
            plot_bounds=np.asarray(
                [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
                dtype=np.float32,
            ),
            property_name="Liquid Density [kg/m^3]",
        )
    )

### Fit GP Model to vapor density
param_names = list(R41.param_names) + ["temperature"]
property_name = "sim_vap_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R41.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R41.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R41.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R41.vap_density_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R41.vap_density_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R41.n_params,
    R41.temperature_bounds,
    R41.vap_density_bounds,
    plot_bounds=np.asarray(
        [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
        dtype=np.float32,
    ),
    property_name="Vapor Density [kg/m^3]",
)

for fig in figs:
    pdf.savefig(fig)
del figs

# Plot parameter slices
for param_name in R41.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R41.param_names,
        list(R41.expt_liq_density.keys())[2],
        R41.temperature_bounds,
        R41.vap_density_bounds,
        property_name="Vapor Density [kg/m^3]",
    )
    for fig in figs:
        pdf.savefig(fig)
    del figs


# Loop over test params
for test_params in x_test[:, : R41.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(
        np.where((df_all[list(R41.param_names)] == test_params).all(axis=1))[0]
    )
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if (
            np.where(
                (
                    df_all.values[match, : R41.n_params + 1]
                    == x_test[:, : R41.n_params + 1]
                ).all(axis=1)
            )[0].shape[0]
            == 1
        ):
            test_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )
        # Else append to train points
        else:
            train_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R41.temperature_bounds,
            R41.vap_density_bounds,
            plot_bounds=np.asarray(
                [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
                dtype=np.float32,
            ),
            property_name="Vapor Density [kg/m^3]",
        )
    )

### Fit GP Model to Pvap
param_names = list(R41.param_names) + ["temperature"]
property_name = "sim_Pvap"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R41.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R41.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R41.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R41.Pvap_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R41.Pvap_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R41.n_params,
    R41.temperature_bounds,
    R41.Pvap_bounds,
    plot_bounds=np.asarray(
        [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
        dtype=np.float32,
    ),
    property_name="Vapor Pressure [bar]",
)

for fig in figs:
    pdf.savefig(fig)
del figs

# Plot parameter slices
for param_name in R41.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R41.param_names,
        list(R41.expt_liq_density.keys())[2],
        R41.temperature_bounds,
        R41.Pvap_bounds,
        property_name="Vapor Pressure [bar]",
    )
    for fig in figs:
        pdf.savefig(fig)
    del figs


# Loop over test params
for test_params in x_test[:, : R41.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(
        np.where((df_all[list(R41.param_names)] == test_params).all(axis=1))[0]
    )
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if (
            np.where(
                (
                    df_all.values[match, : R41.n_params + 1]
                    == x_test[:, : R41.n_params + 1]
                ).all(axis=1)
            )[0].shape[0]
            == 1
        ):
            test_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )
        # Else append to train points
        else:
            train_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R41.temperature_bounds,
            R41.Pvap_bounds,
            plot_bounds=np.asarray(
                [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
                dtype=np.float32,
            ),
            property_name="Vapor pressure [bar]",
        )
    )


### Fit GP Model to Hvap
param_names = list(R41.param_names) + ["temperature"]
property_name = "sim_Hvap"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R41.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R41.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R41.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R41.Hvap_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R41.Hvap_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R41.n_params,
    R41.temperature_bounds,
    R41.Hvap_bounds,
    plot_bounds=np.asarray(
        [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
        dtype=np.float32,
    ),
    property_name="Enthalpy of Vaporization [kJ/kg]",
)

for fig in figs:
    pdf.savefig(fig)
del figs

# Plot parameter slices
for param_name in R41.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R41.param_names,
        list(R41.expt_liq_density.keys())[2],
        R41.temperature_bounds,
        R41.Hvap_bounds,
        property_name="Enthalpy of Vaporization [kJ/kg]",
    )
    for fig in figs:
        pdf.savefig(fig)
    del figs

# Loop over test params
for test_params in x_test[:, : R41.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(
        np.where((df_all[list(R41.param_names)] == test_params).all(axis=1))[0]
    )
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if (
            np.where(
                (
                    df_all.values[match, : R41.n_params + 1]
                    == x_test[:, : R41.n_params + 1]
                ).all(axis=1)
            )[0].shape[0]
            == 1
        ):
            test_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )
        # Else append to train points
        else:
            train_points.append(
                [df_all["temperature"].iloc[match], df_all[property_name].iloc[match]]
            )

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R41.temperature_bounds,
            R41.Hvap_bounds,
            plot_bounds=np.asarray(
                [R41.temperature_bounds[0] - 10, R41.temperature_bounds[1] + 10],
                dtype=np.float32,
            ),
            property_name="Enthalpy of vaporization [kJ/kg]",
        )
    )

pdf.close()
