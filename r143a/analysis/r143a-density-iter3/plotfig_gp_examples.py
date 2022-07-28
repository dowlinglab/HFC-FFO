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

from fffit.plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

from fffit.models import run_gpflow_scipy

sys.path.append("../")

from utils.r143a import R143aConstants
from utils.id_new_samples import prepare_df_density

R143a = R143aConstants()

pdf = PdfPages('fig_gp_examples.pdf')

#To put files in different PDFS change line 32 to 
# pdf_model = PdfPages('fig_gp_examples_model.pdf')
# pdf_params = PdfPages('fig_gp_examples_params.pdf')
# pdf_train_test = PdfPages('fig_gp_examples_train_test.pdf')

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 3
gp_shuffle_seed = 8278573

##############################################################################
##############################################################################

csv_path = "../csv/"
#in_csv_names = ["r143a-density-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
in_csv_names = "r143a-density-iter" + str(iternum) + "-results.csv" 
out_csv_name = "r143a-density-iter" + str(iternum + 1) + "-params.csv"

# Read files

#df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csvs = pd.read_csv(csv_path + in_csv_names, index_col=0) 
#df_csv = pd.concat(df_csvs)
df_csv = df_csvs
df_all, df_liq, df_vap = prepare_df_density(df_csv, R143a,liquid_density_threshold=500)
df_all = df_liq #Set df_all to df_liq to only plot liquid GP parity plots
#Print len df_all to troubleshoot # of plots

### Fit GP Model to liquid density
param_names = list(R143a.param_names) + ["temperature"]
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)
#Print len(x_train) to troubleshoot # of plots

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R143a.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R143a.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R143a.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R143a.liq_density_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R143a.liq_density_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R143a.n_params,
    R143a.temperature_bounds,
    R143a.liq_density_bounds,
    property_name="Liquid Density [kg/m^3]",
)

try:
    for fig in figs:
        #pdf_model.savefig(fig)
        pdf.savefig(fig)
except:
    #pdf_model.savefig(figs)
    pdf.savefig(figs)
    
del figs

# Plot parameter slices
for param_name in R143a.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R143a.param_names,
        300,
        R143a.temperature_bounds,
        R143a.liq_density_bounds,
        property_name="Liquid Density [kg/m^3]",
    )
    try:
        for fig in figs:
            #pdf_params.savefig(fig)
            pdf.savefig(fig)
    except:
        #pdf_params.savefig(figs)
        pdf.savefig(figs)
    del figs

# Loop over test params
for test_params in x_test[:,:R143a.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(np.where((df_all[list(R143a.param_names)] == test_params).all(axis=1))[0])
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if np.where((df_all.values[match,:R143a.n_params+1] == x_test[:,:R143a.n_params+1]).all(axis=1))[0].shape[0] == 1:
            test_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])
        # Else append to train points
        else:
            train_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R143a.temperature_bounds,
            R143a.liq_density_bounds,
            property_name="Liquid Density [kg/m^3]"
        )
    )
    
#     pdf_train_test.savefig(
#         plot_model_vs_test(
#             models,
#             test_params,
#             np.asarray(train_points),
#             np.asarray(test_points),
#             R143a.temperature_bounds,
#             R143a.liq_density_bounds,
#             property_name="Liquid Density [kg/m^3]"
#         )
#     )
'''
### Fit GP Model to vapor density
param_names = list(R143a.param_names) + ["temperature"]
property_name = "sim_vap_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R143a.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R143a.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R143a.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R143a.vap_density_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R143a.vap_density_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R143a.n_params,
    R143a.temperature_bounds,
    R143a.vap_density_bounds,
    property_name="Vapor Density [kg/m^3]",
)

for fig in figs:
    pdf.savefig(fig)
del figs

# Plot parameter slices
for param_name in R32.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R32.param_names,
        300,
        R32.temperature_bounds,
        R32.vap_density_bounds,
        property_name="Vapor Density [kg/m^3]",
    )
    for fig in figs:
        pdf.savefig(fig)
    del figs


# Loop over test params
for test_params in x_test[:,:R32.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(np.where((df_all[list(R32.param_names)] == test_params).all(axis=1))[0])
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if np.where((df_all.values[match,:R32.n_params+1] == x_test[:,:R32.n_params+1]).all(axis=1))[0].shape[0] == 1:
            test_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])
        # Else append to train points
        else:
            train_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R32.temperature_bounds,
            R32.vap_density_bounds,
            property_name="Vapor Density [kg/m^3]"
        )
    )

### Fit GP Model to Pvap
param_names = list(R32.param_names) + ["temperature"]
property_name = "sim_Pvap"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R32.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R32.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R32.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R32.Pvap_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R32.Pvap_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R32.n_params,
    R32.temperature_bounds,
    R32.Pvap_bounds,
    property_name="Vapor Pressure [bar]",
)

for fig in figs:
    pdf.savefig(fig)
del figs

# Plot parameter slices
for param_name in R32.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R32.param_names,
        300,
        R32.temperature_bounds,
        R32.Pvap_bounds,
        property_name="Vapor Pressure [bar]",
    )
    for fig in figs:
        pdf.savefig(fig)
    del figs


# Loop over test params
for test_params in x_test[:,:R32.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(np.where((df_all[list(R32.param_names)] == test_params).all(axis=1))[0])
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if np.where((df_all.values[match,:R32.n_params+1] == x_test[:,:R32.n_params+1]).all(axis=1))[0].shape[0] == 1:
            test_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])
        # Else append to train points
        else:
            train_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R32.temperature_bounds,
            R32.Pvap_bounds,
            property_name="Vapor pressure [bar]"
        )
    )


### Fit GP Model to Hvap
param_names = list(R32.param_names) + ["temperature"]
property_name = "sim_Hvap"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_all, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
)

# Fit model
models = {}
models["RBF"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R32.n_params + 1)),
)
models["Matern32"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern32(lengthscales=np.ones(R32.n_params + 1)),
)

models["Matern52"] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R32.n_params + 1)),
)

# Plot model performance on train and test points
pdf.savefig(plot_model_performance(models, x_train, y_train, R32.Hvap_bounds))
pdf.savefig(plot_model_performance(models, x_test, y_test, R32.Hvap_bounds))

# Plot temperature slices
figs = plot_slices_temperature(
    models,
    R32.n_params,
    R32.temperature_bounds,
    R32.Hvap_bounds,
    property_name="Enthalpy of Vaporization [kJ/kg]",
)

for fig in figs:
    pdf.savefig(fig)
del figs

# Plot parameter slices
for param_name in R32.param_names:
    figs = plot_slices_params(
        models,
        param_name,
        R32.param_names,
        300,
        R32.temperature_bounds,
        R32.Hvap_bounds,
        property_name="Enthalpy of Vaporization [kJ/kg]",
    )
    for fig in figs:
        pdf.savefig(fig)
    del figs

# Loop over test params
for test_params in x_test[:,:R32.n_params]:
    train_points = []
    test_points = []
    # Locate rows where parameter set == test parameter set
    matches = np.unique(np.where((df_all[list(R32.param_names)] == test_params).all(axis=1))[0])
    # Loop over all matches -- these will be different temperatures
    for match in matches:
        # If the match (including T) is in the test set, then append to test points
        if np.where((df_all.values[match,:R32.n_params+1] == x_test[:,:R32.n_params+1]).all(axis=1))[0].shape[0] == 1:
            test_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])
        # Else append to train points
        else:
            train_points.append([df_all["temperature"].iloc[match],df_all[property_name].iloc[match]])

    pdf.savefig(
        plot_model_vs_test(
            models,
            test_params,
            np.asarray(train_points),
            np.asarray(test_points),
            R32.temperature_bounds,
            R32.Hvap_bounds,
            property_name="Enthalpy of vaporization [kJ/kg]"
        )
    )
'''
pdf.close()
