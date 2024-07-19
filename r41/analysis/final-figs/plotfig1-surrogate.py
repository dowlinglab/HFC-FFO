import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn


from fffit.models import run_gpflow_scipy
from fffit.utils import (
    shuffle_and_split,
    values_real_to_scaled,
    values_scaled_to_real,
    variances_scaled_to_real,
)


sys.path.append("../")

from utils.r32 import R32Constants
from utils.id_new_samples import prepare_df_vle

R32 = R32Constants()

matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")

iternum = 3
gp_shuffle_seed = 7579596

csv_path = "/scratch365/rdefever/hfcs-fffit/hfcs-fffit/analysis/csv/"
in_csv_names = ["r32-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum+1)]
out_csv_name = "r32-vle-iter" + str(iternum + 1) + "-params.csv"

# Read files
df_csvs = [pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names]
df_csv = pd.concat(df_csvs)
df_all = prepare_df_vle(df_csv, R32)


def main():

    seaborn.set_palette("Paired")

    # Liquid density first
    param_names = list(R32.param_names) + ["temperature"]
    property_name = "sim_liq_density"
    property_bounds = R32.liq_density_bounds

    # Extract train/test data
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_all,
        param_names,
        property_name,
        shuffle_seed=gp_shuffle_seed,
        fraction_train=0.8,
    )

    # Fit model
    model = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.RBF(lengthscales=np.ones(R32.n_params + 1)),
    )

    # Use model to predict results
    gp_mu_train, gp_var_train = model.predict_f(x_train)
    gp_mu_test, gp_var_test = model.predict_f(x_test)

    # Convert results to physical values
    y_train_physical = values_scaled_to_real(y_train, property_bounds)
    y_test_physical = values_scaled_to_real(y_test, property_bounds)
    gp_mu_train_physical = values_scaled_to_real(gp_mu_train, property_bounds)
    gp_mu_test_physical = values_scaled_to_real(gp_mu_test, property_bounds)

    # Plot
    fig, ax = plt.subplots()

    ax.scatter(
        y_train_physical,
        gp_mu_train_physical,
        label="Train",
        alpha=0.4,
        s=130,
        c="C1",
    )
    ax.scatter(
        y_test_physical,
        gp_mu_test_physical,
        marker="+",
        label="Test",
        alpha=0.7,
        s=170,
        c="C5",
    )

    xylim = [750, 1250]

    ax.plot(
        np.arange(xylim[0], xylim[1] + 100, 100),
        np.arange(xylim[0], xylim[1] + 100, 100),
        color="black",
        linewidth=3,
        alpha=0.6,
    )

    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[0], xylim[1])

    ax.set_xticks([800, 1000, 1200])
    ax.set_yticks([800, 1000, 1200])
    ax.set_xticks([850, 900, 950, 1050, 1100, 1150], minor=True)
    ax.set_yticks([850, 900, 950, 1050, 1100, 1150], minor=True)

    ax.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax.tick_params("both", which="major", length=8)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.set_xlabel(r"$\mathregular{\rho_{liq}\ sim. (kg/m^3)}$", fontsize=28, labelpad=20)
    ax.set_ylabel(r"$\mathregular{\rho_{liq}\ sur. (kg/m^3)}$", fontsize=28, labelpad=10)
    ax.legend(fontsize=24, handletextpad=0.00, loc="lower right", bbox_to_anchor=(1.01, -0.01))

    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig("pdfs/fig1-surrogate-liquiddensity.pdf")

    # Vapor density next
    param_names = list(R32.param_names) + ["temperature"]
    property_name = "sim_vap_density"
    property_bounds = R32.vap_density_bounds

    # Extract train/test data
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_all,
        param_names,
        property_name,
        shuffle_seed=gp_shuffle_seed,
        fraction_train=0.8,
    )

    # Fit model
    model = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.RBF(lengthscales=np.ones(R32.n_params + 1)),
    )

    # Use model to predict results
    gp_mu_train, gp_var_train = model.predict_f(x_train)
    gp_mu_test, gp_var_test = model.predict_f(x_test)

    # Convert results to physical values
    y_train_physical = values_scaled_to_real(y_train, property_bounds)
    y_test_physical = values_scaled_to_real(y_test, property_bounds)
    gp_mu_train_physical = values_scaled_to_real(gp_mu_train, property_bounds)
    gp_mu_test_physical = values_scaled_to_real(gp_mu_test, property_bounds)

    # Plot
    fig, ax = plt.subplots()

    ax.scatter(
        y_train_physical,
        gp_mu_train_physical,
        label="Train",
        alpha=0.6,
        s=130,
        c="C1",
    )
    ax.scatter(
        y_test_physical,
        gp_mu_test_physical,
        marker="+",
        label="Test",
        alpha=0.8,
        s=170,
        c="C5",
    )

    xylim = [0, 125]

    ax.plot(
        np.arange(xylim[0], xylim[1] + 100, 100),
        np.arange(xylim[0], xylim[1] + 100, 100),
        color="black",
        linewidth=3,
        alpha=0.6
    )

    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[0], xylim[1])

    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([25, 75, 125], minor=True)
    ax.set_yticks([25, 75, 125], minor=True)

    ax.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax.tick_params("both", which="major", length=8)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.set_xlabel(r"$\mathregular{\rho_{vap}\ sim. (kg/m^3)}$", fontsize=28, labelpad=20)
    ax.set_ylabel(r"$\mathregular{\rho_{vap}\ sur. (kg/m^3)}$", fontsize=28, labelpad=10)
    ax.legend(fontsize=24, handletextpad=0.00)

    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig("pdfs/fig1-surrogate-vapordensity.pdf")


if __name__ == "__main__":
    main()
