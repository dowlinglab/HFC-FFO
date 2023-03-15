import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from fffit.utils import values_scaled_to_real
from fffit.utils import values_real_to_scaled
from fffit.utils import variances_scaled_to_real

mpl_is_inline = 'inline' in matplotlib.get_backend()


def plot_model_performance(
    models, x_data, y_data, property_bounds, xylim=None
):
    """Plot the predictions vs. result for one or more GP models

    Parameters
    ----------
    models : dict { label : model }
        Each model to be plotted (value, GPFlow model) is provided
        with a label (key, string)
    x_data : np.array
        data to create model predictions for
    y_data : np.ndarray
        correct answer
    property_bounds : array-like
        bounds for scaling density between physical
        and dimensionless values
    xylim : array-like, shape=(2,), optional
        lower and upper x and y limits of the plot

    Returns
    -------
    matplotlib.Figure.figure
    """
    y_data_physical = values_scaled_to_real(y_data, property_bounds)
    min_xylim = np.min(y_data_physical)
    max_xylim = np.max(y_data_physical)

    fig, ax = plt.subplots()

    for (label, model) in models.items():
        gp_mu, gp_var = model.predict_f(x_data)
        gp_mu_physical = values_scaled_to_real(gp_mu, property_bounds)
        ax.scatter(y_data_physical, gp_mu_physical, label=label, zorder=2.5, alpha=0.4)
        meansqerr = np.mean(
            (gp_mu_physical - y_data_physical.reshape(-1, 1)) ** 2
        )
        print("Model: {}. Mean squared err: {:.2e}".format(label, meansqerr))
        mape = np.mean(
            np.abs((gp_mu_physical - y_data_physical.reshape(-1, 1))/y_data_physical.reshape(-1, 1) )
        )
        print("Model: {}. MAPE: {:.6e}".format(label, mape))
        if np.min(gp_mu_physical) < min_xylim:
            min_xylim = np.min(gp_mu_physical)
        if np.max(gp_mu_physical) > max_xylim:
            max_xylim = np.max(gp_mu_physical)

    if xylim is None:
        xylim = [min_xylim, max_xylim]

    ax.plot(
        np.arange(xylim[0], xylim[1] + 100, 100),
        np.arange(xylim[0], xylim[1] + 100, 100),
        color="xkcd:blue grey",
        label="y=x",
    )

    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[0], xylim[1])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Model Prediction")
    ax.legend()
    ax.set_aspect("equal", "box")

    if not mpl_is_inline:
        return fig


def plot_slices_temperature(
    models,
    n_params,
    temperature_bounds,
    property_bounds,
    plot_bounds,#=[220.0, 340.0],
    property_name="property",
):
    """Plot the model predictions as a function of temperature
    Slices are plotted where the values of the other parameters
    are all set to 0.0 --> 1.0 in increments of 0.1
    Parameters
    ----------
    models : dict
        models to plot, key=label, value=gpflow.model
    n_params : int
        number of non-temperature parameters in the model
    temperature_bounds: array-like
        bounds for scaling temperature between physical
        and dimensionless values
    property_bounds: array-like
        bounds for scaling the property between physical
        and dimensionless values
    plot_bounds : array-like, optional
        temperature bounds for the plot
    property_name : str, optional, default="property"
        property name with units for axis label

    Returns
    -------
    figs : list
        list of matplotlib.figure.Figure objects
    """

    n_samples = 100
    vals = np.linspace(plot_bounds[0], plot_bounds[1], n_samples).reshape(
        -1, 1
    )
    vals_scaled = values_real_to_scaled(vals, temperature_bounds)

    figs = []
    for other_vals in np.arange(0, 1.1, 0.1):
        other = np.tile(other_vals, (n_samples, n_params))
        xx = np.hstack((other, vals_scaled))

        fig, ax = plt.subplots()
        for (label, model) in models.items():
            mean_scaled, var_scaled = model.predict_f(xx)
            mean = values_scaled_to_real(mean_scaled, property_bounds)
            var = variances_scaled_to_real(var_scaled, property_bounds)

            ax.plot(vals, mean, lw=2, label=label)
            ax.fill_between(
                vals[:, 0],
                mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                alpha=0.3,
            )

        ax.set_title(f"Other vals = {other_vals:.2f}")
        ax.set_xlabel("Temperature")
        ax.set_ylabel(property_name)
        fig.legend()
        figs.append(fig)

    if not mpl_is_inline:
        return figs


def plot_slices_params(
    models,
    param_to_plot,
    param_names,
    temperature,
    temperature_bounds,
    property_bounds,
    property_name="property",
):
    """Plot the model predictions as a function of param_to_plot
    at the specified temperature

    Parameters
    ----------
    models : dict {"label" : gpflow.model }
        GPFlow models to plot
    param_to_plot : string
        Parameter to vary
    param_names : list, tuple
        list of parameter names
    temperature : float
        temperature at which to plot the surface
    temperature_bounds: array-like
        bounds for scaling temperature between physical
        and dimensionless values
    property_bounds: array-like
        bounds for scaling property between physical
        and dimensionless values
    property_name : string, optional, default="property"
        name of property to plot

    Returns
    -------
    figs : list
        list of matplotlib.figure.Figure objects
    """

    try:
        param_idx = param_names.index(param_to_plot)
    except ValueError:
        raise ValueError(
            f"parameter: {param_to_plot} not found in parameter_names: {param_names}"
        )

    n_params = len(param_names)

    n_samples = 100
    vals_scaled = np.linspace(-0.1, 1.1, n_samples).reshape(-1, 1)
    temp_vals = np.tile(temperature, (n_samples, 1))
    temp_vals_scaled = values_real_to_scaled(temp_vals, temperature_bounds)

    figs = []
    for other_vals in np.arange(0, 1.1, 0.1):
        other1 = np.tile(other_vals, (n_samples, param_idx))
        other2 = np.tile(other_vals, (n_samples, n_params - 1 - param_idx))
        xx = np.hstack((other1, vals_scaled, other2, temp_vals_scaled))

        fig, ax = plt.subplots()
        for (label, model) in models.items():
            mean_scaled, var_scaled = model.predict_f(xx)
            mean = values_scaled_to_real(mean_scaled, property_bounds)
            var = variances_scaled_to_real(var_scaled, property_bounds)

            ax.plot(vals_scaled, mean, lw=2, label=label)
            ax.fill_between(
                vals_scaled[:, 0],
                mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                alpha=0.3,
            )

        math_parameter = "$\\" + param_to_plot + "$"
        ax.set_title(
            f"{math_parameter} at T = {temperature:.0f} K. Other vals = {other_vals:.2f}."
        )
        ax.set_xlabel(math_parameter)
        ax.set_ylabel(property_name)
        fig.legend()
        figs.append(fig)

    if not mpl_is_inline:
        return figs


def plot_model_vs_test(
    models,
    param_values,
    train_points,
    test_points,
    temperature_bounds,
    property_bounds,
    plot_bounds=[220.0, 340.0],
    property_name="property",
):
    """Plots the GP model(s) as a function of temperature with all other parameters
    taken as param_values. Overlays training and testing points with the same
    param_values.

    Parameters
    ----------
    models : dict {"label" : gpflow.model }
        GPFlow models to plot
    param_values : np.ndarray, shape=(n_params)
        The parameters at which to evaluate the GP model
    train_points : np.ndarray, shape=(n_points, 2)
        The temperature (scaled) and property (scaled) of each training point
    test_points : np.ndarray, shape=(n_points, 2)
        The temperature (scaled) and property (scaled) of each test point
    temperature_bounds: array-like
        bounds for scaling temperature between physical
        and dimensionless values
    property_bounds: array-like
        bounds for scaling property between physical
        and dimensionless values
    plot_bounds : array-like, optional
        temperature bounds for the plot
    property_name : str, optional, default="property"
        property name with units for axis label

    Returns
    -------
    matplotlib.figure.Figure
    """

    n_samples = 100
    vals = np.linspace(plot_bounds[0], plot_bounds[1], n_samples).reshape(
        -1, 1
    )
    vals_scaled = values_real_to_scaled(vals, temperature_bounds)

    other = np.tile(param_values, (n_samples, 1))
    xx = np.hstack((other, vals_scaled))

    fig, ax = plt.subplots()
    for (label, model) in models.items():
        mean_scaled, var_scaled = model.predict_f(xx)

        mean = values_scaled_to_real(mean_scaled, property_bounds)
        var = variances_scaled_to_real(var_scaled, property_bounds)
        ax.plot(vals, mean, lw=2, label="GP model" + label)
        ax.fill_between(
            vals[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            alpha=0.25,
        )

    if train_points.shape[0] > 0:
        md_train_temp = values_scaled_to_real(
            train_points[:, 0], temperature_bounds
        )
        md_train_property = values_scaled_to_real(
            train_points[:, 1], property_bounds
        )
        ax.plot(
            md_train_temp, md_train_property, "s", color="black", label="Train"
        )
    if test_points.shape[0] > 0:
        md_test_temp = values_scaled_to_real(
            test_points[:, 0], temperature_bounds
        )
        md_test_property = values_scaled_to_real(
            test_points[:, 1], property_bounds
        )
        ax.plot(md_test_temp, md_test_property, "ro", label="Test")

    ax.set_xlabel("Temperature")
    ax.set_ylabel(property_name)
    fig.legend()

    if not mpl_is_inline:
        return fig
