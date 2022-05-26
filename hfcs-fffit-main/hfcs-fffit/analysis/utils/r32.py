import numpy as np
import unyt as u

class R32Constants:
    """Experimental data and other constants for R32"""
    def __init__(self):
        assert (
            self.expt_liq_density.keys()
            == self.expt_vap_density.keys()
            == self.expt_Pvap.keys()
            == self.expt_Hvap.keys()
        )

    @property
    def molecular_weight(self):
        """Molecular weight of the molecule in g/mol"""
        return 52.024

    @property
    def expt_Tc(self):
        """Critical temperature in K"""
        return 351.35

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 429.756

    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    @property
    def param_names(self):
        """Adjustable parameter names"""

        param_names = (
            "sigma_C",
            "sigma_F",
            "sigma_H",
            "epsilon_C",
            "epsilon_F",
            "epsilon_H",
        )

        return param_names

    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray([[3.0, 4.0], [2.5, 3.5], [1.7, 2.7],]) * u.Angstrom
            )  # C  # F  # H
            .in_units(u.nm)
            .value
        )

        bounds_epsilon = (
            (
                np.asarray(
                    [[20.0, 60.0], [15.0, 40.0], [2.0, 10.0],]
                )  # C  # F  # H
                * u.K
                * u.kb
            )
            .in_units("kJ/mol")
            .value
        )

        bounds = np.vstack((bounds_sigma, bounds_epsilon))

        return bounds

    @property
    def expt_liq_density(self):
        """Dictionary with experimental liquid density

        Temperature units K
        Density units kg/m**3
        """

        expt_liq_density = {
            241: 1156.9,
            261: 1095.2,
            281: 1027.0,
            301: 948.31,
            321: 850.77,
        }

        return expt_liq_density

    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        """

        expt_vap_density = {
            241: 7.0557,
            261: 14.818,
            281: 28.426,
            301: 51.676,
            321: 92.786,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        """

        expt_Pvap = {
            241: 2.5159,
            261: 5.4327,
            281: 10.426,
            301: 18.295,
            321: 29.989,
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            241: (505.47-146.18),
            261: (512.47-179.37),
            281: (516.47-214.15),
            301: (516.09-251.40),
            321: (508.48-292.95),
        }

        return expt_Hvap

    @property
    def temperature_bounds(self):
        """Bounds on temperature in units of K"""

        lower_bound = np.min(list(self.expt_Pvap.keys()))
        upper_bound = np.max(list(self.expt_Pvap.keys()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def liq_density_bounds(self):
        """Bounds on liquid density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_liq_density.values()))
        upper_bound = np.max(list(self.expt_liq_density.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def vap_density_bounds(self):
        """Bounds on vapor density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_vap_density.values()))
        upper_bound = np.max(list(self.expt_vap_density.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def Pvap_bounds(self):
        """Bounds on vapor pressure in units of bar"""

        lower_bound = np.min(list(self.expt_Pvap.values()))
        upper_bound = np.max(list(self.expt_Pvap.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def Hvap_bounds(self):
        """Bounds on enthaply of vaporization in units of kJ/kg"""

        lower_bound = np.min(list(self.expt_Hvap.values()))
        upper_bound = np.max(list(self.expt_Hvap.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds
