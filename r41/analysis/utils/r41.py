import numpy as np
import unyt as u

class R41Constants:
    """Experimental data and other constants for R41"""
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
        return 34.033

    @property
    def expt_Tc(self):
        """Critical temperature in K"""
        return 317.28

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 316.51

    @property
    def n_atoms(self):
        """Number of adjustable parameters"""
        return 5

    @property
    def smiles_str(self):
        """Smiles string representation"""
        return "CF"

    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    @property
    def param_names(self):
        """Adjustable parameter names"""

        param_names = (
            "sigma_C1",
            "sigma_F1",
            "sigma_H1",
            "epsilon_C1",
            "epsilon_F1",
            "epsilon_H1",
        )

        return param_names
    
    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray(
                    [
                        [2.0, 4.0], #[3.0, 4.0],  # C
                        [2.0, 4.0],  # F
                        [1.5, 3.0],  # H
                    ]
                )
                * u.Angstrom
            )
            .in_units(u.nm)
            .value
        )

        bounds_epsilon = (
            (
                np.asarray(
                    [
                        [10.0,75.0], #[20.0, 75.0],  # C
                        [15.0, 50.0],  # F
                        [2.0, 10.0],  # H
                    ]
                )
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
            210: 847.46,
            230: 800.92,
            250: 749.06,
            270: 688.78,
            290: 613.19,
        }

        return expt_liq_density
    
    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        """

        expt_vap_density = {
            210: 4.5753,
            230: 10.278,
            250: 20.671,
            270: 38.976,
            290: 72.468,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        """

        expt_Pvap = {
            210: (218.52 * u.kPa).to_value(u.bar),
            230: (509.28 * u.kPa).to_value(u.bar),
            250: (1029.6 * u.kPa).to_value(u.bar),
            270: (1874.0 * u.kPa).to_value(u.bar),
            290: (3154.8 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            210: 465.553,
            230: 431.84,
            250: 391.28,
            270: 340.48,
            290: 272.32,
        }

        return expt_Hvap

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: https://doi.org/10.1021/je050186n
        Hvap not mentioned, put 2%
        """
        uncertainty = {
            "expt_liq_density": 0.002,
            "expt_vap_density": 0.002,
            "expt_Pvap": 0.002,
            "expt_Hvap": 0.02
        }
        return uncertainty
    
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
