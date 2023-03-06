import unyt as u
import copy

# Project name
project_name = "r170"

# Molecule definition

smiles = "CC"
mol_weight = 30.069 * u.amu
n_atoms = 8

# Thermodynamic conditions for simulation

simulation_temperatures = [
290.0 * u.K,
270.0 * u.K,
250.0 * u.K,
230.0 * u.K,
210.0 * u.K,
]


# Reference data to compare to (i.e. experiments or other simulation studies)
reference_data = [
(290.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 35.159 * u.bar),
(270.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 22.10 * u.bar),
(250.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 13.008 * u.bar),
(230.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 7.0018 * u.bar),
(210.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 3.338 * u.bar),
]

#reference_data = [
#(290.0 * u.K, 351.31 * u.kilogram/(u.meter)**3, 77.214 * u.kilogram/(u.meter)**3, 35.159 * u.bar),
#(270.0 * u.K, 407.72 * u.kilogram/(u.meter)**3, 42.089 * u.kilogram/(u.meter)**3, 22.10 * u.bar),
#(250.0 * u.K, 448.05 * u.kilogram/(u.meter)**3, 23.591 * u.kilogram/(u.meter)**3, 13.008 * u.bar),
#(230.0 * u.K, 481.29 * u.kilogram/(u.meter)**3, 12.676 * u.kilogram/(u.meter)**3, 7.0018 * u.bar),
#(210.0 * u.K, 510.45 * u.kilogram/(u.meter)**3, 6.2390 * u.kilogram/(u.meter)**3, 3.338 * u.bar),
#]

# Simulation related settings

n_vap = 160
n_liq = 640

# simulation_length must be consistent with the "units" field in custom args below
# For instance, if the "units" field is "sweeps" and simulation_length = 1000, 
# This will run a total of 1000 sweeps 
# (1 sweep = N steps, where N is the total number of molecules (n_vap + n_liq) 

sim_length = {}
sim_length["liq"] = {}
sim_length["liq"]["nvt"] = 2500000
sim_length["liq"]["npt"] = 2000
sim_length["gemc"] = {}
sim_length["gemc"]["eq"] = 10000000
sim_length["gemc"]["prod"] = 25000000

# Define custom args
# See page below for all options 
# https://mosdef-cassandra.readthedocs.io/en/latest/guides/kwargs.html
custom_args = {
    "vdw_style": "lj",
    "cutoff_style": "cut_tail",
    "vdw_cutoff": 12.0 * u.angstrom,
    "charge_style": "ewald",
    "charge_cutoff": 12.0 * u.angstrom, 
    "ewald_accuracy": 1.0e-5, 
    "mixing_rule": "lb",
    #"units": "sweeps",
    #"steps_per_sweep": n_liq,
    "units": "steps",
    "coord_freq": 1000,
    "prop_freq": 1000,
}

custom_args_gemc = copy.deepcopy(custom_args)

custom_args_gemc["units"] = "steps"
custom_args_gemc["coord_freq"] = 5000
custom_args_gemc["prop_freq"] = 1000
custom_args_gemc["vdw_cutoff_box1"] = custom_args["vdw_cutoff"] 
custom_args_gemc["charge_cutoff_box1"] = custom_args["charge_cutoff"]
'''custom_args_gemc["vdw_cutoff_box1"] = 12.0 * u.angstrom
custom_args_gemc["vdw_cutoff_box2"] = 12.0 * u.angstrom
custom_args_gemc["charge_cutoff_box1"] = 12.0 * u.angstrom
custom_args_gemc["charge_cutoff_box2"] = 12.0 * u.angstrom'''
