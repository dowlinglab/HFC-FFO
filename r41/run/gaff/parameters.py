import unyt as u
import copy

# Project name
project_name = "r14"

# Molecule definition

smiles = "C(F)(F)(F)F"
mol_weight = 88.0043 * u.amu
n_atoms = 5

# Thermodynamic conditions for simulation

simulation_temperatures = [
210.0 * u.K,
190.0 * u.K,
170.0 * u.K,
150.0 * u.K,
130.0 * u.K,
]


# Reference data to compare to (i.e. experiments or other simulation studies)
# starting point (experimental methane data leads to simulation issue possibly because experimental data lies outside GAFF VLE envelope, use experimental critical density as starting point for all T)
# Usually for an unknown system, start with vap and liq box both at critical density (inside the VLE envelope)
reference_data = [
(210.0 * u.K, 625.0 * u.kilogram/(u.meter)**3, 625.0 * u.kilogram/(u.meter)**3, 21.864 * u.bar),
(190.0 * u.K, 625.0 * u.kilogram/(u.meter)**3, 625.0 * u.kilogram/(u.meter)**3, 10.663 * u.bar),
(170.0 * u.K, 625.0 * u.kilogram/(u.meter)**3, 625.0 * u.kilogram/(u.meter)**3, 4.4054 * u.bar),
(150.0 * u.K, 625.0 * u.kilogram/(u.meter)**3, 625.0 * u.kilogram/(u.meter)**3, 1.412 * u.bar),
(130.0 * u.K, 625.0 * u.kilogram/(u.meter)**3, 625.0 * u.kilogram/(u.meter)**3, 0.30392 * u.bar),
]

#reference_data = [
#(170.0 * u.K, 310.50 * u.kilogram/(u.meter)**3, 38.974 * u.kilogram/(u.meter)**3, 23.283 * u.bar),
#(160.0 * u.K, 336.31 * u.kilogram/(u.meter)**3, 25.382 * u.kilogram/(u.meter)**3, 15.921 * u.bar),
#(150.0 * u.K, 357.90 * u.kilogram/(u.meter)**3, 16.328 * u.kilogram/(u.meter)**3, 10.4 * u.bar),
#(140.0 * u.K, 376.87 * u.kilogram/(u.meter)**3, 10.152 * u.kilogram/(u.meter)**3, 6.4118 * u.bar),
#(130.0 * u.K, 394.04 * u.kilogram/(u.meter)**3, 5.9804 * u.kilogram/(u.meter)**3, 3.6732 * u.bar),
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
sim_length["gemc"]["eq"] = 20000000
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
