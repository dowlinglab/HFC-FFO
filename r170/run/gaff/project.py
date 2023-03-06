from flow import FlowProject, directives
from pathlib import Path
from parameters import mol_weight, smiles, custom_args, custom_args_gemc, n_vap, n_liq, reference_data, sim_length, n_atoms
import sys
from templates import ndcrc

vle = FlowProject.make_group(name="vle")

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

#class Project(FlowProject):
#    """Subclass of FlowProject to provide custom methods and attributes."""
#    pass
    
@FlowProject.label
def nvt_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os 

    with job:
        try:
            thermo_data = np.genfromtxt(
                "nvt.eq.out.prp", skip_header=3
            )
            completed = int(thermo_data[-1][0]) == sim_length["liq"]["nvt"] #job.sp.nsteps_liqeq
        except:
            completed = False
            pass

    return completed

@FlowProject.label
def npt_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os

    with job:
        try:
            thermo_data = np.genfromtxt(
                "npt.eq.out.prp", skip_header=3
            )
            completed = int(thermo_data[-1][0]) == sim_length["liq"]["npt"] #job.sp.nsteps_liqeq
        except:
            completed = False
            pass

    return completed

@FlowProject.label
def gemc_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os
    
    with job:
        try:
            thermo_data = np.genfromtxt(
                "gemc.eq.rst.001.out.box1.prp", skip_header=3
            )
            completed = int(thermo_data[-1][0]) == sim_length["gemc"]["prod"] #job.sp.nsteps_liqeq
        except:
            completed = False
            pass

    return completed

@vle
@FlowProject.operation
@FlowProject.post(lambda job: "vapboxl" in job.doc)
@FlowProject.post(lambda job: "liqboxl" in job.doc)
def calc_boxl(job):
    "Calculate the initial box length of the boxes"

    import unyt as u

    ref = {t.in_units(u.K).to_value(): (rho_liq, rho_vap, p_vap) for t, rho_liq, rho_vap, p_vap in reference_data}

    vap_density = ref[job.sp.T][1]
    mol_density = vap_density / mol_weight
    vol_vap = n_vap / mol_density
    vapboxl = vol_vap ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    vapboxl = round(float(vapboxl.in_units(u.nm).to_value()), 2)

    # Save to job document file
    job.doc.vapboxl = vapboxl  # nm, compatible with mbuild



    liq_density = ref[job.sp.T][0]
    mol_density = liq_density / mol_weight
    vol_liq = n_liq / mol_density
    liqboxl = vol_liq ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    liqboxl = round(float(liqboxl.in_units(u.nm).to_value()), 2)

    # Save to job document file
    job.doc.liqboxl = liqboxl  # nm, compatible with mbuild

@vle
@FlowProject.operation
@FlowProject.pre.after(calc_boxl)
@FlowProject.post(nvt_finished)
@directives(omp_num_threads=12)
def NVT_liqbox(job):
    "Equilibrate the liquid box using NVT simulation"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield("r170_gaff.xml")

    # Load the compound and apply the ff
    compound = mbuild.load(smiles, smiles=True)
    compound_ff = ff.apply(compound)

    # Create box list and species list
    boxl = job.doc.liqboxl
    box = mbuild.Box(lengths=[boxl, boxl, boxl])

    box_list = [box]
    species_list = [compound_ff]

    mols_to_add = [[n_liq]]

    system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

    # Create a new moves object
    moves = mc.MoveSet("nvt", species_list)

    # Property outputs relevant for NPT simulations
    thermo_props = [
            "energy_total", 
            "pressure"
            ]

    custom_args["run_name"] = "nvt.eq"
    custom_args["properties"] = thermo_props

    # Move into the job dir and start doing things
    with job:

        # Run equilibration
        mc.run(
            system=system,
            moveset=moves,
            run_type="equilibration",
            run_length=sim_length["liq"]["nvt"],
            temperature=job.sp.T * u.K,
            **custom_args
        )


@vle
@FlowProject.operation
@FlowProject.pre.after(NVT_liqbox)
@FlowProject.post.isfile("nvt.final.xyz")
@FlowProject.post(lambda job: "nvt_liqbox_final_dim" in job.doc)
def extract_final_NVT_config(job):
    "Extract final coords and box dims from the liquid box simulation"

    import subprocess

    lines = n_liq * n_atoms
    cmd = [
        "tail",
        "-n",
        str(lines + 2),
        job.fn("nvt.eq.out.xyz"),
    ]

    # Save final liuqid box xyz file
    xyz = subprocess.check_output(cmd).decode("utf-8")
    with open(job.fn("nvt.final.xyz"), "w") as xyzfile:
        xyzfile.write(xyz)

    # Save final box dims to job.doc
    box_data = []
    with open(job.fn("nvt.eq.out.H")) as f:
        for line in f:
            box_data.append(line.strip().split())
    job.doc.nvt_liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm


@vle
@FlowProject.operation
@FlowProject.pre.after(extract_final_NVT_config)
@FlowProject.post(npt_finished)
@directives(omp_num_threads=12)
def NPT_liqbox(job):
    "Equilibrate the liquid box"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield("r170_gaff.xml")

    # Load the compound and apply the ff
    compound = mbuild.load(smiles, smiles=True)
    compound_ff = ff.apply(compound)

    with job:
        liq_box = mbuild.formats.xyz.read_xyz(job.fn("nvt.final.xyz"))

    boxl = job.doc.nvt_liqbox_final_dim

    liq_box.box = mbuild.Box(lengths=[boxl, boxl, boxl], angles=[90., 90., 90.])

    liq_box.periodicity = [True, True, True]

    box_list = [liq_box]

    species_list = [compound_ff]

    mols_in_boxes = [[n_liq]]

    system = mc.System(box_list, species_list, mols_in_boxes=mols_in_boxes)

    # Create a new moves object
    moves = mc.MoveSet("npt", species_list)

    # Edit the volume move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    new_prob_volume = 1.0 / n_liq
    moves.prob_volume = new_prob_volume

    moves.prob_translate = (
        moves.prob_translate + orig_prob_volume - new_prob_volume
    )

    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "mass_density",
    ]

    # Define custom args

    custom_args["run_name"] = "npt.eq"
    custom_args["properties"] = thermo_props

    # Move into the job dir and start doing things
    with job:
        # Run equilibration
        for item in reference_data:
            if item[0].value == job.sp.T:
                pressure = item[-1]

        mc.run(
            system=system,
            moveset=moves,
            run_type="equilibration",
            run_length=sim_length["liq"]["npt"],
            temperature=job.sp.T * u.K,
            pressure= pressure,
            **custom_args
        )

@vle
@FlowProject.operation
@FlowProject.pre.after(NPT_liqbox)
@FlowProject.post.isfile("npt.final.xyz")
@FlowProject.post(lambda job: "npt_liqbox_final_dim" in job.doc)
def extract_final_NPT_config(job):
    "Extract final coords and box dims from the liquid box simulation"

    import subprocess

    lines = n_liq * n_atoms
    cmd = [
        "tail",
        "-n",
        str(lines + 2),
        job.fn("npt.eq.out.xyz"),
    ]

    # Save final liuqid box xyz file
    xyz = subprocess.check_output(cmd).decode("utf-8")
    with open(job.fn("npt.final.xyz"), "w") as xyzfile:
        xyzfile.write(xyz)

    # Save final box dims to job.doc
    box_data = []
    with open(job.fn("npt.eq.out.H")) as f:
        for line in f:
            box_data.append(line.strip().split())
    job.doc.npt_liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm


@vle
@FlowProject.operation
@FlowProject.pre.after(extract_final_NPT_config)
@FlowProject.post(gemc_finished)
@directives(omp_num_threads=12)
def GEMC(job):
    "Equilibrate GEMC"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield("r170_gaff.xml")

    # Load the compound and apply the ff
    compound = mbuild.load(smiles, smiles=True)
    compound_ff = ff.apply(compound)

    with job:
        liq_box = mbuild.formats.xyz.read_xyz(job.fn("npt.final.xyz"))

    boxl_liq = job.doc.npt_liqbox_final_dim

    liq_box.box = mbuild.Box(lengths=[boxl_liq, boxl_liq, boxl_liq], angles=[90., 90., 90.])

    liq_box.periodicity = [True, True, True]

    boxl_vap = job.doc.vapboxl

    vap_box = mbuild.Box(lengths=[boxl_vap, boxl_vap, boxl_vap], angles=[90., 90., 90.])

    box_list = [liq_box, vap_box]

    species_list = [compound_ff]

    mols_in_boxes = [[n_liq], [0]]

    mols_to_add = [[0], [n_vap]]

    system = mc.System(box_list, species_list, mols_in_boxes=mols_in_boxes, mols_to_add=mols_to_add)

    # Create a new moves object
    moves = mc.MoveSet("gemc", species_list)


    # Edit the volume and swap move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    orig_prob_swap = moves.prob_swap
    new_prob_volume = 1.0 / (n_vap + n_liq)
    new_prob_swap = 4.0 / 0.05 / (n_vap + n_liq)
    moves.prob_volume = new_prob_volume
    moves.prob_swap = new_prob_swap

    moves.prob_translate = (
        moves.prob_translate + orig_prob_volume - new_prob_volume
    )
    moves.prob_translate = (
        moves.prob_translate + orig_prob_swap - new_prob_swap
    )

    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
        "enthalpy",
    ]

    # Define custom args

    custom_args_gemc["run_name"] = "gemc.eq"
    custom_args_gemc["properties"] = thermo_props

    custom_args_gemc["charge_cutoff_box2"] = 0.4 * (boxl_vap * u.nanometer).to("angstrom")
    custom_args_gemc["vdw_cutoff_box2"] = 0.4 * (boxl_vap * u.nanometer).to("angstrom")

    # Move into the job dir and start doing things
    with job:

        mc.run(
            system=system,
            moveset=moves,
            run_type="equilibration",
            run_length=sim_length["gemc"]["eq"],
            temperature=job.sp.T * u.K,
            **custom_args_gemc
        )

        custom_args_gemc["run_name"] = "gemc.prod"
        #custom_args_gemc["restart_name"] = "gemc.eq"

        mc.restart(
            restart_from="gemc.eq",
            run_type="production",
            total_run_length=sim_length["gemc"]["prod"],
        )



#@Project.post(lambda job: "liq_density_unc" in job.doc)
#@Project.post(lambda job: "vap_density_unc" in job.doc)
#@Project.post(lambda job: "Pvap_unc" in job.doc)
#@Project.post(lambda job: "Hvap_unc" in job.doc)
#@Project.post(lambda job: "liq_enthalpy_unc" in job.doc)
#@Project.post(lambda job: "vap_enthalpy_unc" in job.doc)

@vle
@FlowProject.operation
@FlowProject.pre.after(GEMC)
@FlowProject.post(lambda job: "liq_density" in job.doc)
@FlowProject.post(lambda job: "vap_density" in job.doc)
@FlowProject.post(lambda job: "Pvap" in job.doc)
@FlowProject.post(lambda job: "Hvap" in job.doc)
@FlowProject.post(lambda job: "liq_enthalpy" in job.doc)
@FlowProject.post(lambda job: "vap_enthalpy" in job.doc)
@FlowProject.post(lambda job: "nmols_liq" in job.doc)
@FlowProject.post(lambda job: "nmols_vap" in job.doc)
def calculate_props(job):
    """Calculate the density"""

    import numpy as np
    import pylab as plt
    from block_average import block_average
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
        "enthalpy",
    ]

    with job:
        df_box1 = np.genfromtxt("gemc.eq.rst.001.out.box1.prp")
        df_box2 = np.genfromtxt("gemc.eq.rst.001.out.box2.prp")

    energy_col = 1
    density_col = 5
    pressure_col = 2
    enth_col = 6
    n_mols_col = 4

    # pull steps
    steps = df_box1[:, 0]

    # pull energy
    liq_energy= df_box1[:, energy_col]
    vap_energy= df_box2[:, energy_col]

    # pull density and take average
    liq_density = df_box1[:, density_col]
    liq_density_ave = np.mean(liq_density)
    vap_density = df_box2[:, density_col]
    vap_density_ave = np.mean(vap_density)
    
    # pull vapor pressure and take average
    Pvap = df_box2[:, pressure_col]
    Pvap_ave = np.mean(Pvap)
    
    # pull enthalpy and take average
    liq_enthalpy = df_box1[:, enth_col]
    liq_enthalpy_ave = np.mean(liq_enthalpy)
    vap_enthalpy = df_box2[:, enth_col]
    vap_enthalpy_ave = np.mean(vap_enthalpy)
    
    # pull number of moles and take average
    nmols_liq = df_box1[:, n_mols_col]
    nmols_liq_ave = np.mean(nmols_liq)
    nmols_vap = df_box2[:, n_mols_col]
    nmols_vap_ave = np.mean(nmols_vap)
    
    # calculate enthalpy of vaporization
    Hvap = (vap_enthalpy/nmols_vap) - (liq_enthalpy/nmols_liq)
    Hvap_ave = np.mean(Hvap)
    
    # save average density
    job.doc.liq_density = liq_density_ave
    job.doc.vap_density = vap_density_ave
    job.doc.Pvap = Pvap_ave
    job.doc.Hvap = Hvap_ave
    job.doc.liq_enthalpy = liq_enthalpy_ave
    job.doc.vap_enthalpy = vap_enthalpy_ave
    job.doc.nmols_liq = nmols_liq_ave
    job.doc.nmols_vap = nmols_vap_ave

    plt.rcParams['font.family'] = "DIN Alternate"
    font = {'family' : 'DIN Alternate',
            'weight' : 'normal',
                    'size'   : 12}
    
    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Energy')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    
    ax.title.set_text(f"Energy vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(steps, liq_energy, label='Liquid Energy')
    ax.plot(steps, vap_energy, label='Vapor Energy')
    ax.legend(loc="best")

    plt.savefig("energy.png")

    Props = {
        "liq_density": liq_density,
        "vap_density": vap_density,
        "Pvap": Pvap,
        "Hvap" : Hvap,
        "liq_enthalpy": liq_enthalpy,
        "vap_enthalpy": vap_enthalpy,
        "nmols_liq": nmols_liq,
        "nmols_vap": nmols_vap,
    }

    for name, prop in Props.items():
        (means_est, vars_est, vars_err) = block_average(prop)

        with open(job.fn(name + "_blk_avg.txt"), "w") as ferr:
            ferr.write("# nblk_ops, mean, vars, vars_err\n")
            for nblk_ops, (mean_est, var_est, var_err) in enumerate(
                zip(means_est, vars_est, vars_err)
            ):
                ferr.write(
                    "{}\t{}\t{}\t{}\n".format(
                        nblk_ops, mean_est, var_est, var_err
                    )
                )

        job.doc[name + "_unc"] = np.max(np.sqrt(vars_est))

@vle
@FlowProject.operation
@FlowProject.pre.after(GEMC)
def plot(job):

    import pandas as pd
    import pylab as plt

    with job:

        nvt_box1 = pd.read_table("nvt.eq.out.prp", sep="\s+", names=["step", "energy", "pressure"], skiprows=3)
        npt_box1 = pd.read_table("npt.eq.out.prp", sep="\s+", names=["step", "energy", "pressure", "density"], skiprows=3)
        gemc_eq_box1 = pd.read_table("gemc.eq.out.box1.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
        gemc_eq_box2 = pd.read_table("gemc.eq.out.box2.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
        gemc_prod_box1 = pd.read_table("gemc.eq.rst.001.out.box1.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
        gemc_prod_box2 = pd.read_table("gemc.eq.rst.001.out.box2.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)

    plt.rcParams['font.family'] = "DIN Alternate"
    
    font = {'family' : 'DIN Alternate',
            'weight' : 'normal',
                    'size'   : 12}
  

    #####################
    # GEMC Vapor Pressure
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Pressure (bar)')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Vapor pressure vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box2["step"][20:], gemc_eq_box2["pressure"][20:], label='GEMC-eq', color='red')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["pressure"], label='GEMC-prod', color='indianred')

    ax.legend(loc="best")
    plt.savefig(f"gemc-pvap-{job.sp.T}.png")

    #####################
    # GEMC nmols
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Number of molecules')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Number of molecules vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["nmols"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["nmols"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["nmols"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["nmols"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    plt.savefig(f"gemc-nmols-{job.sp.T}.png")

    #####################
    # GEMC volume
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Volume $\AA^3$')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Volume vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["volume"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["volume"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["volume"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["volume"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    plt.savefig(f"gemc-volume-{job.sp.T}.png")

    #####################
    # GEMC density
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Density $(kg / m^3)$')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Density vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["density"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["density"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["density"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["density"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    plt.savefig(f"gemc-density-{job.sp.T}.png")

    #####################
    # GEMC enthalpy 
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Enthalpy (kJ/mol-ext)')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Enthalpy vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["enthalpy"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["enthalpy"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["enthalpy"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["enthalpy"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    plt.savefig(f"gemc-enthalpy-{job.sp.T}.png")


    #############
    # NPT-Density
    #############

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Density $(kg / m^3)$')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    
    ax.title.set_text(f"NPT Density vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(npt_box1["step"], npt_box1["density"], label='NpT')

    ax.legend(loc="best")
    plt.savefig(f"npt-density-{job.sp.T}.png")


    # Shift steps so that we get an overall plot of energy across 
    # different workflow steps

    npt_box1["step"] += nvt_box1["step"].iloc[-1]
    gemc_eq_box1["step"] += npt_box1["step"].iloc[-1]
    gemc_eq_box2["step"] += npt_box1["step"].iloc[-1]
    gemc_prod_box1["step"] += npt_box1["step"].iloc[-1]
    gemc_prod_box2["step"] += npt_box1["step"].iloc[-1]

    #############
    # Energy
    #############

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Energy (kJ/mol-ext)')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    
    ax.title.set_text(f"Liquid Energy vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(nvt_box1["step"][20:], nvt_box1["energy"][20:], label='NVT', color="black")

    ax.plot(npt_box1["step"], npt_box1["energy"], label='NpT', color="gray")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["energy"], label='GEMC-eq-box1', color="blue")
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["energy"], label='GEMC-prod-box1', color="royalblue")
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["energy"], label='GEMC-eq-box2', color="red")
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["energy"], label='GEMC-prod-box2', color="indianred")

    ax.legend(loc="best")
    plt.savefig(f"all-energy-{job.sp.T}.png")


if __name__ == "__main__":
    FlowProject().main()
