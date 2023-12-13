from flow import FlowProject, directives
import templates.ndcrc
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Project(FlowProject):
    pass


@Project.post.isfile("ff.xml")
@Project.operation
def create_forcefield(job):
    """Create the forcefield .xml file for the job"""

    content = _generate_r50_xml(job)

    with open(job.fn("ff.xml"), "w") as ff:
        ff.write(content)


@Project.post(lambda job: "vapboxl" in job.doc)
@Project.operation
def calc_vapboxl(job):
    "Calculate the initial box length of the vapor box"

    import unyt as u

    pressure = job.sp.P * u.bar
    temperature = job.sp.T * u.K
    nmols_vap = job.sp.N_vap

    vol_vap = nmols_vap * u.kb * temperature / pressure
    boxl = vol_vap ** (1.0 / 3.0)
    # Strip unyts and round to 0.1 angstrom
    boxl = round(float(boxl.in_units(u.nm).value), 2)
    # Save to job document file
    job.doc.vapboxl = boxl  # nm, compatible with mbuild


@Project.post(lambda job: "liqboxl" in job.doc)
@Project.operation
def calc_liqboxl(job):
    "Calculate the initial box length of the liquid box"

    import unyt as u

    nmols_liq = job.sp.N_liq
    liq_density = job.sp.expt_liq_density * u.Unit("kg/m**3")
    molweight = 16.0425* u.amu
    mol_density = liq_density / molweight
    vol_liq = nmols_liq / mol_density
    boxl = vol_liq ** (1.0 / 3.0)
    # Strip unyts and round to 0.1 angstrom
    boxl = round(float(boxl.in_units(u.nm).value), 2)
    # Save to job document file
    job.doc.liqboxl = boxl  # nm, compatible with mbuild


@Project.label
def liqbox_equilibrated(job):
    "Confirm liquid box equilibration completed"

    import numpy as np
    
    try:
        thermo_data = np.genfromtxt(
            job.fn("liqbox-equil/equil.out.prp"), skip_header=3
        )
        completed = int(thermo_data[-1][0]) == job.sp.nsteps_liqeq
    except:
        completed = False
        pass

    return completed


@Project.pre.after(create_forcefield, calc_liqboxl, calc_vapboxl)
@Project.post(liqbox_equilibrated)
@Project.operation(directives={"omp_num_threads": 2})
def equilibrate_liqbox(job):
    "Equilibrate the liquid box"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u
    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load("C", smiles=True)
    compound_ff = ff.apply(compound)

    # Create box list and species list
    boxl = job.doc.liqboxl
    box = mbuild.Box(lengths=[boxl, boxl, boxl])

    box_list = [box]
    species_list = [compound_ff]

    mols_to_add = [[job.sp.N_liq]]

    system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

    # Create a new moves object
    moves = mc.MoveSet("npt", species_list)

    # Edit the volume move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    new_prob_volume = 1.0 / job.sp.N_liq
    moves.prob_volume = new_prob_volume

    moves.prob_translate = (
        moves.prob_translate + orig_prob_volume - new_prob_volume
    )

    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
    ]

    # Define custom args
    custom_args = {
        "run_name": "equil",
        "charge_style": "ewald",
        "rcut_min": 1.0 * u.angstrom,
        "vdw_cutoff": 12.0 * u.angstrom,
        "units": "sweeps",
        "steps_per_sweep": job.sp.N_liq,
        "coord_freq": 500,
        "prop_freq": 10,
        "properties": thermo_props,
    }

    # Move into the job dir and start doing things
    with job:
        liq_dir = "liqbox-equil"
        try:
            os.mkdir(liq_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        os.chdir(liq_dir)
        # Run equilibration
        mc.run(
            system=system,
            moveset=moves,
            run_type="equilibration",
            run_length=job.sp.nsteps_liqeq,
            temperature=job.sp.T * u.K,
            pressure=job.sp.P * u.bar,
            **custom_args
        )


@Project.pre.after(equilibrate_liqbox)
@Project.post.isfile("liqbox.xyz")
@Project.post(lambda job: "liqbox_final_dim" in job.doc)
@Project.operation
def extract_final_liqbox(job):
    "Extract final coords and box dims from the liquid box simulation"

    import subprocess

    n_atoms = job.sp.N_liq * 5 # number of atom per molecule; need to adjust!!!!!
    cmd = [
        "tail",
        "-n",
        str(n_atoms + 2),
        job.fn("liqbox-equil/equil.out.xyz"),
    ]

    # Save final liuqid box xyz file
    xyz = subprocess.check_output(cmd).decode("utf-8")
    with open(job.fn("liqbox.xyz"), "w") as xyzfile:
        xyzfile.write(xyz)

    # Save final box dims to job.doc
    box_data = []
    with open(job.fn("liqbox-equil/equil.out.H")) as f:
        for line in f:
            box_data.append(line.strip().split())
    job.doc.liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm


@Project.label
def gemc_equil_complete(job):
    "Confirm gemc equilibration has completed"
    import numpy as np

    try:
        thermo_data = np.genfromtxt(
            job.fn("equil.out.box1.prp"), skip_header=2
        )
        completed = int(thermo_data[-1][0]) == job.sp.nsteps_eq
    except:
        completed = False
        pass

    return completed


@Project.label
def gemc_prod_complete(job):
    "Confirm gemc production has completed"
    import numpy as np
    
    try:
        thermo_data = np.genfromtxt(job.fn("prod.out.box1.prp"), skip_header=3)
        completed = int(thermo_data[-1][0]) == job.sp.nsteps_prod
    except:
        completed = False
        pass

    return completed


@Project.pre.after(extract_final_liqbox)
@Project.post(gemc_prod_complete)
@Project.operation(directives={"omp_num_threads": 2})
def run_gemc(job):
    "Run gemc"
    
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u
    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load("C", smiles=True)
    compound_ff = ff.apply(compound)

    # Create box list and species list
    boxl = job.doc.liqbox_final_dim  # saved in nm
    liq_box = mbuild.load(job.fn("liqbox.xyz"))
    liq_box.periodicity = [boxl, boxl, boxl]
    boxl = job.doc.vapboxl  # nm
    vap_box = mbuild.Box(lengths=[boxl, boxl, boxl])

    box_list = [liq_box, vap_box]
    species_list = [compound_ff]

    mols_in_boxes = [[job.sp.N_liq], [0]]
    mols_to_add = [[0], [job.sp.N_vap]]

    system = mc.System(
        box_list,
        species_list,
        mols_in_boxes=mols_in_boxes,
        mols_to_add=mols_to_add,
    )

    # Create a new moves object
    moves = mc.MoveSet("gemc", species_list)

    # Edit the volume and swap move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    orig_prob_swap = moves.prob_swap
    new_prob_volume = 1.0 / (job.sp.N_liq + job.sp.N_vap)
    new_prob_swap = 4.0 / 0.05 / (job.sp.N_liq + job.sp.N_vap)
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
    custom_args = {
        "run_name": "equil",
        "charge_style": "ewald",
        "rcut_min": 1.0 * u.angstrom,
        "charge_cutoff_box2": 25.0 * u.angstrom,
        "vdw_cutoff_box1": 12.0 * u.angstrom,
        "vdw_cutoff_box2": 25.0 * u.angstrom,
        "units": "sweeps",
        "steps_per_sweep": job.sp.N_liq + job.sp.N_vap,
        "coord_freq": 500,
        "prop_freq": 10,
        "properties": thermo_props,
    }

    # Move into the job dir and start doing things
    with job:
        # Run equilibration
        mc.run(
            system=system,
            moveset=moves,
            run_type="equilibration",
            run_length=job.sp.nsteps_eq,
            temperature=job.sp.T * u.K,
            **custom_args
        )

        # Adjust custom args for production
        #custom_args["run_name"] = "prod"
        #custom_args["restart_name"] = "equil"
        
        # Run production
        mc.restart(
            #system=system,
            restart_from="equil",
            #moveset=moves,
            run_type="production",
            total_run_length=job.sp.nsteps_prod,
            run_name="prod",
            #temperature=job.sp.T * u.K,
            #**custom_args
        )


@Project.pre.after(run_gemc)
@Project.post(lambda job: "liq_density" in job.doc)
@Project.post(lambda job: "vap_density" in job.doc)
@Project.post(lambda job: "Pvap" in job.doc)
@Project.post(lambda job: "Hvap" in job.doc)
@Project.post(lambda job: "liq_enthalpy" in job.doc)
@Project.post(lambda job: "vap_enthalpy" in job.doc)
@Project.post(lambda job: "nmols_liq" in job.doc)
@Project.post(lambda job: "nmols_vap" in job.doc)
@Project.post(lambda job: "liq_density_unc" in job.doc)
@Project.post(lambda job: "vap_density_unc" in job.doc)
@Project.post(lambda job: "Pvap_unc" in job.doc)
@Project.post(lambda job: "Hvap_unc" in job.doc)
@Project.post(lambda job: "liq_enthalpy_unc" in job.doc)
@Project.post(lambda job: "vap_enthalpy_unc" in job.doc)
@Project.operation
def calculate_props(job):
    """Calculate the density"""

    import numpy as np
    from block_average import block_average

    # Load the thermo data
    df_box1 = np.genfromtxt(job.fn("prod.out.box1.prp"))
    df_box2 = np.genfromtxt(job.fn("prod.out.box2.prp"))

    density_col = 6
    pressure_col = 3
    enth_col = 7
    n_mols_col = 5
    # pull density and take average
    liq_density = df_box1[:, density_col - 1]
    liq_density_ave = np.mean(liq_density)
    vap_density = df_box2[:, density_col - 1]
    vap_density_ave = np.mean(vap_density)

    # pull vapor pressure and take average
    Pvap = df_box2[:, pressure_col - 1]
    Pvap_ave = np.mean(Pvap)

    # pull enthalpy and take average
    liq_enthalpy = df_box1[:, enth_col - 1]
    liq_enthalpy_ave = np.mean(liq_enthalpy)
    vap_enthalpy = df_box2[:, enth_col - 1]
    vap_enthalpy_ave = np.mean(vap_enthalpy)

    # pull number of moles and take average
    nmols_liq = df_box1[:, n_mols_col - 1]
    nmols_liq_ave = np.mean(nmols_liq)
    nmols_vap = df_box2[:, n_mols_col - 1]
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

#####################################################################
################# HELPER FUNCTIONS BEYOND THIS POINT ################
#####################################################################
def _generate_r50_xml(job): 

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(H)(H)(H)(H)" desc="carbon"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="HC(H)(H)H" desc="H bonded to C"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.68637877"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106424" k="329.95108893"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="-0.512608" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.128152" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content
if __name__ == "__main__":
    Project().main()
