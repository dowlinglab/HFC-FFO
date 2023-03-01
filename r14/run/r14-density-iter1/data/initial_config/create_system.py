import mbuild
import foyer
import unyt as u


def main():

    content = _generate_r14_xml()
    with open("ff.xml", "w") as inp:
        inp.write(content)

    r14 = mbuild.load("FC(F)(F)F", smiles=True)
    system = mbuild.fill_box(r14, n_compounds=300, density=1200,seed=12345,overlap=0.3)

    ff = foyer.Forcefield("ff.xml")

    system_ff = ff.apply(system)
    system_ff.combining_rule = "lorentz"

    system_ff.save("system.gro")
    system_ff.save("system.top")

    content = _generate_em_mdp()
    with open("em.mdp", "w") as inp:
        inp.write(content)

def _generate_r14_xml(): 

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(F)(F)(F)(F)" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="FC(F)(F)F" desc="F bonded to C"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029" k="596.30"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.781024"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.195256" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=(3.4 * u.Angstrom).in_units(u.nm).value,
        sigma_F1=(3.118 * u.Angstrom).in_units(u.nm).value,
        epsilon_C1=(55.052 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_F1=(30.696 * u.K * u.kb).in_units("kJ/mol").value,
    )

    return content


def _generate_em_mdp():

    contents = """
; MDP file for energy minimization

integrator	    = steep		    ; Algorithm (steep = steepest descent minimization)
emtol		    = 100.0  	    ; Stop minimization when the maximum force < 100.0 kJ/mol/nm
emstep          = 0.01          ; Energy step size
nsteps		    = 50000	  	    ; Maximum number of (minimization) steps to perform

nstlist		    = 1		    ; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet
ns-type		    = grid		; Method to determine neighbor list (simple, grid)
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps
coulombtype	    = PME		; Treatment of long range electrostatic interactions
rcoulomb	    = 1.0		; Short-range electrostatic cut-off
rvdw		    = 1.0		; Short-range Van der Waals cut-off
pbc		        = xyz 		; Periodic Boundary Conditions (yes/no)
constraints     = all-bonds
lincs-order     = 8
lincs-iter      = 4
"""

    return contents


if __name__ == "__main__":
    main()
