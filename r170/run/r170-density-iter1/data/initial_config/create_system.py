import mbuild
import foyer
import unyt as u


def main():

    content = _generate_r170_xml()
    with open("ff.xml", "w") as inp:
        inp.write(content)

    r170 = mbuild.load("CC", smiles=True)
    system = mbuild.fill_box(r170, n_compounds=150, density=500,seed=12345,overlap=0.3)

    ff = foyer.Forcefield("ff.xml")

    system_ff = ff.apply(system)
    system_ff.combining_rule = "lorentz"

    system_ff.save("system.gro")
    system_ff.save("system.top")

    content = _generate_em_mdp()
    with open("em.mdp", "w") as inp:
        inp.write(content)


def _generate_r170_xml(): 

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(C)(H)(H)(H)" desc="carbon"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="H" desc="H bonded to C"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31000265"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.68637877"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.92073484" k="388.01928783"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106424" k="329.95108893"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="hc" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.62757555" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="-0.006120"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.002040"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=(3.4 * u.Angstrom).in_units(u.nm).value,
        sigma_H1=(2.65 * u.Angstrom).in_units(u.nm).value,
        epsilon_C1=(55.052 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_H1=(7.091 * u.K * u.kb).in_units("kJ/mol").value,
    ) #these values doesn't matter, not updated, since they'll be replaced by LHS 

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
