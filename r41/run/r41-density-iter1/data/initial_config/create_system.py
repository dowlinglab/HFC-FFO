import mbuild
import foyer
import unyt as u


def main():

    content = _generate_r41_xml()
    with open("ff.xml", "w") as inp:
        inp.write(content)

    r41 = mbuild.load("CF", smiles=True)
    system = mbuild.fill_box(r41, n_compounds=300, density=600,seed=12345,overlap=0.3)

    ff = foyer.Forcefield("ff.xml")

    system_ff = ff.apply(system)
    system_ff.combining_rule = "lorentz"

    system_ff.save("system.gro")
    system_ff.save("system.top")

    content = _generate_em_mdp()
    with open("em.mdp", "w") as inp:
        inp.write(content)

def _generate_r41_xml():

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(F)" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C1"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H(C)" desc="H bonded to C1"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.35"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="f" class2="c3" class3="h1" angle="1.8823376" k="431.53717916"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.9120082" k="327.85584464"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.119281"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.274252" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.051657"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=(3.4 * u.Angstrom).in_units(u.nm).value,
        sigma_F1=(2.8 * u.Angstrom).in_units(u.nm).value,
        sigma_H1=(2.4 * u.Angstrom).in_units(u.nm).value,
        epsilon_C1=(38.742 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_F1=(28.383 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_H1=(7.096 * u.K * u.kb).in_units("kJ/mol").value,
    ) #these values don't matter since they'll be replaced by LHS. Just need to be valid XML 


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
