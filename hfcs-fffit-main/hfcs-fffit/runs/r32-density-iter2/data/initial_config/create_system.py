import mbuild
import foyer
import unyt as u


def main():

    content = _generate_r32_xml()
    with open("ff.xml", "w") as inp:
        inp.write(content)

    r32 = mbuild.load("C(F)(F)", smiles=True)
    system = mbuild.fill_box(r32, n_compounds=150, density=1000)

    ff = foyer.Forcefield("ff.xml")

    system_ff = ff.apply(system)
    system_ff.combining_rule = "lorentz"

    system_ff.save("system.gro")
    system_ff.save("system.top")

    content = _generate_em_mdp()
    with open("em.mdp", "w") as inp:
        inp.write(content)


def _generate_r32_xml():

    content = """<ForceField>
 <AtomTypes>
  <Type name="C" class="C" element="C" mass="12.011" def="C(F)(F)" desc="central carbon"/>
  <Type name="H" class="H" element="H" mass="1.008" def="H(C)" desc="first H bonded to C1_s1"/>
  <Type name="F" class="F" element="F" mass="18.998" def="F(C)" desc="F bonded to C1_s1"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="C" class2="H" length="0.10961" k="277566.56"/>
  <Bond class1="C" class2="F" length="0.13497" k="298653.92"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="H" class2="C" class3="H" angle="1.9233528356977512" k="326.352"/>
  <Angle class1="F" class2="C" class3="H" angle="1.898743693244631" k="427.6048"/>
  <Angle class1="F" class2="C" class3="F" angle="1.8737854849411122" k="593.2912"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C" charge="0.405467" sigma="0.3400" epsilon="0.457730"/>
  <Atom type="H" charge="0.048049" sigma="0.229317" epsilon="0.065689"/>
  <Atom type="F" charge="-0.250783" sigma="0.311814" epsilon="0.255224"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C=(3.4 * u.Angstrom).in_units(u.nm).value,
        sigma_F=(2.8 * u.Angstrom).in_units(u.nm).value,
        sigma_H=(2.4 * u.Angstrom).in_units(u.nm).value,
        epsilon_C=(38.742 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_F=(28.383 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_H=(7.096 * u.K * u.kb).in_units("kJ/mol").value,
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
