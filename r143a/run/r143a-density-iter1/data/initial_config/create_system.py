import mbuild
import foyer
import unyt as u


def main():

    content = _generate_r143a_xml()
    with open("ff.xml", "w") as inp:
        inp.write(content)

    r143a = mbuild.load("C(F)(F)(F)C", smiles=True)
    system = mbuild.fill_box(r143a, n_compounds=150, density=1000)

    ff = foyer.Forcefield("ff.xml")

    system_ff = ff.apply(system)
    system_ff.combining_rule = "lorentz"

    system_ff.save("system.gro")
    system_ff.save("system.top")

    content = _generate_em_mdp()
    with open("em.mdp", "w") as inp:
        inp.write(content)


def _generate_r143a_xml():

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(C)(F)(F)(F)" desc="carbon bonded to 3 Fs and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.010" def="C(C)(H)(H)(H)" desc="carbon bonded to 3 Hs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C1"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="H(C)" desc="H bonded to C2"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.35"/>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.40"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.73"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956" k="544.13"/>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.920735" k="388.02"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029" k="596.30"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106" k="329.95"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="0.794946" phase2="0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.78821"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="-0.583262"  sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.252614" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.184298"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=(3.4 * u.Angstrom).in_units(u.nm).value,
        sigma_C2=(3.4 * u.Angstrom).in_units(u.nm).value,
        sigma_F1=(2.8 * u.Angstrom).in_units(u.nm).value,
        sigma_H1=(2.4 * u.Angstrom).in_units(u.nm).value,
        epsilon_C1=(38.742 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_C2=(38.742 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_F1=(28.383 * u.K * u.kb).in_units("kJ/mol").value,
        epsilon_H1=(7.096 * u.K * u.kb).in_units("kJ/mol").value,
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
