# Input Files

TChem requires two input files to prescribe the choice of models. For a gas-phase system, a user provides (1) the reaction mechanisms, (2) the thermal properties, and (3) initial state vector. For surface chemistry, in addition to (1), (2), and (3), (4) the surface reaction mechanism, (5) the surface thermal properties, and (6) input site fraction vector. Alternatively, the kinetic model data can be provided in a single file with appropriate keyword selections.


## Reaction Mechanism and Thermal Property Data

TChem can parse kinetic model and thermal properties data in either [Chemkin](https://www.osti.gov/biblio/5681118) format or the [Cantera-Yaml](https://cantera.org/documentation/dev/sphinx/html/yaml/index.html) format. Additional information for the Chemkin formats can be found in the technical reports below:

 * R. J. Kee, F. M. Rupley, and J. A. Miller,Chemkin-II: A Fortran Chemical Kinetics Package for the Analysis of Gas-Phase Chemical Kinetics, Sandia Report, SAND89-8009B (1995). [(link)](https://www.osti.gov/biblio/5681118)

* Michael E. Coltrin, Robert J. Kee, Fran M. Rupley, Ellen Meeks, Surface Chemkin-III: A FORTRAN Package for Analyzing Heterogeneous Chemical Kinetics at a Solid-surface--Gas-phase Interface, Sandia Report SAND96-8217 (1996). [(link)](https://www.osti.gov/biblio/481906-surface-chemkin-iii-fortran-package-analyzing-heterogeneous-chemical-kinetics-solid-surface-gas-phase-interface)

We provide an example using the YAML input. See ``${TCHEM_REPOSITORY_PATH}/src/example/TChem_ThermalProperties_Yaml.cpp`` for more details.

## Input State Vectors  (sample.dat)

The set of specifications for gas-phase reaction examples are provided in a matrix form as:
````
T P SPECIES_NAME1 SPECIES_NAME2 ... SPECIES_NAMEN
T#1 P#1 Y1#1 Y2#1 ... YN#1 (sample #1)
T#2 P#2 Y1#2 Y2#2 ... YN#2 (sample #2)
...
...
...
T#N P#N Y1#N Y2#N ... YN#N (sample #N)
````   
Here T is the temperature [K], P is the pressure [Pa] and SPECIES_NAME1 is the name of the first gas species from the reaction mechanism input file. Y1#1 is the mass fraction of SPECIES_NAME1 in sample #1. The sum of the mass fractions on each row has to be equal to one since TChem does not normalize mass fractions. New samples can be created by adding rows to the input file. The excerpt below illustrates a setup for an example with 8 samples using a mixture of CH$_4$, O$_2$, N$_2$, and Ar:

````
T P CH4 O2 N2 AR
800 101325 1.48e-01 1.97e-01 6.43e-01 1.14e-02
800 101325 2.82e-02 2.25e-01 7.34e-01 1.30e-02
800 4559625 1.48e-01 1.97e-01 6.43e-01 1.14e-02
800 4559625 2.82e-02 2.25e-01 7.34e-01 1.30e-02
1250 101325 1.48e-01 1.97e-01 6.43e-01 1.14e-02
1250 101325 2.82e-02 2.25e-01 7.34e-01 1.30e-02
1250 4559625 1.48e-01 1.97e-01 6.43e-01 1.14e-02
1250 4559625 2.82e-02 2.25e-01 7.34e-01 1.30e-02
````
The eight samples in the above example correspond to the corners of a cube in a 3D parameter space with temperatures between 800 K and 1250 K, pressures between 1 atm to 45 atm, and equivalence ratios ($\phi$) for methane/air mixtures between 0.5 to 3.


## Input Site Fraction (inputSurf.dat)

Similarly, the set of specifications for surface reaction examples are provided in a matrix form as:
````
SURF_SPECIES_NAME_1 SURF_SPECIES_NAME_2 ... SURF_SPECIES_NAME_N
Z1#1 Z2#1 ... ZN#1 (sample #1)
Z1#2 Z2#2 ... ZN#2 (sample #2)
...
...
...
Z1#M Z2#M ... ZN#M (sample #M)
````  
where SURF_SPECIES_NAME1 is name of the first surface species from the chemSur.inp file and Z1#1 is the site fraction of this species for sample #1, and so forth.
