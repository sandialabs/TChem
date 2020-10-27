# Input Files

TChem requires several input files to prescribe the modeling choices. For a gas-phase system the user provides (1) the reaction mechanisms and (2) thermal properties. Alternatively, these can be provided inside the same file with appropriate keyword selection. For the homogenous 0D ignition utility an additional file specifies the input state vectors and other modeling choices. For surface chemistry calculations, the surface chemistry model and the corresponding thermal properties can be specified in separate files or, similarly to the gas-phase chemistry case, in the same file, with appropriate keywords. Three more files are needed for the model problems with both gas and surface interface. In additional to the surface chemistry and thermodynamic properties' files, the parameters that specify the model problem are provided in a separate file.

## Reaction Mechanism Input File

TChem uses a type Chemkin Software input file. A complete description can be found in "Chemkin-II: A Fortran Chemical Kinetics Package for the Analysis of Gas-Phase Chemical Kinetics,' R. J. Kee, F. M. Rupley, and J. A. Miller, Sandia Report, SAND89-8009B (1995)." [(link)](https://www.osti.gov/biblio/5681118)

## Thermal Property Data (therm.dat)

TChem currently employs the 7-coefficient NASA polynomials. The format for the data input follows specifications in Table I in McBride *et al* (1993).

* Bonnie J McBride, Sanford Gordon, Martin A. Reno, ``Coefficients for Calculating Thermodynamic and Transport Properties of Individual Species,'' NASA Technical Memorandum 4513 (1993).

Support for 9-coefficient NASA polynomials is expected in the next TChem release.

## Input State Vectors  (sample.dat)
The format of the sample.dat file is:
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

## Surface Reaction Mechanism Input File and Thermal Property Data

TChem uses a the specifications in Coltrin *et al* (1996) for the input file for the surface reaction mechanism and thermodynamic properties. [(link)](https://www.osti.gov/biblio/481906-surface-chemkin-iii-fortran-package-analyzing-heterogeneous-chemical-kinetics-solid-surface-gas-phase-interface)

* Michael E. Coltrin, Robert J. Kee, Fran M. Rupley, Ellen Meeks, ``Surface Chemkin-III: A FORTRAN Package for Analyzing Heterogenous Chemical Kinetics at a Solid-surface--Gas-phase Interface,'' SANDIA Report SAND96-8217 (1996).

## Input Site Fraction (inputSurf.dat)
The format of the inputSurf.dat file is:
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
