# TChem - A Software Toolkit for the Analysis of Complex Kinetic Models

1\.  [Introduction](#introduction)  
1.1\.  [Citing](#citing)  
1.2\.  [Nomenclature](#nomenclature)  
2\.  [Building TChem](#buildingtchem)  
2.1\.  [Download Libraries](#downloadlibraries)  
2.2\.  [Building Libraries and Configuring TChem](#buildinglibrariesandconfiguringtchem)  
2.2.1\.  [Kokkos](#kokkos)  
2.2.2\.  [KokkosKernels](#kokkoskernels)  
2.2.3\.  [GTEST](#gtest)  
2.2.4\.  [TChem](#tchem)  
3\.  [Input Files](#inputfiles)  
3.1\.  [Reaction Mechanism Input File](#reactionmechanisminputfile)  
3.2\.  [Thermal Property Data (therm.dat)](#thermalpropertydatatherm.dat)  
3.3\.  [Input State Vectors  (sample.dat)](#inputstatevectorssample.dat)  
3.4\.  [Surface Reaction Mechanism Input File and Thermal Property Data](#surfacereactionmechanisminputfileandthermalpropertydata)  
3.5\.  [Input Site Fraction (inputSurf.dat)](#inputsitefractioninputsurf.dat)  
4\.  [Thermodynamic Properties](#thermodynamicproperties)  
4.1\.  [Mass-Molar Conversions](#mass-molarconversions)  
4.2\.  [Equation of State](#equationofstate)  
4.3\.  [Gas-Phase Properties](#gas-phaseproperties)  
4.4\.  [Examples](#examples)  
4.5\.  [Surface Species Properties](#surfacespeciesproperties)  
5\.  [Reaction Rates](#reactionrates)  
5.1\.  [Gas-Phase Chemistry](#gas-phasechemistry)  
5.1.1\.  [Forward and Reverse Rate Constants](#forwardandreverserateconstants)  
5.1.2\.  [Concentration of the "Third-Body"](#concentrationofthe"third-body")  
5.1.3\.  [Pressure-dependent Reactions](#pressure-dependentreactions)  
5.1.4\.  [Note on Units for Net Production rates](#noteonunitsfornetproductionrates)  
5.1.5\.  [Example](#example)  
5.2\.  [Surface Chemistry](#surfacechemistry)  
5.2.1\.  [Forward and Reverse Rate Constants](#forwardandreverserateconstants-1)  
5.3\.  [Sticking Coefficients](#stickingcoefficients)  
5.3.1\.  [Note on Units for surface production rates](#noteonunitsforsurfaceproductionrates)  
5.3.2\.  [Example](#example-1)  
6\.  [Reactors](#reactors)  
6.1\.  [Time Integration](#timeintegration)  
6.1.1\.  [TrBDF2](#trbdf2)  
6.1.2\.  [Timestep Adaptivity](#timestepadaptivity)  
6.1.3\.  [Interface to Time Integrator](#interfacetotimeintegrator)  
6.2\.  [Homogenous Batch Reactors](#homogenousbatchreactors)  
6.2.1\.  [Problem Definition](#problemdefinition)  
6.2.2\.  [Jacobian Formulation](#jacobianformulation)  
6.2.2.1\.  [Evaluation of <img src="svgs/86aa7d769ae41b760b561beb8d611acb.svg?invert_in_darkmode" align=middle width=12.19759034999999pt height=30.267491100000004pt/> Components](#evaluationoftildejcomponents)  
6.2.2.1.1\.  [Efficient Evaluation of the <img src="svgs/86aa7d769ae41b760b561beb8d611acb.svg?invert_in_darkmode" align=middle width=12.19759034999999pt height=30.267491100000004pt/> Terms](#efficientevaluationofthetildejterms)  
6.2.2.2\.  [Evaluation of <img src="svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/> Components](#evaluationofjcomponents)  
6.2.3\.  [Running the 0D Ignition Utility](#runningthe0dignitionutility)  
6.2.4\.  [Ignition Delay Time Parameter Study for IsoOctane](#ignitiondelaytimeparameterstudyforisooctane)  
6.3\.  [Plug Flow Reactor (PFR) Problem with Gas and Surfaces Reactions](#plugflowreactorpfrproblemwithgasandsurfacesreactions)  
6.3.1\.  [Problem Definition](#problemdefinition-1)  
6.3.2\.  [Jacobian Formulation](#jacobianformulation-1)  
6.3.3\.  [Running the Plug Flow Reactor with Surface Reactions Utility](#runningtheplugflowreactorwithsurfacereactionsutility)  
6.3.4\.  [Initial Condition for PFR Problem](#initialconditionforpfrproblem)  
7\.  [Application Programming Interface](#applicationprogramminginterface)  
7.1\.  [C++](#c++)  
7.1.1\.  [Function List](#functionlist)  
7.1.1.1\.  [SpecificHeatCapacityPerMass](#specificheatcapacitypermass)  
7.1.1.2\.  [SpecificHeatCapacityConsVolumePerMass](#specificheatcapacityconsvolumepermass)  
7.1.1.3\.  [EnthalpyMass](#enthalpymass)  
7.1.1.4\.  [EntropyMass](#entropymass)  
7.1.1.5\.  [InternalEnergyMass](#internalenergymass)  
7.1.1.6\.  [NetProductionRatesPerMass](#netproductionratespermass)  
7.1.1.7\.  [NetProductionRatesPerMole](#netproductionratespermole)  
7.1.1.8\.  [NetProductionRateSurfacePerMass](#netproductionratesurfacepermass)  
7.1.1.9\.  [NetProductionRateSurfacePerMole](#netproductionratesurfacepermole)  
7.1.1.10\.  [Ignition 0D](#ignition0d)  
7.1.1.11\.  [PlugFlowReactor](#plugflowreactor)  
7.1.1.12\.  [SimpleSurface](#simplesurface)  
7.1.1.13\.  [InitialConditionSurface](#initialconditionsurface)  
7.1.1.14\.  [RateOfProgress](#rateofprogress)  
7.1.1.15\.  [SourceTerm](#sourceterm)  
7.1.1.16\.  [Smatrix](#smatrix)  
7.1.1.17\.  [IgnitionZeroDNumJacobian](#ignitionzerodnumjacobian)  
7.1.1.18\.  [JacobianReduced](#jacobianreduced)  
7.1.1.19\.  [PlugFlowReactorRHS](#plugflowreactorrhs)  
8\.  [On-going and Future Works](#on-goingandfutureworks)  
9\.  [Acknowledgement](#acknowledgement)  

<a name="introduction"></a>

## 1\. Introduction

TChem is an open source software library for solving complex computational chemistry problems and analyzing detailed chemical kinetic models. The software provides support for
* the support of complex kinetic models for gas-phase and surface chemistry,
* thermodynamic properties based on NASA polynomials,
* species production/consumption rates,
* stable time integrator for solving stiff time ordinary differential equations,
* reactor models such as homogenous gas-phase ignition (with analytical Jacobians), continuously stirred tank reactor, plug-flow reactor.


This toolkit builds upon earlier versions that were written in C and featured tools for gas-phase chemistry only. The current version of the software was completely refactored in C++, uses an object-oriented programming model, and adopts [Kokkos](https://github.com/kokkos) as its portability layer to make it ready for the next generation computing architectures i.e., multi/many core computing platforms with GPU accelerators. We have expanded the range of kinetic models to include surface chemistry and have added examples pertaining to Continuously Stirred Tank Reactors (CSTR) and Plug Flow Reactor (PFR) models to complement the homogenous ignition examples present in the earlier versions. To exploit the massive parallelism available from modern computing platforms, the current software interface is designed to evaluate samples in parallel, which enables large scale parametric studies, e.g. for sensitivity analysis and model calibration.



<a name="citing"></a>

### 1.1\. Citing

* Kyungjoo Kim, Oscar Diaz-Ibarra, Cosmin Safta, and Habib Najm, TChem v2.0 - A Software Toolkit for the Analysis of Complex Kinetic Models, Sandia National Laboratories, SAND 2020-10762, 2020.*

<a name="nomenclature"></a>

### 1.2\. Nomenclature

In the table below, <img src="svgs/e2b6809da50e776dc041bdae2b5704c1.svg?invert_in_darkmode" align=middle width=15.84100649999999pt height=14.15524440000002pt/> stands for reaction order, for the forward and reverse paths, respectively.

Symbol|Description|Units
--|--|--
<img src="svgs/259ce06b5bc89a23bcdcc86d59ca9a1b.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> | Number of species | -
<img src="svgs/83f96341b70056c1342f4e593867bf19.svg?invert_in_darkmode" align=middle width=38.90714519999999pt height=22.465723500000017pt/> | Number of reactions | -
<img src="svgs/fda26aae5e9a669fc3ca900427c61989.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> | number of gas-phase species |  -
<img src="svgs/a3be6c1dfffdf48fdac5347cf58d0e51.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> | number of surface species   |  -
<img src="svgs/939bfcf9c5e0a4fc239ac72428ac42d7.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> | number of surface species in phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> |  -
<img src="svgs/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode" align=middle width=8.49888434999999pt height=14.15524440000002pt/> | gas-phase density | kg/m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>
<img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/> | thermodynamic pressure | Pa
<img src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999991pt height=22.465723500000017pt/> | Temperature | K
<img src="svgs/7db7bb668a45f8c25263aa8d42c92f0e.svg?invert_in_darkmode" align=middle width=18.52532714999999pt height=22.465723500000017pt/>| Heat capacity at constant pressure| J/(K.kmol) |
<img src="svgs/97c3a943499dc4b5d2e307caa94e1750.svg?invert_in_darkmode" align=middle width=29.695490549999988pt height=22.465723500000017pt/> |  for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>| J/(K.kmol) |
<img src="svgs/718d4a9e99a84cde65c94ce39602d379.svg?invert_in_darkmode" align=middle width=13.89028244999999pt height=14.15524440000002pt/> |   specific| J/(K.kg) |
<img src="svgs/41050db2ee0425f7caa709b57ad8d518.svg?invert_in_darkmode" align=middle width=25.06044419999999pt height=14.15524440000002pt/> |  specific, for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>| J/(K.kg) |
<img src="svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> | Molar enthalpy of a mixture | J/kmol
<img src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode" align=middle width=11.027402099999989pt height=22.465723500000017pt/> | Molar entropy of a mixture | J/(kmol.K)
<img src="svgs/e0a50ae47e22e126ee04c138a9fc9abe.svg?invert_in_darkmode" align=middle width=16.80942944999999pt height=22.465723500000017pt/> | Mass fraction of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | -
<img src="svgs/1a35cf75b6c416e1e4a2b594e79040e6.svg?invert_in_darkmode" align=middle width=20.88478094999999pt height=22.465723500000017pt/> | Mole fraction of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | -
<img src="svgs/97c3a943499dc4b5d2e307caa94e1750.svg?invert_in_darkmode" align=middle width=29.695490549999988pt height=22.465723500000017pt/> | Heat capacity at constant pressure for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | J/(kmol.K)
<img src="svgs/599bfbec9094392a1a4ebef233875461.svg?invert_in_darkmode" align=middle width=20.93043149999999pt height=22.465723500000017pt/> | Molar enthalpy of <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> species | J/kmol
<img src="svgs/599bfbec9094392a1a4ebef233875461.svg?invert_in_darkmode" align=middle width=20.93043149999999pt height=22.465723500000017pt/> | for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | J/kmol|
<img src="svgs/65be2d4deaeb3d31a5ebfda5c600cda4.svg?invert_in_darkmode" align=middle width=16.24759289999999pt height=22.831056599999986pt/> |  specific| J/kg|
<img src="svgs/4e4a33f79ca2b33807aa3625e14c76e9.svg?invert_in_darkmode" align=middle width=27.41775464999999pt height=22.831056599999986pt/> |  specific, for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>| J/kg|
<img src="svgs/0656c827132046c3f6de4a1102b4aa0a.svg?invert_in_darkmode" align=middle width=17.345954999999993pt height=22.465723500000017pt/> | Molar entropy of <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> species | J/(kmol.K)
<img src="svgs/0656c827132046c3f6de4a1102b4aa0a.svg?invert_in_darkmode" align=middle width=17.345954999999993pt height=22.465723500000017pt/> |  for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>| J/(K.kmol)|
<img src="svgs/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.7054801999999905pt height=14.15524440000002pt/> |  specific| J/(K.kg)|
<img src="svgs/a609feccd3555ee26b37f32493b3eb0a.svg?invert_in_darkmode" align=middle width=14.97150929999999pt height=14.15524440000002pt/> |  specific, for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>| J/(K.kg)|
<img src="svgs/243ff7a534430724ea3bec1ed658c741.svg?invert_in_darkmode" align=middle width=20.190673799999992pt height=22.465723500000017pt/> | Gibbs free energy of <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> species | J/kmol
<img src="svgs/243ff7a534430724ea3bec1ed658c741.svg?invert_in_darkmode" align=middle width=20.190673799999992pt height=22.465723500000017pt/> | for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | J/kmol|
<img src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.430376349999989pt height=14.15524440000002pt/> |  specific| J/kg|
<img src="svgs/70bca5639e2ade10d623f74916cd912c.svg?invert_in_darkmode" align=middle width=15.10661129999999pt height=14.15524440000002pt/> |  specific, for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>| J/kg|
<img src="svgs/607980980b21998a867d017baf966d07.svg?invert_in_darkmode" align=middle width=19.08890444999999pt height=22.731165599999983pt/> | Molar concentration of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | kmol/m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>
<img src="svgs/e0a50ae47e22e126ee04c138a9fc9abe.svg?invert_in_darkmode" align=middle width=16.80942944999999pt height=22.465723500000017pt/> |  mass fraction of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>  | -|
<img src="svgs/1a35cf75b6c416e1e4a2b594e79040e6.svg?invert_in_darkmode" align=middle width=20.88478094999999pt height=22.465723500000017pt/> |  mole fraction of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>  | -|
<img src="svgs/3173677423a86e28caa6d954dbc490ff.svg?invert_in_darkmode" align=middle width=18.487510799999992pt height=22.465723500000017pt/> |  site fraction of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>  | -|
<img src="svgs/d75686d63e340b0222c77ac3a200d85d.svg?invert_in_darkmode" align=middle width=30.797324249999992pt height=34.337843099999986pt/> | for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> | -|
<img src="svgs/9458db33d70635758e0914dac4a04f2f.svg?invert_in_darkmode" align=middle width=18.400027799999993pt height=22.465723500000017pt/> | surface site density of phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> | kmol/m<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>|
<img src="svgs/0efad1c55006c28a6337b8775abe2b97.svg?invert_in_darkmode" align=middle width=40.13336084999999pt height=24.65753399999998pt/> |site occupancy by species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>| -|
<img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> | mixture molecular weight | kg/kmol|
<img src="svgs/1fedb120e317e2a9cd396c62f703d5e0.svg?invert_in_darkmode" align=middle width=22.79116289999999pt height=22.465723500000017pt/> | for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | kg/kmol|
<img src="svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> | universal gas constant | J/(kmol.K)|
<img src="svgs/dccf537697edeb21f9f48f2e853c2cab.svg?invert_in_darkmode" align=middle width=20.908639949999987pt height=22.831056599999986pt/> | Forward rate constant of <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> reaction | <img src="svgs/2c781ce15c67f00f33de98e04725b2cd.svg?invert_in_darkmode" align=middle width=97.8241968pt height=38.19539789999999pt/>
<img src="svgs/0f52f3e621b5a78f32a8c1c8503d6037.svg?invert_in_darkmode" align=middle width=19.66619984999999pt height=22.831056599999986pt/> | Reverse rate constant of <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> reaction | <img src="svgs/2c781ce15c67f00f33de98e04725b2cd.svg?invert_in_darkmode" align=middle width=97.8241968pt height=38.19539789999999pt/>
<img src="svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> | Universal gas constant | J/(kmol.K) |
<img src="svgs/bba9afc1720a6cfa67ec73c842e45be0.svg?invert_in_darkmode" align=middle width=11.989211849999991pt height=21.95701200000001pt/> | Rate of progress of <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> reaction | kmol/(m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>.s)
<img src="svgs/9766609a48282e6f30837a712595b37c.svg?invert_in_darkmode" align=middle width=13.16154179999999pt height=14.15524440000002pt/>| sticking coefficient for reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> | <img src="svgs/2c781ce15c67f00f33de98e04725b2cd.svg?invert_in_darkmode" align=middle width=97.8241968pt height=38.19539789999999pt/>|
<img src="svgs/09c6348357035f1b46742e4ee1fd2393.svg?invert_in_darkmode" align=middle width=17.49816089999999pt height=21.95701200000001pt/> | Production rate of <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> species | kmol/(m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>.s)
<img src="svgs/d7b5f57186d774508e5023c837be56a3.svg?invert_in_darkmode" align=middle width=14.97150929999999pt height=21.95701200000001pt/> | surface molar production rate of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> | kmol/(m<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>.s)|
<a name="buildingtchem"></a>

## 2\. Building TChem

TChem is designed and implemented using Kokkos (a performance portable parallel programming model) and it requires Kokkos and KokkosKernels. For testing, we use GTEST infrastructure. Additionally, it can use OpenBLAS or Intel MKL (more precisely we use CBLAS and LAPACKE interface from those libraries).

For convenience, we explain how to build the TChem code using the following environment variable that a user can modify according to their working environments.

```
/// repositories
export TCHEM_REPOSITORY_PATH=/where/you/clone/tchem/git/repo
export KOKKOS_REPOSITORY_PATH=/where/you/clone/kokkos/git/repo
export KOKKOSKERNELS_REPOSITORY_PATH=/where/you/clone/kokkoskernels/git/repo
export GTEST_REPOSITORY_PATH=/where/you/clone/gtest/git/repo

/// build directories
export TCHEM_BUILD_PATH=/where/you/build/tchem
export KOKKOS_BUILD_PATH=/where/you/build/kokkos
export KOKKOSKERNELS_BUILD_PATH=/where/you/build/kokkoskernels
export GTEST_BUILD_PATH=/where/you/build/gtest

/// install directories
export TCHEM_INSTALL_PATH=/where/you/install/tchem
export KOKKOS_INSTALL_PATH=/where/you/install/kokkos
export KOKKOSKERNELS_INSTALL_PATH=/where/you/install/kokkoskernels
export GTEST_INSTALL_PATH=/where/you/install/gtest
export OPENBLAS_INSTALL_PATH=/where/you/install/openblas
export LAPACKE_INSTALL_PATH=/where/you/install/lapacke
```

<a name="downloadlibraries"></a>

### 2.1\. Download Libraries

Clone Kokkos, KokkosKernels and TChem repositories. Note that we use the develop branch of Kokkos and KokkosKernels.

```
git clone getz.ca.sandia.gov:/home/gitroot/TChem++ ${TCHEM_REPOSITORY_PATH};
git clone https://github.com/kokkos/kokkos.git ${KOKKOS_REPOSITORY_PATH};
cd ${KOKKOS_REPOSITORY_PATH}; git checkout --track origin/develop;
git clone https://github.com/kokkos/kokkos-kernels.git ${KOKKOSKERNELS_REPOSITORY_PATH};
cd ${KOKKOSKERNELS_REPOSITORY_PATH}; git checkout --track origin/develop;
git clone https://github.com/google/googletest.git ${GTEST_REPOSITORY_PATH}
```

Here, we compile and install the TPLs separately; then, compile TChem against those installed TPLs.

<a name="buildinglibrariesandconfiguringtchem"></a>

### 2.2\. Building Libraries and Configuring TChem

<a name="kokkos"></a>

#### 2.2.1\. Kokkos

This example build Kokkos on Intel Sandybridge architectures and install it to ``${KOKKOS_INSTALL_PATH}``. For more details, see [Kokkos github pages](https://github.com/kokkos/kokkos).

```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
```

To compile for NVIDIA GPUs, one can customize the following cmake script. Note that we use Kokkos ``nvcc_wrapper`` as its compiler. The architecture flag indicates that the host architecture is Intel SandyBridge and the GPU architecture is Volta 70 generation. With Kokkko 3.1, the CUDA architecture flag is optional (the script automatically detects a correct CUDA arch flag).
```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${KOKKOS_REPOSITORY_PATH}/bin/nvcc_wrapper"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA:BOOL=ON \
    -D Kokkos_ENABLE_CUDA_UVM:BOOL=OFF \
    -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_VOLTA70=ON \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
```

<a name="kokkoskernels"></a>

#### 2.2.2\. KokkosKernels

Compiling KokkosKernels follows Kokkos configuration of which information is available at ``${KOKKOS_INSTALL_PATH}``.

```
cd ${KOKKOSKERNELS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOSKERNELS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    -D CMAKE_CXX_FLAGS="-g"  \
    -D KokkosKernels_INST_LAYOUTRIGHT:BOOL=ON \
    -D Kokkos_DIR="${KOKKOS_INSTALL_PATH}/lib64/cmake/Kokkos" \
    -D KokkosKernels_ENABLE_TPL_LAPACKE:BOOL=ON \
    -D KokkosKernels_ENABLE_TPL_CBLAS:BOOL=ON \
    -D CBLAS_INCLUDE_DIRS="/opt/local/include" \
    ${KOKKOSKERNELS_REPOSITORY_PATH}
make -j install
```

For GPUs, the compiler is changed with ``nvcc_wrapper`` by adding ``-D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"``.

<a name="gtest"></a>

#### 2.2.3\. GTEST

We use GTEST as our testing infrastructure. With the following cmake script, the GTEST can be compiled and installed.

```
cd ${GTEST_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${GTEST_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    ${GTEST_REPOSITORY_PATH}
make -j install
```

<a name="tchem"></a>

#### 2.2.4\. TChem

The following example cmake script compiles TChem on host linking with the libraries described in the above e.g., kokkos, kokkoskernels, gtest and openblas. The openblas and lapacke libraries are required on a host device providing an optimized version of dense linear algebra library. With an Intel compiler, one can replace these libraries with Intel MKL by adding an option ``TCHEM_ENABLE_MKL=ON`` instead of using openblas and lapacke. On Mac OSX, we use the openblas library managed by **macports**. This version of openblas has different header names and we need to distinguish this version of the code from others which are typically used in linux distributions. To discern the two version of the code, cmake looks for ``cblas_openblas.h`` to tell that the installed version is from MacPort. This mechanism can be broken if MacPort openblas is changed later. The macport openblas version include lapacke interface and one can remove ``LAPACKE_INSTALL_PATH`` from the configure script.
```
cd ${TCHEM_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${TCHEM_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D TCHEM_ENABLE_VERBOSE=OFF \
    -D TCHEM_ENABLE_KOKKOS=ON \
    -D TCHEM_ENABLE_KOKKOSKERNELS=ON \
    -D TCHEM_ENABLE_TEST=ON \
    -D TCHEM_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH="${KOKKOS_INSTALL_PATH}" \
    -D KOKKOSKERNELS_INSTALL_PATH="${KOKKOSKERNELS_INSTALL_PATH}" \
    -D OPENBLAS_INSTALL_PATH="${OPENBLAS_INSTALL_PATH}" \
    -D LAPACKE_INSTALL_PATH="${LAPACKE_INSTALL_PATH}" \
    -D GTEST_INSTALL_PATH="${GTEST_INSTALL_PATH}" \
    ${TCHEM_SRC_PATH}
make -j install
```
For GPUs, we can use the above cmake script replacing the compiler with ``nvcc_wrapper`` by adding ``-D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"``.
<a name="inputfiles"></a>

## 3\. Input Files

TChem requires several input files to prescribe the modeling choices. For a gas-phase system the user provides (1) the reaction mechanisms and (2) thermal properties. Alternatively, these can be provided inside the same file with appropriate keyword selection. For the homogenous 0D ignition utility an additional file specifies the input state vectors and other modeling choices. For surface chemistry calculations, the surface chemistry model and the corresponding thermal properties can be specified in separate files or, similarly to the gas-phase chemistry case, in the same file, with appropriate keywords. Three more files are needed for the model problems with both gas and surface interface. In additional to the surface chemistry and thermodynamic properties' files, the parameters that specify the model problem are provided in a separate file.

<a name="reactionmechanisminputfile"></a>

### 3.1\. Reaction Mechanism Input File

TChem uses a type Chemkin Software input file. A complete description can be found in "Chemkin-II: A Fortran Chemical Kinetics Package for the Analysis of Gas-Phase Chemical Kinetics,' R. J. Kee, F. M. Rupley, and J. A. Miller, Sandia Report, SAND89-8009B (1995)." [(link)](https://www.osti.gov/biblio/5681118)

<a name="thermalpropertydatatherm.dat"></a>

### 3.2\. Thermal Property Data (therm.dat)

TChem currently employs the 7-coefficient NASA polynomials. The format for the data input follows specifications in Table I in McBride *et al* (1993).

* Bonnie J McBride, Sanford Gordon, Martin A. Reno, ``Coefficients for Calculating Thermodynamic and Transport Properties of Individual Species,'' NASA Technical Memorandum 4513 (1993).

Support for 9-coefficient NASA polynomials is expected in the next TChem release.

<a name="inputstatevectorssample.dat"></a>

### 3.3\. Input State Vectors  (sample.dat)
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
Here T is the temperature [K], P is the pressure [Pa] and SPECIES_NAME1 is the name of the first gas species from the reaction mechanism input file. Y1#1 is the mass fraction of SPECIES_NAME1 in sample #1. The sum of the mass fractions on each row has to be equal to one since TChem does not normalize mass fractions. New samples can be created by adding rows to the input file. The excerpt below illustrates a setup for an example with 8 samples using a mixture of CH<img src="svgs/5dfc3ab84de9c94bbfee2e75b72e1184.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=14.15524440000002pt/>, O<img src="svgs/10f8f9bf55a697fc978ffe2990e3209d.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=14.15524440000002pt/>, N<img src="svgs/10f8f9bf55a697fc978ffe2990e3209d.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=14.15524440000002pt/>, and Ar:

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
The eight samples in the above example correspond to the corners of a cube in a 3D parameter space with temperatures between 800 K and 1250 K, pressures between 1 atm to 45 atm, and equivalence ratios (<img src="svgs/f50853d41be7d55874e952eb0d80c53e.svg?invert_in_darkmode" align=middle width=9.794543549999991pt height=22.831056599999986pt/>) for methane/air mixtures between 0.5 to 3.

<a name="surfacereactionmechanisminputfileandthermalpropertydata"></a>

### 3.4\. Surface Reaction Mechanism Input File and Thermal Property Data

TChem uses a the specifications in Coltrin *et al* (1996) for the input file for the surface reaction mechanism and thermodynamic properties. [(link)](https://www.osti.gov/biblio/481906-surface-chemkin-iii-fortran-package-analyzing-heterogeneous-chemical-kinetics-solid-surface-gas-phase-interface)

* Michael E. Coltrin, Robert J. Kee, Fran M. Rupley, Ellen Meeks, ``Surface Chemkin-III: A FORTRAN Package for Analyzing Heterogenous Chemical Kinetics at a Solid-surface--Gas-phase Interface,'' SANDIA Report SAND96-8217 (1996).

<a name="inputsitefractioninputsurf.dat"></a>

### 3.5\. Input Site Fraction (inputSurf.dat)
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
<a name="thermodynamicproperties"></a>

## 4\. Thermodynamic Properties

We first present conversion formulas and the gas-phase equation of state, followed by a description of molar and mass-based expression for several thermodynamic properties.

<a name="mass-molarconversions"></a>

### 4.1\. Mass-Molar Conversions

The molar mass of the mixture, <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> is computed as
<p align="center"><img src="svgs/15bbefd6fe070a46e39f5f2ca587ed7d.svg?invert_in_darkmode" align=middle width=268.17991035pt height=59.1786591pt/></p>

where <img src="svgs/1a35cf75b6c416e1e4a2b594e79040e6.svg?invert_in_darkmode" align=middle width=20.88478094999999pt height=22.465723500000017pt/> and <img src="svgs/e0a50ae47e22e126ee04c138a9fc9abe.svg?invert_in_darkmode" align=middle width=16.80942944999999pt height=22.465723500000017pt/> are the mole and mass fractions, respectively, of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, and <img src="svgs/b09bc86177601e586d14fb48ca4cb31d.svg?invert_in_darkmode" align=middle width=22.79116289999999pt height=22.465723500000017pt/> is the molecular weight of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>. Mass and mole fractions can be computed from each other as
<p align="center"><img src="svgs/9e95c5a8ca5130aad552b63e4d7d342f.svg?invert_in_darkmode" align=middle width=232.75112024999999pt height=16.438356pt/></p>

The the molar concentration of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is given by <img src="svgs/854ee57c45dfe5541364c526c63fc0c7.svg?invert_in_darkmode" align=middle width=177.94164629999997pt height=24.65753399999998pt/>, and the molar concentration of the mixture is given by
<p align="center"><img src="svgs/cbfe59f451914a48a7d1311e40487eeb.svg?invert_in_darkmode" align=middle width=112.8186543pt height=49.96363845pt/></p>


For problems that include heterogenous chemistry, the site fractions <img src="svgs/3173677423a86e28caa6d954dbc490ff.svg?invert_in_darkmode" align=middle width=18.487510799999992pt height=22.465723500000017pt/> describe the composition of species on the surface. The number of surface phases is denoted by <img src="svgs/a90dab66c08841e0f3f61e4042f7733a.svg?invert_in_darkmode" align=middle width=47.25194264999998pt height=22.465723500000017pt/> and the site fractions are normalized with respect to each phase.
<p align="center"><img src="svgs/c57c890612dc428007654e0f248e3dae.svg?invert_in_darkmode" align=middle width=255.29248635000002pt height=52.2439797pt/></p>

Here, <img src="svgs/939bfcf9c5e0a4fc239ac72428ac42d7.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> is the number of species on surface phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>. TChem currently handles <img src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/> surface phase only, <img src="svgs/3b38266cee9b3164990cb71d4d23bd70.svg?invert_in_darkmode" align=middle width=78.21064844999998pt height=22.465723500000017pt/>. The surface concentration of surface species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is given by

<p align="center"><img src="svgs/94970833d1b97187335330656fa9d589.svg?invert_in_darkmode" align=middle width=142.3007124pt height=22.9224534pt/></p>

where <img src="svgs/9458db33d70635758e0914dac4a04f2f.svg?invert_in_darkmode" align=middle width=18.400027799999993pt height=22.465723500000017pt/> is the surface site density of surface phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> and <img src="svgs/ae7f10b32ab01ef5567d34322bacdc17.svg?invert_in_darkmode" align=middle width=40.13336084999999pt height=24.65753399999998pt/> is the site occupancy number for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>. <img src="svgs/ae7f10b32ab01ef5567d34322bacdc17.svg?invert_in_darkmode" align=middle width=40.13336084999999pt height=24.65753399999998pt/> represents the number of sites in phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> occupied by species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>.


<a name="equationofstate"></a>

### 4.2\. Equation of State    

The ideal gas equation of state is used throughout the library,
<p align="center"><img src="svgs/4a6ed8460192dc853d7841074495a0c7.svg?invert_in_darkmode" align=middle width=510.58399589999993pt height=59.1786591pt/></p>

where <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/> is the thermodynamic pressure, <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> and <img src="svgs/b09bc86177601e586d14fb48ca4cb31d.svg?invert_in_darkmode" align=middle width=22.79116289999999pt height=22.465723500000017pt/> are the molecular weights of the mixture and of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, respectively, <img src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999991pt height=22.465723500000017pt/> is the temperature, and <img src="svgs/607980980b21998a867d017baf966d07.svg?invert_in_darkmode" align=middle width=19.08890444999999pt height=22.731165599999983pt/> is the molar concentration of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>.


<a name="gas-phaseproperties"></a>

### 4.3\. Gas-Phase Properties

The standard-state thermodynamic properties for a thermally perfect gas are computed based on NASA polynomials \cite{McBride:1993}. The molar heat capacity at constant pressure for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is computed as
<p align="center"><img src="svgs/9e1bdc871221c59e0d0e089c5c10b9a6.svg?invert_in_darkmode" align=middle width=359.59422674999996pt height=33.62942055pt/></p>

where <img src="svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> the universal gas constant. The molar enthalpy is computed as
<p align="center"><img src="svgs/631987758e054abec9f4bfbf32d108ca.svg?invert_in_darkmode" align=middle width=635.7452095499999pt height=42.79493295pt/></p>

The molar entropy is given by
<p align="center"><img src="svgs/7d5884aeae96235ea7240df02bc3207c.svg?invert_in_darkmode" align=middle width=625.6138844999999pt height=42.79493295pt/></p>

The temperature units are Kelvin in the polynomial expressions above. Other thermodynamics properties are computed based on the polynomial fits above. The molar heat capacity at constant volume <img src="svgs/966dfb743557c8028b07ceaffee38676.svg?invert_in_darkmode" align=middle width=29.907271349999988pt height=22.465723500000017pt/>, the internal energy <img src="svgs/4a5617bf7bfc59a2c01ea4f011dbc8d0.svg?invert_in_darkmode" align=middle width=18.48976799999999pt height=22.465723500000017pt/>, and the Gibbs free energy <img src="svgs/243ff7a534430724ea3bec1ed658c741.svg?invert_in_darkmode" align=middle width=20.190673799999992pt height=22.465723500000017pt/>:
<p align="center"><img src="svgs/62f4549a7d43bae530b2e4900457745a.svg?invert_in_darkmode" align=middle width=367.8139179pt height=18.905967299999997pt/></p>

The mixture properties in molar units are given by
<p align="center"><img src="svgs/cd175e30e5375f7a0712fedd7f396d3e.svg?invert_in_darkmode" align=middle width=535.3380747pt height=49.96363845pt/></p>

where <img src="svgs/1a35cf75b6c416e1e4a2b594e79040e6.svg?invert_in_darkmode" align=middle width=20.88478094999999pt height=22.465723500000017pt/> the mole fraction of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>. The entropy and Gibbs free energy for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> account for the entropy of mixing and thermodynamic pressure
<p align="center"><img src="svgs/9e9459b17ace4c7311b355cc0844f724.svg?invert_in_darkmode" align=middle width=307.148622pt height=36.09514755pt/></p>

The mixture values for these properties are computed as above
<p align="center"><img src="svgs/0b97fdedec9e3b81bcd7b36e74c3455e.svg?invert_in_darkmode" align=middle width=238.01177894999998pt height=49.96363845pt/></p>

The specific thermodynamic properties in mass units are obtained by dividing the above expression by the species molecular weight, <img src="svgs/b09bc86177601e586d14fb48ca4cb31d.svg?invert_in_darkmode" align=middle width=22.79116289999999pt height=22.465723500000017pt/>,
<p align="center"><img src="svgs/cfaafb368aad9f061dc8cd9d7de657c7.svg?invert_in_darkmode" align=middle width=659.78253165pt height=18.905967299999997pt/></p>

and
<p align="center"><img src="svgs/565d218ec44f03073f47d33fb1bc89ab.svg?invert_in_darkmode" align=middle width=193.10518754999998pt height=16.438356pt/></p>

For the thermodynamic properties in mass units the mixture properties are given by
<p align="center"><img src="svgs/fcf5b27f967e9d70adfae49a1956e74a.svg?invert_in_darkmode" align=middle width=716.2883739pt height=49.96363845pt/></p>

where <img src="svgs/e0a50ae47e22e126ee04c138a9fc9abe.svg?invert_in_darkmode" align=middle width=16.80942944999999pt height=22.465723500000017pt/> the mass fraction of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>.

The mixture properties in mass units can also be evaluated from the equivalent molar properties as
<p align="center"><img src="svgs/607e40189eac71b30f6dbf57c6c5b447.svg?invert_in_darkmode" align=middle width=512.4719984999999pt height=17.031940199999998pt/></p>

where <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> is the molecular weight of the mixture.
<a name="examples"></a>

### 4.4\. Examples

A example to compute <img src="svgs/718d4a9e99a84cde65c94ce39602d379.svg?invert_in_darkmode" align=middle width=13.89028244999999pt height=14.15524440000002pt/> and <img src="svgs/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode" align=middle width=9.47111549999999pt height=22.831056599999986pt/> in mass base is at "example/TChem_ThermalProperties.cpp". Enthalpy per species and the mixture enthalpy are computed with this [function call](#cxx-api-EnthalpyMass). Heat capacity per species and mixture with this [function call](#cxx-api-SpecificHeatCapacityPerMass). This example can be used in bath mode, and several sample are compute in one run. The next two figures were compute with 40000 samples changing temperature and equivalent ratio for methane/air mixtures.

![Enthalpy](src/markdown/Figures/gri3.0_OneSample/MixtureEnthalpy.jpg)
Figure. Mixture Enthalpy compute with gri3.0 mechanism.

![SpecificHeatCapacity](src/markdown/Figures/gri3.0_OneSample/MixtureSpecificHeatCapacity.jpg)
Figure.  Mixutre Specific Heat Capacity <img src="svgs/7db7bb668a45f8c25263aa8d42c92f0e.svg?invert_in_darkmode" align=middle width=18.52532714999999pt height=22.465723500000017pt/> compute with gri3.0 mechanism.


<a name="surfacespeciesproperties"></a>

### 4.5\. Surface Species Properties

The thermal properties of the surface species are computed with the same equation used by the gas phase describe above.

<!-- ## Examples -->
<a name="reactionrates"></a>

## 5\. Reaction Rates

In this chapter we present reaction rate expressions for gas-phase reactions in [Section](#gas-phasechemistry) and for surface species or between surface and gas-phase species in [Section](#surface chemistry).


<a name="gas-phasechemistry"></a>

### 5.1\. [Gas-Phase Chemistry](#cxx-api-ReactionRates)

The production rate for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in molar units is written as
<p align="center"><img src="svgs/e13aa4c27397b9f67cb186705363f6b3.svg?invert_in_darkmode" align=middle width=235.62922514999997pt height=47.988758399999995pt/></p>

where <img src="svgs/83f96341b70056c1342f4e593867bf19.svg?invert_in_darkmode" align=middle width=38.90714519999999pt height=22.465723500000017pt/> is the number of reactions and <img src="svgs/8d69b56f74dd3aa880ae9044c2b9ef61.svg?invert_in_darkmode" align=middle width=20.03720234999999pt height=24.7161288pt/> and <img src="svgs/0e34d813eb8572232bb709f72638f649.svg?invert_in_darkmode" align=middle width=20.03720234999999pt height=24.7161288pt/> are the stoichiometric coefficients of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> for the reactant and product side of the reaction, respectively. The rate-of-progress of reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> is <img src="svgs/d2009378d3ea828d5c4239c244cb6b3d.svg?invert_in_darkmode" align=middle width=67.43927849999999pt height=22.465723500000017pt/>, with

<img src="svgs/e680843a73d5214f63af89c487342755.svg?invert_in_darkmode" align=middle width=13.30617584999999pt height=22.465723500000017pt/>|Reaction Type
--|--|--
<img src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | basic reaction
<img src="svgs/1929f0d333ecacfc356b41b810e0e3ef.svg?invert_in_darkmode" align=middle width=16.47377489999999pt height=22.731165599999983pt/> | 3-rd body enhanced, no pressure dependence
<img src="svgs/6dcd557e98383f2d4b919d66c9c25071.svg?invert_in_darkmode" align=middle width=53.01827685pt height=29.205422400000014pt/> | unimolecular/recombination fall-off reactions
<img src="svgs/7691c53f9aa1213707a59ead5607e1cc.svg?invert_in_darkmode" align=middle width=53.01827685pt height=27.77565449999998pt/> | chemically activated bimolecular reactions

and

<p align="center"><img src="svgs/e731ad6c99fdbf568493d486b38bcf0d.svg?invert_in_darkmode" align=middle width=241.56291225pt height=51.82436325pt/></p>

The above expressions are detailed below.

<a name="forwardandreverserateconstants"></a>

#### 5.1.1\. Forward and Reverse Rate Constants

The forward rate constant has typically an Arrhenius expression,
<p align="center"><img src="svgs/d3b0c006aae01017a67a2471e77eb110.svg?invert_in_darkmode" align=middle width=190.97296845pt height=39.452455349999994pt/></p>

where <img src="svgs/4ebf880807deff5796460f39aea46f80.svg?invert_in_darkmode" align=middle width=16.97969789999999pt height=22.465723500000017pt/>, <img src="svgs/3d13090ef3ed1448f3c4dc166d06ab4d.svg?invert_in_darkmode" align=middle width=13.948864049999989pt height=22.831056599999986pt/>, and <img src="svgs/97790d793f190b3b985b582fea9ceb20.svg?invert_in_darkmode" align=middle width=16.78561829999999pt height=22.465723500000017pt/> are the pre-exponential factor, temperature exponent, and activation energy, respectively, for reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. For reactions with reverse Arrhenius parameters specified, the reverse rate constant <img src="svgs/d99ec1601508d72f39e8f229c0784c89.svg?invert_in_darkmode" align=middle width=20.488092899999987pt height=22.831056599999986pt/> is computed similar to <img src="svgs/c401619f66d3a4a96392fe1c60a5e6ed.svg?invert_in_darkmode" align=middle width=21.73055114999999pt height=22.831056599999986pt/>. If the reverse Arrhenius parameters are not specified, <img src="svgs/d99ec1601508d72f39e8f229c0784c89.svg?invert_in_darkmode" align=middle width=20.488092899999987pt height=22.831056599999986pt/> is computed as
<p align="center"><img src="svgs/900a2560791ec74560b5de36b95f985b.svg?invert_in_darkmode" align=middle width=104.6960574pt height=17.8538118pt/></p>

where <img src="svgs/402a64748a7c548103e8a6157b535764.svg?invert_in_darkmode" align=middle width=25.308656999999986pt height=22.465723500000017pt/> is the equilibrium constant (in concentration units) for reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>
<p align="center"><img src="svgs/1182793fd8c0d5cf203436e2b98a19a9.svg?invert_in_darkmode" align=middle width=492.43297455pt height=59.1786591pt/></p>

When computing the equilibrium constant, the atmospheric pressure, <img src="svgs/6619f6364c444f53aaaeb589c503980d.svg?invert_in_darkmode" align=middle width=65.27339939999999pt height=22.465723500000017pt/>atm, and the universal gas constant <img src="svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> are converted to cgs units, dynes/cm<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/> and erg/(mol.K), respectively.

Note: If a reaction is irreversible, <img src="svgs/7f87b958a998866226cc01233172402e.svg?invert_in_darkmode" align=middle width=45.97403579999999pt height=22.831056599999986pt/>.

<a name="concentrationofthe"third-body""></a>

#### 5.1.2\. Concentration of the "Third-Body"   

If the expression "+M" is present in the reaction string, some of the species might have custom efficiencies for their contribution in the mixture. For these reactions, the mixture concentration is computed as


<p align="center"><img src="svgs/d87720ec5bd4c339560b90876020eeb3.svg?invert_in_darkmode" align=middle width=121.08558330000001pt height=51.82436325pt/></p>

where <img src="svgs/8175b4b012861c57d7f99a503fdcaa72.svg?invert_in_darkmode" align=middle width=21.27105584999999pt height=14.15524440000002pt/> is the efficiency of species <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/> in reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and <img src="svgs/0983467432bb3b4f9708f334be19484b.svg?invert_in_darkmode" align=middle width=17.92738529999999pt height=22.731165599999983pt/> is the concentration of species <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/>. <img src="svgs/8175b4b012861c57d7f99a503fdcaa72.svg?invert_in_darkmode" align=middle width=21.27105584999999pt height=14.15524440000002pt/> coefficients are set to 1 unless specified in the kinetic model description.

<a name="pressure-dependentreactions"></a>

#### 5.1.3\. Pressure-dependent Reactions

* Reduced pressure <img src="svgs/0901b44a21af681b03ff0f22d16152bb.svg?invert_in_darkmode" align=middle width=22.27650644999999pt height=22.465723500000017pt/>. If expression "(+M)" is used to describe a reaction, then <img src="svgs/c8e50fa593d6aa9683d8ca7b539649b5.svg?invert_in_darkmode" align=middle width=89.63996744999999pt height=31.10494860000002pt/>.
* For reactions that contain expressions like "(+<img src="svgs/a8a6d2775483063fe212034bd95c34ff.svg?invert_in_darkmode" align=middle width=24.45028409999999pt height=22.465723500000017pt/>)" (<img src="svgs/a8a6d2775483063fe212034bd95c34ff.svg?invert_in_darkmode" align=middle width=24.45028409999999pt height=22.465723500000017pt/> is the name of species <img src="svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/>), the reduced pressure is computed as <img src="svgs/22ee389e4632e86737bc6179d55fa8cf.svg?invert_in_darkmode" align=middle width=96.65391779999999pt height=31.10494860000002pt/>.

For *unimolecular/recombination fall-off reactions* the Arrhenius parameters for the high-pressure limit rate constant <img src="svgs/1ef2bb61a0b900d70f5bb5878877524b.svg?invert_in_darkmode" align=middle width=21.66295889999999pt height=22.831056599999986pt/> are given on the reaction line, while the parameters for the low-pressure limit rate constant <img src="svgs/1b63f35b880a6307974273a3ff6063c9.svg?invert_in_darkmode" align=middle width=15.11042279999999pt height=22.831056599999986pt/> are given on the auxiliary reaction line that contains the keyword **LOW**. For *chemically activated bimolecular reactions* the parameters for <img src="svgs/1b63f35b880a6307974273a3ff6063c9.svg?invert_in_darkmode" align=middle width=15.11042279999999pt height=22.831056599999986pt/> are given on the reaction line while the parameters for <img src="svgs/1ef2bb61a0b900d70f5bb5878877524b.svg?invert_in_darkmode" align=middle width=21.66295889999999pt height=22.831056599999986pt/> are given on the auxiliary reaction line that contains the keyword **HIGH**.

The following expressions are employed to compute the <img src="svgs/e17c35f619f835117e1ff8e25d5f8a9c.svg?invert_in_darkmode" align=middle width=15.22169714999999pt height=22.465723500000017pt/>:
<img src="svgs/e17c35f619f835117e1ff8e25d5f8a9c.svg?invert_in_darkmode" align=middle width=15.22169714999999pt height=22.465723500000017pt/>|Reaction Type
--|--
<img src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | Lindemann reaction
<img src="svgs/322293f4c8de899c7240408bf945d8fd.svg?invert_in_darkmode" align=middle width=99.37438334999997pt height=44.200856699999996pt/> | Troe reaction
<img src="svgs/e7c6f613d22ebb4044633031821c56e2.svg?invert_in_darkmode" align=middle width=224.14090379999993pt height=35.5436301pt/> | SRI reaction

* For the Troe form, <img src="svgs/dc3da706b6c446fa1688c99721a8004b.svg?invert_in_darkmode" align=middle width=35.774184599999984pt height=22.465723500000017pt/>, <img src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/>, and <img src="svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.29340979999999pt height=22.465723500000017pt/> are
<p align="center"><img src="svgs/103d55e68b9a670bd7dabe5f14c12b76.svg?invert_in_darkmode" align=middle width=454.85663354999997pt height=39.452455349999994pt/></p>

<p align="center"><img src="svgs/f4f27006386e8ea611bad44a2da0fd14.svg?invert_in_darkmode" align=middle width=606.28904985pt height=15.43376505pt/></p>

Parameters <img src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.68915409999999pt height=14.15524440000002pt/>, <img src="svgs/f53bd91c343fed10d5c829d244104c4d.svg?invert_in_darkmode" align=middle width=32.09489414999999pt height=22.63846199999998pt/>, <img src="svgs/e8d7ad864665e4937272cf42ee61b3de.svg?invert_in_darkmode" align=middle width=18.62450534999999pt height=22.63846199999998pt/>, and <img src="svgs/a0f611fb7f1a79d56d386bd563021941.svg?invert_in_darkmode" align=middle width=25.359699749999994pt height=22.63846199999998pt/> are provided (in this order) in the kinetic model description for each Troe-type reaction. If <img src="svgs/a0f611fb7f1a79d56d386bd563021941.svg?invert_in_darkmode" align=middle width=25.359699749999994pt height=22.63846199999998pt/> is omitted, only the first two terms are used to compute <img src="svgs/dc3da706b6c446fa1688c99721a8004b.svg?invert_in_darkmode" align=middle width=35.774184599999984pt height=22.465723500000017pt/>.
* For the SRI form exponent <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> is computed as
<p align="center"><img src="svgs/c3ba58a2116b314428261bf36e9116b5.svg?invert_in_darkmode" align=middle width=196.82641439999998pt height=42.80407395pt/></p>

Parameters <img src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.68915409999999pt height=14.15524440000002pt/>, <img src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054796099999991pt height=22.831056599999986pt/>, <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.11380504999999pt height=14.15524440000002pt/>, <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/>, and <img src="svgs/8cd34385ed61aca950a6b06d09fb50ac.svg?invert_in_darkmode" align=middle width=7.654137149999991pt height=14.15524440000002pt/> are provided in the kinetic model description for each SRI-type reaction. If <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/> and <img src="svgs/8cd34385ed61aca950a6b06d09fb50ac.svg?invert_in_darkmode" align=middle width=7.654137149999991pt height=14.15524440000002pt/> are omitted, these parameters are set to <img src="svgs/4cfe92e893a7541f68473ecb08419237.svg?invert_in_darkmode" align=middle width=38.69280359999998pt height=22.831056599999986pt/> and <img src="svgs/a959dddbf8d19a8a7e96404006dc8941.svg?invert_in_darkmode" align=middle width=37.79097794999999pt height=21.18721440000001pt/>.

Miller~\cite{PLOGprinceton} has developed an alternative expressionfor the pressure dependence for pressure fall-off reactions that cannot be fitted with a single Arrhenius rate expression. This approach employs linear interpolation of <img src="svgs/9ed7bbf06bf98221bc2cbe38991f9f8a.svg?invert_in_darkmode" align=middle width=45.70312064999999pt height=22.831056599999986pt/> as a function of pressure for reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> as follows
<p align="center"><img src="svgs/309712647dd768282d52b8dd39b58e39.svg?invert_in_darkmode" align=middle width=492.30777915pt height=41.6135775pt/></p>

Here, <img src="svgs/365a2e8e2120634e7f4a962f13a3acac.svg?invert_in_darkmode" align=middle width=225.82180829999996pt height=37.80850590000001pt/> is the Arrhenius rate corresponding to pressure <img src="svgs/c59566c48fcebc8309e5140360f10c00.svg?invert_in_darkmode" align=middle width=12.49435604999999pt height=14.15524440000002pt/>. For <img src="svgs/fefb7c0bcbb6c76c07b05834d0106b0d.svg?invert_in_darkmode" align=middle width=45.01131029999999pt height=17.723762100000005pt/> the Arrhenius rate is set to <img src="svgs/21663a2f28e9f47c55377a9395652923.svg?invert_in_darkmode" align=middle width=76.65731039999999pt height=22.831056599999986pt/>, and similar for <img src="svgs/86f1d37bd3330b95361d54e5c11db386.svg?invert_in_darkmode" align=middle width=53.256447749999985pt height=17.723762100000005pt/> where <img src="svgs/3bf9c1fe4273ed003fd49e744378a5ac.svg?invert_in_darkmode" align=middle width=17.85866609999999pt height=22.465723500000017pt/> is the number of pressures for which the Arrhenius factors are provided, for reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. This formulation can be combined with 3<img src="svgs/a8306226788e6b4f1c4e0e15bb3b0120.svg?invert_in_darkmode" align=middle width=12.39732779999999pt height=27.91243950000002pt/> body information, e.g. <img src="svgs/a5dda47c08c32d1164e21afd3abdd059.svg?invert_in_darkmode" align=middle width=52.51947854999999pt height=22.731165599999983pt/>.

<a name="noteonunitsfornetproductionrates"></a>

#### 5.1.4\. Note on Units for Net Production rates

In most cases, the kinetic models input files contain parameters that are based on *calories, cm, moles, kelvin, seconds*. The mixture temperature and species molar concentrations are necessary to compute the reaction rate. Molar concentrations are computed as above are in [kmol/m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>]. For the purpose of reaction rate evaluation, the concentrations are transformed to [mol/cm<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>]. The resulting reaction rates and species production rates are in [mol/(cm<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>.s)]. In the last step these are converted to SI units [kg/(m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>.s)].

<a name="example"></a>

#### 5.1.5\. Example

The production rate for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in mass units (kg/m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>/s) (<img src="svgs/accccf62db7f34355b5c9c0d3f2991df.svg?invert_in_darkmode" align=middle width=41.111216849999984pt height=22.465723500000017pt/>) is computed with the [function call](#cxx-api-ReactionRates) and in mole units (<img src="svgs/a7588e8950c755ddcb25631cefaa7343.svg?invert_in_darkmode" align=middle width=17.49816089999999pt height=21.95701200000001pt/> kmol/m<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>/s) with [function call](#cxx-api-ReactionRatesMole). A example is located at src/example/TChem_NetProductionRatesPerMass.cpp. This example computes the production rate in mass units for any type of gas reaction mechanism.

<a name="surfacechemistry"></a>

### 5.2\. [Surface Chemistry](#cxx-api-ReactionRatesSurface)

The production rate for gas and surface species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in molar/<img src="svgs/89ef0b1086da48459dd5f47ed088933b.svg?invert_in_darkmode" align=middle width=20.985647099999987pt height=26.76175259999998pt/> units is written as

<p align="center"><img src="svgs/21eb32fc163f794dc876ec8d8a7e721c.svg?invert_in_darkmode" align=middle width=233.102595pt height=47.988758399999995pt/></p>

where <img src="svgs/83f96341b70056c1342f4e593867bf19.svg?invert_in_darkmode" align=middle width=38.90714519999999pt height=22.465723500000017pt/> is the number of reactions on the surface phase and <img src="svgs/8d69b56f74dd3aa880ae9044c2b9ef61.svg?invert_in_darkmode" align=middle width=20.03720234999999pt height=24.7161288pt/> and <img src="svgs/0e34d813eb8572232bb709f72638f649.svg?invert_in_darkmode" align=middle width=20.03720234999999pt height=24.7161288pt/> are the stoichiometric coefficients of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> for the reactant and product side of the reaction, respectively.

The rate of progress <img src="svgs/9294da67e8fbc8ee3f1ac635fc79c893.svg?invert_in_darkmode" align=middle width=11.989211849999991pt height=14.15524440000002pt/> of the <img src="svgs/930303c1b2e611a8c8b5b1708041319e.svg?invert_in_darkmode" align=middle width=21.07043894999999pt height=22.831056599999986pt/> surface reaction is equal to:

<p align="center"><img src="svgs/30dd6dc2648dcb1602be08b1f4017fb5.svg?invert_in_darkmode" align=middle width=234.96965745pt height=51.82436325pt/></p>


Where <img src="svgs/0983467432bb3b4f9708f334be19484b.svg?invert_in_darkmode" align=middle width=17.92738529999999pt height=22.731165599999983pt/> is the concentration of the species <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/>. If the species <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/> is a gas species, this is the molar concentration (<img src="svgs/4ae109cb9802e31a2894694f304648a3.svg?invert_in_darkmode" align=middle width=63.32813849999999pt height=32.40174300000001pt/>). If, on the other hand, the species <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/> is a surface species, it the surface molar concentration computed by <img src="svgs/fa83cb74bd784773c2f17097cab73bd1.svg?invert_in_darkmode" align=middle width=68.04511559999999pt height=29.40633629999998pt/> is . <img src="svgs/b29d3be23e3dcd41524e9b1c2aa25c08.svg?invert_in_darkmode" align=middle width=17.32598999999999pt height=22.465723500000017pt/> is site fraction, <img src="svgs/9458db33d70635758e0914dac4a04f2f.svg?invert_in_darkmode" align=middle width=18.400027799999993pt height=22.465723500000017pt/> is density of surface site of the phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, and <img src="svgs/c9f8acd74c96c449b38d4c93218986d9.svg?invert_in_darkmode" align=middle width=26.81998439999999pt height=14.15524440000002pt/> is the site occupancy number (We assume <img src="svgs/8297fb0ef40b4094af5ce4d4854c8fc1.svg?invert_in_darkmode" align=middle width=57.77871494999999pt height=21.18721440000001pt/> ).

<a name="forwardandreverserateconstants-1"></a>

#### 5.2.1\. Forward and Reverse Rate Constants

The forward rate constant is computed as we describe in the gas section. If parameters are not specified for reverse rate, this rate is computed with equilibrium constant defined by:

<p align="center"><img src="svgs/4b53f217352f474068cffe519d6d13d6.svg?invert_in_darkmode" align=middle width=76.78739475pt height=38.5152603pt/></p>

The equilibrium constant for the surface reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> is computed as

<p align="center"><img src="svgs/6f0756d6526003890f031bc29a28ad89.svg?invert_in_darkmode" align=middle width=320.27857785pt height=57.904143pt/></p>  

Here, <img src="svgs/fda26aae5e9a669fc3ca900427c61989.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> and <img src="svgs/a3be6c1dfffdf48fdac5347cf58d0e51.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> represent the number of gas-phase and surface species, respectively, and <img src="svgs/4a21d41467e2f8f0863977829fd3b7de.svg?invert_in_darkmode" align=middle width=45.71790464999999pt height=21.839370299999988pt/>atm. TChem currently assumes the surface site density <img src="svgs/9458db33d70635758e0914dac4a04f2f.svg?invert_in_darkmode" align=middle width=18.400027799999993pt height=22.465723500000017pt/> for all phases to be constant. The equilibrium constant in pressure units is computed as

<p align="center"><img src="svgs/7c001b9ede0e5160815cb71570f8d4e6.svg?invert_in_darkmode" align=middle width=200.1143463pt height=39.452455349999994pt/></p>

based on entropy and enthalpy changes from reactants to products (including gas-phase and surface species). The net change for surface of the site occupancy number for phase <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> for reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> is given by

<p align="center"><img src="svgs/ec942be9945077e8e5c4a3ce691f10bc.svg?invert_in_darkmode" align=middle width=177.65334675pt height=52.2439797pt/></p>

<a name="stickingcoefficients"></a>

### 5.3\. Sticking Coefficients

The reaction rate for some surface reactions are described in terms of the probability that a collision results in a reaction. For these reaction, the forward rate is computed as

<p align="center"><img src="svgs/ac3b20b327e05b381441b0d4b83060c1.svg?invert_in_darkmode" align=middle width=302.47149075pt height=42.42388095pt/></p>

where <img src="svgs/9925e4d3a0486c8d876c6c8eec9e256d.svg?invert_in_darkmode" align=middle width=13.16154179999999pt height=14.15524440000002pt/> is the sticking coefficient, <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> is the molecular weight of the gas-phase mixture, <img src="svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> is the universal gas constant, <img src="svgs/c8eddaa9b1fbda344404636b42d210ac.svg?invert_in_darkmode" align=middle width=26.69419004999999pt height=22.465723500000017pt/> is the total surface site concentration over all phases, and <img src="svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> is the sum of stoichiometric coefficients for all surface species in reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>.

<a name="noteonunitsforsurfaceproductionrates"></a>

#### 5.3.1\. Note on Units for surface production rates

The units of the surface and gas species concentration presented above are in units of kmol/m<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/> (surface species) or kmol/<img src="svgs/d7690fcd6ed35f2b6db0419dcd2bed0c.svg?invert_in_darkmode" align=middle width=20.985647099999987pt height=26.76175259999998pt/> (gas species). To match the units of the kinetic model and compute the rate constants, we transformed the concentration units to mol/cm<img src="svgs/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/> or mol/cm<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>. The resulting rate constant has units of mol/cm<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>. In the last step these are converted to SI units [kg/(m<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>.s)].

<a name="example-1"></a>

#### 5.3.2\. Example

The production rate for species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> in mass units (kg/m<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>/s) (<img src="svgs/0ce45e53c156d30a0b3415aabf7119b6.svg?invert_in_darkmode" align=middle width=38.58458669999999pt height=22.465723500000017pt/>) is computed with the [function call](#cxx-api-ReactionRatesSurface), in molar units ( <img src="svgs/f9298376b5e06f5cb827bee6dbb2ddc6.svg?invert_in_darkmode" align=middle width=14.97150929999999pt height=21.95701200000001pt/> kmole/m<img src="svgs/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>/s) with [function call](#cxx-api-ReactionRatesSurfaceMole). A example is located at src/example/TChem_NetProductionSurfacePerMass.cpp. In this example, we compute the production rates of gas phase and also the production rate of the surface phase in mass units.

<a name="reactors"></a>

## 6\. Reactors


We present the setup for canonical examples that are available through TChem. All models presented in this section are setup to be run in parallel, possibly exploiting several layers of parallelism available on the platform of choice. We start with a description of a 2-nd order backward differentiation formula (BDF2) time stepping algorithm in [Section](#timeintegration). BDF2 was implemented via Kokkos and takes advantage of parallel threads available through the Kokkos interface. We then present results for homogenous batch reactors in [Section](#0dignition), and the plug-flow reactor, in [Section](#plugflowreactorpfrproblemwithgasandsurfacesreactions).

<a name="timeintegration"></a>

### 6.1\. Time Integration

For solving a stiff time ODEs, a time step size is limited by a stability condition rather than a truncation error. To obtain a reliable solution, we use a stable time integration method i.e., 2nd order Trapezoidal Backward Difference Formula (TrBDF2). The TrBDF2 scheme is a composite single step method. The method is 2nd order accurate and <img src="svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18724254999999pt height=22.465723500000017pt/>-stable.

* R. E. Bank, W. M. Coughran, W. Fichtner, E. H. Grosse, D. J. Rose & R. K. Smith Transient simulation of silicon devices and circuits. IEEE Trans. Comput. Aided Des. CAD-4, 436-451, 1985.

<a name="trbdf2"></a>

#### 6.1.1\. TrBDF2

For example, we consider a following system of time Ordinary Differential Equations (ODEs).
<p align="center"><img src="svgs/9cb6870639f4f8d4b71c445b605f53a5.svg?invert_in_darkmode" align=middle width=96.28770359999999pt height=33.81208709999999pt/></p>

As its name states, the method advances the solution from <img src="svgs/ec9b770ea2cbdbac68a649eb61dc4a33.svg?invert_in_darkmode" align=middle width=14.06212004999999pt height=20.221802699999984pt/> to an intermediate time <img src="svgs/3abdd595ccbe5b3eef81089fede47357.svg?invert_in_darkmode" align=middle width=118.53961514999999pt height=22.465723500000017pt/> by applying the Trapezoidal rule.
<p align="center"><img src="svgs/a5fd4aa9e45e5866babfe2736b61748e.svg?invert_in_darkmode" align=middle width=233.40216734999998pt height=33.62942055pt/></p>

Next, it uses BDF2 to march the solution from <img src="svgs/5b625c0326733d4bfe04746b6e14c83c.svg?invert_in_darkmode" align=middle width=31.766240549999992pt height=20.221802699999984pt/> to <img src="svgs/360b4a8d967ba9ef8f3bfcd5fb238d4c.svg?invert_in_darkmode" align=middle width=108.05557289999999pt height=22.465723500000017pt/> as follows.
<p align="center"><img src="svgs/f4d2281e9236da3cc558a8d0cd5580ee.svg?invert_in_darkmode" align=middle width=373.4322504pt height=39.887022449999996pt/></p>

We solve the above non-linear equations iteratively using the Newton method. The Newton equation of the first Trapezoidal step is described:
<p align="center"><img src="svgs/d4c05753fd1972a3aaef1d8590200dbd.svg?invert_in_darkmode" align=middle width=446.23266104999993pt height=49.315569599999996pt/></p>  

Then, the Newton equation of the BDF2 is described as follows.
<p align="center"><img src="svgs/5ea417b3c0dd47c67d057f174c943f25.svg?invert_in_darkmode" align=middle width=649.4946743999999pt height=49.315569599999996pt/></p>

Here, we denote a Jacobian as <img src="svgs/41540fafb1b5307e6a02b1596472aac6.svg?invert_in_darkmode" align=middle width=74.83060199999998pt height=30.648287999999997pt/>. The modified Jacobian's used for solving the Newton equations of the above Trapezoidal rule and the BDF2 are given as follows
<p align="center"><img src="svgs/f29d4c818562bc4d8f98705142861f02.svg?invert_in_darkmode" align=middle width=354.22853234999997pt height=36.82577085pt/></p>

while their right hand sides are defined as
<p align="center"><img src="svgs/14e37c6fd5e3a7d1638340bffa0ae040.svg?invert_in_darkmode" align=middle width=709.0713316499999pt height=40.11819404999999pt/></p>

In this way, a Newton solver can iteratively solves a problem <img src="svgs/e817263b5d8000d92c3d75f9b138d689.svg?invert_in_darkmode" align=middle width=103.03098299999998pt height=24.65753399999998pt/> with updating <img src="svgs/dfd306741f702522fe4be9ab852cf496.svg?invert_in_darkmode" align=middle width=61.45168094999998pt height=22.831056599999986pt/>.

The timestep size <img src="svgs/5a63739e01952f6a63389340c037ae29.svg?invert_in_darkmode" align=middle width=19.634768999999988pt height=22.465723500000017pt/> can be adapted within a range <img src="svgs/1718bcb245ae52092c84f5c0b9f52c0d.svg?invert_in_darkmode" align=middle width=111.69601079999997pt height=24.65753399999998pt/> using a local error estimator.
<p align="center"><img src="svgs/e8a12459baaa129b1feb968f0b2f54db.svg?invert_in_darkmode" align=middle width=597.2284923pt height=40.11819404999999pt/></p>

This error is minimized when using a <img src="svgs/c8c3d28144b9e3c06c085c71e94ebb14.svg?invert_in_darkmode" align=middle width=81.56977454999999pt height=28.511366399999982pt/>.


<a name="timestepadaptivity"></a>

#### 6.1.2\. Timestep Adaptivity

TChem uses weighted root-mean-square (WRMS) norms evaluating the estimated error. This approach is used in [Sundial package](https://computing.llnl.gov/sites/default/files/public/ida_guide.pdf). A weighting factor is computed as
<p align="center"><img src="svgs/3d93f99936f691f29b73c5c8cc342258.svg?invert_in_darkmode" align=middle width=179.18859584999998pt height=16.438356pt/></p>

and the normalized error norm is computed as follows.
<p align="center"><img src="svgs/298b6d7d16c82457035c96b10fc45e09.svg?invert_in_darkmode" align=middle width=215.89134270000002pt height=49.315569599999996pt/></p>

This error norm close to 1 is considered as *small* and we increase the time step size and if the error norm is bigger than 10, the time step size decreases by half.

<a name="interfacetotimeintegrator"></a>

#### 6.1.3\. Interface to Time Integrator

Our time integrator advance times for each sample independently in a parallel for. A namespace ``Impl`` is used to define a code interface for an individual sample.
```
TChem::Impl::TimeIntegrator::team_invoke_detail(
  /// kokkos team thread communicator
  const MemberType& member,
  /// abstract problem generator computing J_{prob} and f
  const ProblemType& problem,
  /// control parameters
  const ordinal_type& max_num_newton_iterations,
  const ordinal_type& max_num_time_iterations,
  /// absolute and relative tolerence size 2 array
  const RealType1DViewType& tol_newton,
  /// a vector of absolute and relative tolerence size Nspec x 2
  const RealType2DViewType& tol_time,
  /// \Delta t input, min, max
  const real_type& dt_in,
  const real_type& dt_min,
  const real_type& dt_max,
  /// time begin and end
  const real_type& t_beg,
  const real_type& t_end,
  /// input state vector at time begin
  const RealType1DViewType& vals,
  /// output for a restarting purpose: time, delta t, state vector
  const RealType0DViewType& t_out,
  const RealType0DViewType& dt_out,
  const RealType1DViewType& vals_out,
  const WorkViewType& work) {
  /// A pseudo code is illustrated here to describe the workflow

  /// This object is used to estimate the local errors
  TrBDF2<problem_type> trbdf2(problem);
  /// A_{tr} and b_{tr} are computed using the problem provided J_{prob} and f
  TrBDF2_Part1<problem_type> trbdf2_part1(problem);
  /// A_{bdf} and b_{bdf} are computed using the problem provided J_{prob} and f
  TrBDF2_Part2<problem_type> trbdf2_part2(problem);

  for (ordinal_type iter=0;iter<max_num_time_iterations && dt != zero;++iter) {
    /// evaluate function f_n
    problem.computeFunction(member, u_n, f_n);

    /// trbdf_part1 provides A_{tr} and b_{tr} solving A_{tr} du = b_{tr}
    /// and update u_gamma += du iteratively until it converges
    TChem::Impl::NewtonSolver(member, trbdf_part1, u_gamma, du);

    /// evaluate function f_gamma
    problem.computeFunction(member, u_gamma, f_gamma);

    /// trbdf_part2 provides A_{bdf} and b_{bdf} solving A_{bdf} du = b_{bdf}
    /// and update u_np += du iteratively until it converges
    TChem::Impl::NewtonSolver(member, trbdf_part2, u_np, du);

    /// evaluate function f_np
    problem.computeFunction(member, u_np, f_np);

    /// adjust time step
    trbdf2.computeTimeStepSize(member,
      dt_min, dt_max, tol_time, f_n, f_gamma, f_np, /// input for error evaluation
      dt); /// output

    /// account for the time end
    dt = ((t + dt) > t_end) ? t_end - t : dt;      
  }

  /// store current time step and state vectors for a restarting purpose
```  
This ``TimeIntegrator`` code requires for a user to provide a problem object. A problem class should include the following interface in order to be used with the time integrator.
```
template<typename KineticModelConstDataType>
struct MyProblem {
  ordinal_type getNumberOfTimeODEs();
  ordinal_type getNumberOfConstraints();
  /// the number of equations should be sum of number of time ODEs and number of constraints
  ordinal_type getNumberOfEquations();

  /// temporal workspace necessary for this problem class
  ordinal_type getWorkSpaceSize();

  /// x is initialized in the first Newton iteration
  void computeInitValues(const MemberType& member,
                         const RealType1DViewType& x) const;

  /// compute f(x)
  void computeFunction(const MemberType& member,
                       const RealType1DViewType& x,
                       const RealType1DViewType& f) const;

  /// compute J_{prob} at x                       
  void computeJacobian(const MemberType& member,
                       const RealType1DViewType& x,
                       const RealType2DViewType& J) const;
};
```
<a name="homogenousbatchreactors"></a>

### 6.2\. [Homogenous Batch Reactors](#cxx-api-IgnitionZeroD)

<a name="problemdefinition"></a>

#### 6.2.1\. Problem Definition

In this example we consider a transient zero-dimensional constant-pressure problem where temperature <img src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999991pt height=22.465723500000017pt/> and species mass fractions for <img src="svgs/259ce06b5bc89a23bcdcc86d59ca9a1b.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> gas-phase species are resolved in a batch reactor. In this problem an initial condition is set and a time integration solver will evolve the solution until a time provided by the user.

For an open batch reactor the system of ODEs solved by TChem are given by:
* ***Energy equation***
<p align="center"><img src="svgs/777b4a1b078a12a58113039d3a9e1c4c.svg?invert_in_darkmode" align=middle width=224.50411004999998pt height=49.96363845pt/></p>

* ***Species equation***
<p align="center"><img src="svgs/e48f71b9924bae1f2f4e0a193143aeed.svg?invert_in_darkmode" align=middle width=270.7953138pt height=37.0084374pt/></p>

where <img src="svgs/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode" align=middle width=8.49888434999999pt height=14.15524440000002pt/> is the density, <img src="svgs/64e120bb63546232c0b0ecdffde52aa1.svg?invert_in_darkmode" align=middle width=13.89028244999999pt height=14.15524440000002pt/> is the specific heat at constant pressure for the mixture, <img src="svgs/a429d1ad642bbcafbae2517713a9eb99.svg?invert_in_darkmode" align=middle width=19.03453859999999pt height=21.95701200000001pt/> is the molar production rate of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, <img src="svgs/b09bc86177601e586d14fb48ca4cb31d.svg?invert_in_darkmode" align=middle width=22.79116289999999pt height=22.465723500000017pt/> is its molecular weight, and <img src="svgs/0a5174be254172c0eab67e56c6c45b6a.svg?invert_in_darkmode" align=middle width=16.73714459999999pt height=22.831056599999986pt/> is the specific enthalpy.

<a name="jacobianformulation"></a>

#### 6.2.2\. Jacobian Formulation

Efficient integration and accurate analysis of the stiff system of ODEs shown above requires the Jacobian matrix of the *rhs* vector. In this section we will derive the Jacobian matrix components.

Let
<p align="center"><img src="svgs/cc1ab86accaef22be8a402bbb810388c.svg?invert_in_darkmode" align=middle width=202.54409339999998pt height=23.5253469pt/></p>

by the denote the variables in the *lhs* of the 0D system and let
<p align="center"><img src="svgs/40fa0494dc8068f04e467ba620c83693.svg?invert_in_darkmode" align=middle width=236.66497139999998pt height=23.5253469pt/></p>

be the extended state vector. The 0D system can be written in compact form as
<p align="center"><img src="svgs/11225865d91da39256f9b08753744cbe.svg?invert_in_darkmode" align=middle width=202.48133564999998pt height=37.5303027pt/></p>

where <img src="svgs/d2c399cd1d7753ffef8c75d2532ee2aa.svg?invert_in_darkmode" align=middle width=191.32125374999998pt height=27.6567522pt/> and <img src="svgs/8d3f80b3bf492899ffe45de09ebde6c9.svg?invert_in_darkmode" align=middle width=244.69565834999995pt height=30.632847300000012pt/>. The thermodynamic pressure <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/> was introduced for completeness. For open batch reactors <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/> is constant and <img src="svgs/ffe7fd1c572c71eb3391b987f2edd064.svg?invert_in_darkmode" align=middle width=51.17738339999998pt height=22.465723500000017pt/>. The source term <img src="svgs/4d10d7c75897d3a464e366a699a7fd1e.svg?invert_in_darkmode" align=middle width=16.90017779999999pt height=22.465723500000017pt/> is computed considering the ideal gas equation of state
<p align="center"><img src="svgs/7212004f8d34d0c1fcfaf074afbe2caa.svg?invert_in_darkmode" align=middle width=121.91783174999999pt height=36.09514755pt/></p>

with P=const and using the expressions above for <img src="svgs/eac3eab07f20cec76812cb66d45f2b8e.svg?invert_in_darkmode" align=middle width=19.61364404999999pt height=22.465723500000017pt/> and <img src="svgs/72994c9c54acfb33b81b90cb225b4357.svg?invert_in_darkmode" align=middle width=24.171396149999985pt height=22.465723500000017pt/>,
<p align="center"><img src="svgs/cabc1f5a5eff8974005011cf33a02ea3.svg?invert_in_darkmode" align=middle width=276.26860965pt height=49.96363845pt/></p>

Let <img src="svgs/86aa7d769ae41b760b561beb8d611acb.svg?invert_in_darkmode" align=middle width=12.19759034999999pt height=30.267491100000004pt/> and <img src="svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/> be the Jacobian matrices corresponding to <img src="svgs/fc10563fc615ab663892b1cb7c5b29f1.svg?invert_in_darkmode" align=middle width=34.47500429999999pt height=30.632847300000012pt/> and <img src="svgs/4301479a3a8440338f69c69cca4cf901.svg?invert_in_darkmode" align=middle width=34.475025749999986pt height=24.65753399999998pt/>, respectively. Chain-rule differentiation leads to
<p align="center"><img src="svgs/8117ebe3941e23c693a94a132f45d1b6.svg?invert_in_darkmode" align=middle width=150.89004645pt height=40.90933275pt/></p>

Note that each component <img src="svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.41027339999999pt height=14.15524440000002pt/> of <img src="svgs/5e16cba094787c1a10e568c61c63a5fe.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.465723500000017pt/> is also a component of <img src="svgs/ed5e46d980be634296262ccdd38cd5a6.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=30.267491100000004pt/> and the corresponding *rhs* components are also the same, <img src="svgs/e6846fc684829faa89cafea9c7e2438c.svg?invert_in_darkmode" align=middle width=104.51694329999998pt height=30.632847300000012pt/>.

<a name="evaluationoftildejcomponents"></a>

##### 6.2.2.1\. Evaluation of <img src="svgs/86aa7d769ae41b760b561beb8d611acb.svg?invert_in_darkmode" align=middle width=12.19759034999999pt height=30.267491100000004pt/> Components

We first identify the dependencies on the elements of <img src="svgs/ed5e46d980be634296262ccdd38cd5a6.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=30.267491100000004pt/> for each of the components of <img src="svgs/ae05f90efdcc2db28b4c589608615795.svg?invert_in_darkmode" align=middle width=11.75813594999999pt height=30.632847300000012pt/>

* <img src="svgs/9169acb7431069ea81ce06d46ec3d023.svg?invert_in_darkmode" align=middle width=54.240255299999994pt height=30.632847300000012pt/>. We postpone the discussion for this component.

* <img src="svgs/7121a782477cc66490570bffc8930993.svg?invert_in_darkmode" align=middle width=88.51746089999999pt height=30.632847300000012pt/>

* <img src="svgs/a94db1a86e5e75f774b1aa11e35e52ca.svg?invert_in_darkmode" align=middle width=56.95372154999998pt height=30.632847300000012pt/>. <img src="svgs/eac3eab07f20cec76812cb66d45f2b8e.svg?invert_in_darkmode" align=middle width=19.61364404999999pt height=22.465723500000017pt/> is defined above. Here we highlight its dependencies on the elements of <img src="svgs/ed5e46d980be634296262ccdd38cd5a6.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=30.267491100000004pt/>
<p align="center"><img src="svgs/28e17a709303d4e6974167c26cdae468.svg?invert_in_darkmode" align=middle width=519.41233905pt height=49.96363845pt/></p>

where <img src="svgs/607980980b21998a867d017baf966d07.svg?invert_in_darkmode" align=middle width=19.08890444999999pt height=22.731165599999983pt/> is the molar concentration of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, <img src="svgs/7e1cc19956f3e6def2b7ce351cdab04d.svg?invert_in_darkmode" align=middle width=98.96905094999998pt height=24.65753399999998pt/>.
<p align="center"><img src="svgs/72229410dfe82ddef76a9f8c34140a7f.svg?invert_in_darkmode" align=middle width=1306.35591735pt height=44.0142648pt/></p>

* <img src="svgs/ee9ad1358581d4762027c40c7a6a2d97.svg?invert_in_darkmode" align=middle width=78.86886809999999pt height=30.632847300000012pt/>
<p align="center"><img src="svgs/471cd13ef396b57e84e61c2936c8c71b.svg?invert_in_darkmode" align=middle width=1010.4119619pt height=42.41615565pt/></p>

The values for heat capacities and their derivatives are computed based on the NASA polynomial fits as
<p align="center"><img src="svgs/75812095c310a215c0dde79d1ddebb39.svg?invert_in_darkmode" align=middle width=592.19455845pt height=37.0930362pt/></p>

The partial derivatives of the species production rates,   <img src="svgs/e2a241fbe2686639f625d53f453b0374.svg?invert_in_darkmode" align=middle width=121.57186304999998pt height=24.65753399999998pt/>, are computed as
as
<p align="center"><img src="svgs/a9c606d3e79a60e70b245e5d0e6279a8.svg?invert_in_darkmode" align=middle width=900.20708415pt height=68.1617376pt/></p>

The steps for the calculation of <img src="svgs/1aa19c5ae8db79847d59e829274a32ec.svg?invert_in_darkmode" align=middle width=23.195122499999997pt height=29.662026899999994pt/> and <img src="svgs/29ffca0920ebb7b057ba2c6ebecb9cd9.svg?invert_in_darkmode" align=middle width=23.195122499999997pt height=29.662026899999994pt/> are itemized below

* Derivatives of production rate <img src="svgs/a7588e8950c755ddcb25631cefaa7343.svg?invert_in_darkmode" align=middle width=17.49816089999999pt height=21.95701200000001pt/> of species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>
<p align="center"><img src="svgs/3c4a8848cce038b538772604d46b27ed.svg?invert_in_darkmode" align=middle width=430.37877299999997pt height=47.988758399999995pt/></p>

* Derivatives of rate-of-progress variable <img src="svgs/9294da67e8fbc8ee3f1ac635fc79c893.svg?invert_in_darkmode" align=middle width=11.989211849999991pt height=14.15524440000002pt/> of reaction <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>
<p align="center"><img src="svgs/7c1b95ad4188c8427ac854485be6a485.svg?invert_in_darkmode" align=middle width=439.46513325pt height=36.2778141pt/></p>

* Derivatives of <img src="svgs/e680843a73d5214f63af89c487342755.svg?invert_in_darkmode" align=middle width=13.30617584999999pt height=22.465723500000017pt/>

    + Basic reactions $\mathcal{C}_i = 1$: $\frac{\partial\mathcal{C}_i}{\partial T}\equiv \frac{\partial\mathcal{C}_i}{\partial\mathfrak{X}_l}\equiv 0$

    + 3-rd body-enhanced reactions $\mathcal{C}_i = \mathfrak{X}_i$: $\frac{\partial\mathcal{C}_i}{\partial T}\equiv 0$, $\frac{\partial\mathcal{C}_i}{\partial\mathfrak{X}_l}=\alpha_{il}$

    + Unimolecular/recombination fall-off reactions $\mathcal{C}_i = \frac{\Pr_i}{1+\Pr_i}F_i$
<p align="center"><img src="svgs/4e08aa709474ca0fe57f49c90750037f.svg?invert_in_darkmode" align=middle width=570.99382395pt height=40.5178356pt/></p>

        - $\Pr_i=\frac{{k_0}_i}{{k_\infty}_i}\mathfrak{X}_i \Rightarrow \frac{\partial\Pr_i}{\partial T}=\frac{{k'_0}_i{k_\infty}_i-{k_0}_i{k'_\infty}_i}{{k_\infty^2}_i}\mathfrak{X}_i,\,\,\,\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}=\frac{{k_0}_i}{{k_\infty}_i}\alpha_{il}$.

        - $\Pr_i=\frac{{k_0}_i}{{k_\infty}_i}\mathfrak{X}_m \Rightarrow\frac{\partial\Pr_i}{\partial T}=\frac{{k'_0}_i{k_\infty}_i-{k_0}_i{k'_\infty}_i}{{k_\infty^2}_i}\mathfrak{X}_m,\,\,\,\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}=\frac{{k_0}_i}{{k_\infty}_i}\delta_{lm}$, where $\delta_{lm}$ is Kroenecker delta symbol.

        - For Lindemann form $F_i=1 \Rightarrow \frac{\partial F_i}{\partial T}\equiv \frac{\partial F_i}{\partial\mathfrak{X}_l}\equiv 0$.

        - For Troe form
<p align="center"><img src="svgs/72a413e28b485cbab31530a6be3778c2.svg?invert_in_darkmode" align=middle width=1318.0485350999998pt height=59.5682406pt/></p>

where
<p align="center"><img src="svgs/df75775429479ccbc461988cb3671168.svg?invert_in_darkmode" align=middle width=1300.67202045pt height=39.452455349999994pt/></p>

        - For SRI form
<p align="center"><img src="svgs/099718c692469750f01c0407fec594d6.svg?invert_in_darkmode" align=middle width=1215.9538924499998pt height=49.315569599999996pt/></p>

    + Chemically activated bimolecular reactions: $\mathcal{C}_i = \frac{1}{1+\Pr_i}F_i$
<p align="center"><img src="svgs/a2c03c37b7494241aeb2f9087979ba4f.svg?invert_in_darkmode" align=middle width=596.5646906999999pt height=40.5178356pt/></p>

Partial derivatives of <img src="svgs/9ad1568094e2a9f17f1cc28782ec690f.svg?invert_in_darkmode" align=middle width=22.27650644999999pt height=22.465723500000017pt/> and <img src="svgs/e17c35f619f835117e1ff8e25d5f8a9c.svg?invert_in_darkmode" align=middle width=15.22169714999999pt height=22.465723500000017pt/> are computed similar to the ones above.

* Derivatives of <img src="svgs/112d12b231c3277112d94b0a3d028253.svg?invert_in_darkmode" align=middle width=18.58246664999999pt height=22.465723500000017pt/>
<p align="center"><img src="svgs/5b06b752aa2c0c206adb43d311fa55cb.svg?invert_in_darkmode" align=middle width=585.85752225pt height=54.93765749999999pt/></p>

    + ${k_f}_i=A_iT^{\beta_i}\exp\left(-\frac{E_i}{R T}\right)=A_i\exp\left(\beta_i\ln T-\frac{{T_a}_i}{T}\right)$, where ${T_a}_i=E_i/R$. The derivative with respect to temperature can be calculated as ${k'_f}_i=\frac{{k_f}_i}{T}\left(\beta_i+\frac{{T_a}_i}{T}\right)$

    + if reverse Arrhenius parameters are provided, ${k'_r}_i$ is computed similar to above. If ${k_r}_i$ is computed based on ${k_f}_i$ and the equilibrium constant ${K_c}_i$, then its derivative is computed as
<p align="center"><img src="svgs/39a73887a8981ebcda407adf53724baf.svg?invert_in_darkmode" align=middle width=700.27402665pt height=52.84014285pt/></p>

Since <img src="svgs/0f45e01abedb807fa5934327239deaaf.svg?invert_in_darkmode" align=middle width=469.87180019999994pt height=45.4613247pt/>. It follows that
<p align="center"><img src="svgs/dd3325dbe653a147d23ef45ff7e46485.svg?invert_in_darkmode" align=middle width=294.00067455pt height=59.1786591pt/></p>

where <img src="svgs/b775ee2871d37fdb69801bac39217722.svg?invert_in_darkmode" align=middle width=15.10661129999999pt height=24.7161288pt/> is computed based on NASA polynomial fits as
<p align="center"><img src="svgs/2110abbec80e8b015d84a6f3ff3b39a4.svg?invert_in_darkmode" align=middle width=468.36640289999997pt height=32.990165999999995pt/></p>

<a name="efficientevaluationofthetildejterms"></a>

######  6.2.2.1.1\. Efficient Evaluation of the <img src="svgs/86aa7d769ae41b760b561beb8d611acb.svg?invert_in_darkmode" align=middle width=12.19759034999999pt height=30.267491100000004pt/> Terms

* Step 1:
<p align="center"><img src="svgs/9cfe214e0728e7f4a951d22f9fb11863.svg?invert_in_darkmode" align=middle width=1552.1963242499999pt height=59.1786591pt/></p>

Here <img src="svgs/7ae272b8bde40ee0cc7499478cef899c.svg?invert_in_darkmode" align=middle width=27.59316449999999pt height=22.465723500000017pt/> and <img src="svgs/33e9a242fccff66db526b03628effc6b.svg?invert_in_darkmode" align=middle width=27.315396899999985pt height=22.465723500000017pt/> are the forward and reverse parts, respectively of <img src="svgs/ca5b95374a81b49ab0145992a25d7566.svg?invert_in_darkmode" align=middle width=20.036075399999987pt height=22.465723500000017pt/>:
<p align="center"><img src="svgs/707bc38d1141fe05fe7c6358c601899d.svg?invert_in_darkmode" align=middle width=299.93412405pt height=49.58691705pt/></p>

* Step 2: Once <img src="svgs/2d43fb32af068635d9b782cd4dc9ad36.svg?invert_in_darkmode" align=middle width=58.22420009999998pt height=30.267491100000004pt/> are evaluated for all <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>, then <img src="svgs/186d59c845c8e08a0773e851a16cf2f1.svg?invert_in_darkmode" align=middle width=43.48192694999999pt height=30.267491100000004pt/> is computed as
<p align="center"><img src="svgs/50ecd896217abbb133d091a7fe7033c3.svg?invert_in_darkmode" align=middle width=669.6422799pt height=59.1786591pt/></p>

* Step 3:
<p align="center"><img src="svgs/4a098eebe1f9522fa69ab805055d980b.svg?invert_in_darkmode" align=middle width=1407.0424219499998pt height=59.1786591pt/></p>

<a name="evaluationofjcomponents"></a>

##### 6.2.2.2\. Evaluation of <img src="svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/> Components

* *Temperature equation*
<p align="center"><img src="svgs/3b6b95f66f502909a93004814baf20c3.svg?invert_in_darkmode" align=middle width=350.65493595pt height=36.2778141pt/></p>

* *Species equations*
<p align="center"><img src="svgs/09f46a3fa978d93bcad493ebe355f1ba.svg?invert_in_darkmode" align=middle width=544.66597845pt height=36.2778141pt/></p>

For <img src="svgs/daa9ac6fe30dfd6dd5c7d1aba464e74f.svg?invert_in_darkmode" align=middle width=73.34471099999999pt height=22.465723500000017pt/> density is a dependent variable, calculated based on
the ideal gas equation of state:
<p align="center"><img src="svgs/aa0100b240fecc99b861420df28900bd.svg?invert_in_darkmode" align=middle width=137.75833335pt height=44.725160699999996pt/></p>

The partial derivaties of density with respect to the independent variables are computed as
<p align="center"><img src="svgs/1ea9d6a6e84b91e302f419bf16e7c083.svg?invert_in_darkmode" align=middle width=270.9500475pt height=36.2778141pt/></p>

<a name="runningthe0dignitionutility"></a>

#### 6.2.3\. Running the 0D Ignition Utility
The executable to run this example is installed at "TCHEM_INSTALL_PATH/example/",  and the input parameters are (./TChem_IgnitionZeroDSA.x --help) :

````
options:
 --OnlyComputeIgnDelayTime     bool      If true, simulation will end when Temperature is equal to T_threshold
                                         (default: --OnlyComputeIgnDelayTime=false)
 --T_threshold                 double    Temp threshold in ignition delay time
                                         (default: --T_threshold=1.5e+03)
 --atol-newton                 double    Absolute tolerence used in newton solver
                                         (default: --atol-newton=1.0e-10)
 --chemfile                    string    Chem file name e.g., chem.inp
                                         (default: --chemfile=chem.inp)
 --dtmax                       double    Maximum time step size
                                         (default: --dtmax=1.00e-01)
 --dtmin                       double    Minimum time step size
                                         (default: --dtmin=1.00e-08)
 --echo-command-line           bool      Echo the command-line but continue as normal
 --help                        bool      Print this help message
 --inputsPath                  string    path to input files e.g., data/inputs
                                         (default: --inputsPath=data/ignition-zero-d/CO/)
 --max-newton-iterations       int       Maximum number of newton iterations
                                         (default: --max-newton-iterations=100)
 --max-time-iterations         int       Maximum number of time iterations
                                         (default: --max-time-iterations=1000)
 --output_frequency            int       save data at this iterations
                                         (default: --output_frequency=-1)
 --rtol-newton                 double    Relative tolerance used in newton solver
                                         (default: --rtol-newton=1.0e-06)
 --samplefile                  string    Input state file name e.g.,input.dat
                                         (default: --samplefile=sample.dat)
 --tbeg                        double    Time begin
                                         (default: --tbeg=0.0)
 --team-size                   int       User defined team size
                                         (default: --team-size=-1)
 --tend                        double    Time end
                                         (default: --tend=1.0)
 --thermfile                   string    Therm file namee.g., therm.dat
                                         (default: --thermfile=therm.dat)
 --time-iterations-per-interval int       Number of time iterations per interval to store qoi
                                         (default: --time-iterations-per-interval=10)
 --tol-time                    double    Tolerance used for adaptive time stepping
                                         (default: --tol-time=1.0e-04)
 --use_prefixPath              bool      If true, input file are at the prefix path
                                         (default: --use_prefixPath=true)
 --vector-size                 int       User defined vector size
                                         (default: --vector-size=-1)
 --verbose                     bool      If true, printout the first Jacobian values
                                         (default: --verbose=true)
Description:
 This example computes the solution of an ignition problem
````

* ***GRIMech 3.0 model***

We can create a bash scripts to provide inputs to TChem. For example the following script runs an ignition problem with the GRIMech 3.0 model:

````
exec=../../TChem_IgnitionZeroDSA.x
inputs=../../data/ignition-zero-d/gri3.0/
save=1
dtmin=1e-8
dtmax=1e-3
tend=2
max_time_iterations=260
max_newton_iterations=20
atol_newton=1e-12
rtol_newton=1e-6
tol_time=1e-6

$exec --inputsPath=$inputs --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
````

In the above bash script the "inputs" variables is the path to where the inputs files are located in this case (\verb|TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0|). In this directory, the gas reaction mechanism is defined in "chem.inp" and the thermal properties in "therm.dat". Additionally, "sample.dat" contains the initial conditions for the simulation.

The parameters "dtmin" and "dtmax" control the size of the time steps in the solver. The decision on increase or decrease time step depends on the parameter "tol\_time". This parameter controls the error in each time iteration, thus, a bigger value will allow the solver to increase the time step while a smaller value will result in smaller time steps. The time-stepping will end when the time reaches "tend". The simulation will also end when the number of time steps reache  "max\_time\_iterations".  The absolute and relative tolerances in the Newton solver in each iteration are set with "atol\_newton" and "rtol\_newton", respectively, and the maximum number of Newton solver iterations is set with "max\_newton\_iterations".

The user can specify how often a solution is saved with the parameter "save". Thus, a solution will be saved at every iteration for this case. The default value of this input is <img src="svgs/e11a8cfcf953c683196d7a48677b2277.svg?invert_in_darkmode" align=middle width=21.00464354999999pt height=21.18721440000001pt/>, which means no output will be saved. The simulation results are saved in "IgnSolution.dat", with the following format:


````
iter     t       dt      Density[kg/m3]          Pressure[Pascal]        Temperature[K] SPECIES1 ... SPECIESN  
````  
where MF\_SPECIES1 respresents the mass fraction of species \#1, and so forth. Finally, we provide two methods to compute the ignition delay time. In the first approach, we save the time where the gas temperature reaches a threshold temperature. This temperature is set by default to <img src="svgs/06767d6f7bb1cd39b3cdcdf00489ce8e.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/>K. In the second approach, save the location of the inflection point for the temperature profile as a function of time, also equivalent to the time when the second derivative of temperature with respect to time is zero. The result of these two methods are saved in files "IgnitionDelayTimeTthreshold.dat" and "IgnitionDelayTime.dat", respectively.



* **GRIMech 3.0 Results**
 The results presented below are obtained by running "TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDSA.x" with an initial temperature of <img src="svgs/675eeb554f7b336873729327dab98036.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/>K, pressure of <img src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/>atm and a stoichiometric equivalence ratio (<img src="svgs/f50853d41be7d55874e952eb0d80c53e.svg?invert_in_darkmode" align=middle width=9.794543549999991pt height=22.831056599999986pt/>) for methane/air mixtures. The input files are located at "TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/" and selected parameters were presented above. The outputs of the simulation were saved every iteration in "IgnSolution.dat". Time profiles for temperature and mass fractions for selected species are presented the following figures.



![Temperature and <img src="svgs/ae893ddc6a1df993f14669218c18b5e1.svg?invert_in_darkmode" align=middle width=33.14158649999999pt height=22.465723500000017pt/>, <img src="svgs/07e86e45b7ebe272704b69abf2775d41.svg?invert_in_darkmode" align=middle width=19.09133654999999pt height=22.465723500000017pt/>, <img src="svgs/f74275c195a72099e0b06f584fd5489a.svg?invert_in_darkmode" align=middle width=25.920062849999987pt height=22.465723500000017pt/> mass fraction](src/markdown/Figures/gri3.0_OneSample/TempMassFraction2.jpg)

![Temperature and <img src="svgs/2f38b966661ff4345cf5e316b9f242fc.svg?invert_in_darkmode" align=middle width=27.99541469999999pt height=22.465723500000017pt/>, <img src="svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/>, <img src="svgs/6e9a8da465a1a260890dcd5a7ba118bc.svg?invert_in_darkmode" align=middle width=23.219177849999987pt height=22.465723500000017pt/> mass fraction](src/markdown/Figures/gri3.0_OneSample/TempMassFraction3.jpg)

The ignition delay time values based on the two alternative computations discussed above are <img src="svgs/9297d98d27e2aacf9bc52c4082105174.svg?invert_in_darkmode" align=middle width=62.10069029999999pt height=21.18721440000001pt/>s and <img src="svgs/af125457f5210d486f810999b1b58c70.svg?invert_in_darkmode" align=middle width=62.10069029999999pt height=21.18721440000001pt/>s, respectively. The scripts to setup and run this example and the jupyter-notebook used to create these figures can be found under "TCHEM_INSTALL_PATH/example/runs/gri3.0_IgnitionZeroD".

* **GRIMech 3.0 results parametric study**

The following figure shows the ignition delay time as a function of the initial temperature and equivalence ratio values. These results are based on settings provided in "TCHEM_INSTALL_PATH/example/runs/gri3.0_IgnDelay" and correspond to 100 samples. "TChem\_IgnitionZeroDSA.x" runs these samples in parallel. The wall-time is between <img src="svgs/2be9125b047404431cda44420795f135.svg?invert_in_darkmode" align=middle width=77.11192664999999pt height=21.18721440000001pt/> on a 3.1GHz Intel Core i7 cpu.

We also provide a jupyter-notebook to produce the sample file "sample.dat" and to generate the figure presented above.

![Ignition Delta time](src/markdown/Figures/gri3.0_IgnDelay/Gri3IgnDelayTime.jpg)
Figure. Ignition delay times [s] at P=1 atm for several CH<img src="svgs/5dfc3ab84de9c94bbfee2e75b72e1184.svg?invert_in_darkmode" align=middle width=6.5525476499999895pt height=14.15524440000002pt/>/air equivalence ratio <img src="svgs/f50853d41be7d55874e952eb0d80c53e.svg?invert_in_darkmode" align=middle width=9.794543549999991pt height=22.831056599999986pt/> and initial temperature values. Results are based on the GRI-Mech v3.0 kinetic model.



<a name="ignitiondelaytimeparameterstudyforisooctane"></a>

#### 6.2.4\. Ignition Delay Time Parameter Study for IsoOctane


We present a parameter study for several equivalence ratio, pressure, and initial temperature values for iso-Octane/air mixtures. The iso-Octane reaction mechanism used in this study consists of 874 species and 3796 elementary reactions~\cite{W. J. Pitz M. Mehl, H. J. Curran and C. K. Westbrook ref 5 }. We selected four pressure values, <img src="svgs/f4540d93de2e937eeecf814e082ccca0.svg?invert_in_darkmode" align=middle width=104.10974309999999pt height=24.65753399999998pt/> [atm]. For each case we ran a number of simulations that span a grid of <img src="svgs/08f4ed92f27cec32cdd7a6ecd580f9e7.svg?invert_in_darkmode" align=middle width=16.438418699999993pt height=21.18721440000001pt/> initial conditions each for the equivalence ratio and temperature resulting in 900 samples for each pressure value. Each sample was run on a test bed with a Dual-Socket Intel Xeon Platinum architecture.


The data produced by this example is located at "TCHEM_INSTALL_PATH/example/runs/isoOctane_IgnDelay". Because of the time to produce a result we save the data in a hdf5 format in "isoOctaneIgnDelayBlake.hdf5".
The following figures show ignition delay times results for the conditions specified above. These figures were generated with the jupyter notebook shared in the results directory.


![Ignition delay time (s) of isooctane at 10 atm](src/markdown/Figures/isoOctane/TempvsEquiRatio900IsoOctane10atm.jpg)
Figure.  Ignition delay times [s] at 10atm for several equivalence ratio (vertical axes) and temperature (hori- zontal axes) values for iso-Octane/air mixtures

![Ignition delay time (s) of isooctane at 16 atm](src/markdown/Figures/isoOctane/TempvsEquiRatio900IsoOctane16atm.jpg)
Figure.  Ignition delay times [s] at  16atm  for several equivalence ratio (vertical axes) and temperature (hori- zontal axes) values for iso-Octane/air mixtures

![Ignition delay time (s) of isooctane at 34 atm](src/markdown/Figures/isoOctane/TempvsEquiRatio900IsoOctane34atm.jpg)
Figure.  Ignition delay times [s] at  34atm  for several equivalence ratio (vertical axes) and temperature (hori- zontal axes) values for iso-Octane/air mixtures

![Ignition delay time (s) of isooctane at 45 atm](src/markdown/Figures/isoOctane/TempvsEquiRatio900IsoOctane45atm.jpg)
Figure.  Ignition delay times [s] at  45 atm  for several equivalence ratio (vertical axes) and temperature (hori- zontal axes) values for iso-Octane/air mixtures
<a name="plugflowreactorpfrproblemwithgasandsurfacesreactions"></a>

### 6.3\. [Plug Flow Reactor (PFR) Problem with Gas and Surfaces Reactions](#cxx-api-PlugFlowReactor)

<a name="problemdefinition-1"></a>

#### 6.3.1\. Problem Definition

The plug flow reactor (PFR) example employs both gas-phase and surface species. The PFR is assumed to be in steady state, therefore a system of differential-algebraic equations (DAE) must be resolved. The ODE part of the problem correspond to the solution of energy, momentum, total mass and species mass balance. The  algebraic constraint arises from the assumption that the PFR problem is a steady-state problem. Thus, the surface composition on the wall must be stationary.

The equations for the species mass fractions <img src="svgs/e0a50ae47e22e126ee04c138a9fc9abe.svg?invert_in_darkmode" align=middle width=16.80942944999999pt height=22.465723500000017pt/>, temperature <img src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999991pt height=22.465723500000017pt/>, axial velocity <img src="svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.41027339999999pt height=14.15524440000002pt/>, and continuity (represented  by density <img src="svgs/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode" align=middle width=8.49888434999999pt height=14.15524440000002pt/>) resolved by TChem were derived from Ref. \cite{ Chemically Reacting Flow: Theory, Modeling, and Simulation, Second Edition. Robert J. Kee, Michael E. Coltrin, Peter Glarborg, Huayang Zhu.}


* ***Species equation (eq 9.6 Robert J. Kee )***
<p align="center"><img src="svgs/f958c3f6ddcee774168d7f634bb7fd4e.svg?invert_in_darkmode" align=middle width=363.26803755pt height=48.95906565pt/></p>

* ***Energy equation (eq 9.76 Robert J. Kee )***
<p align="center"><img src="svgs/f5eb6315ee2e883d5ee40be1b8ce94e2.svg?invert_in_darkmode" align=middle width=345.32119709999995pt height=48.95906565pt/></p>

* ***Momentum equation***
<p align="center"><img src="svgs/33b9874dd5ee6953d5178354cf563cdf.svg?invert_in_darkmode" align=middle width=393.39680159999995pt height=48.95906565pt/></p>

Where <img src="svgs/17f34be523c24e1ff998fbbaa5d02877.svg?invert_in_darkmode" align=middle width=73.42301669999999pt height=42.44901869999998pt/> and <img src="svgs/03eff197f88c9fadf7abb66be563a57d.svg?invert_in_darkmode" align=middle width=87.64174649999998pt height=24.575218800000012pt/>.

* ***Continuity equation***

<p align="center"><img src="svgs/6a77a71a23b14c6278df699c22a6e2d4.svg?invert_in_darkmode" align=middle width=195.50711399999997pt height=48.95906565pt/></p>

where <img src="svgs/17f34be523c24e1ff998fbbaa5d02877.svg?invert_in_darkmode" align=middle width=73.42301669999999pt height=42.44901869999998pt/>, <img src="svgs/03eff197f88c9fadf7abb66be563a57d.svg?invert_in_darkmode" align=middle width=87.64174649999998pt height=24.575218800000012pt/>, <img src="svgs/e17a12c6a00d025727994e7274e45bda.svg?invert_in_darkmode" align=middle width=18.203450099999987pt height=22.465723500000017pt/> is the surface area, <img src="svgs/646ee7344b9ac741f33d392811137c62.svg?invert_in_darkmode" align=middle width=17.01109574999999pt height=22.465723500000017pt/>' is the surface chemistry parameter. In the equations above <img src="svgs/f9298376b5e06f5cb827bee6dbb2ddc6.svg?invert_in_darkmode" align=middle width=14.97150929999999pt height=21.95701200000001pt/> represents the surface chemistry production rate for a gas-phase species <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>.

* ***Algebraic constraint***
<p align="center"><img src="svgs/b9cdc3e4cd25484a7fada4f17a3c75f2.svg?invert_in_darkmode" align=middle width=291.67293705pt height=14.611878599999999pt/></p>

Here <img src="svgs/a3be6c1dfffdf48fdac5347cf58d0e51.svg?invert_in_darkmode" align=middle width=38.30018609999999pt height=22.465723500000017pt/> represent all surface species.

The number of ODEs is equal to the number of gas-phases species with three additional equations for thermodynamic temperature, continuity and momentum.  The number of constraints is equal to the number of surfaces species. This PFR formulation assumes that surface reactions are taking place on the channel wall and gas-phase reactions inside the channel. Wall friction and heat transfer at the wall are neglected in this example.

<a name="jacobianformulation-1"></a>

#### 6.3.2\. Jacobian Formulation

The current implementation uses a numerical jacobian based on forward finite differences.

<a name="runningtheplugflowreactorwithsurfacereactionsutility"></a>

#### 6.3.3\. Running the Plug Flow Reactor with Surface Reactions Utility


The executable for this example is installed under "TCHEM_INSTALL_PATH/example". The inputs for this example are obtained through.
```
Usage: ./TChem_PlugFlowReactor.x [options]
  options:
  --Area                        double    Cross-sectional Area
                                          (default: --Area=5.3e-04)
  --Pcat                        double    Chemically active perimeter,
                                          (default: --Pcat=2.6e-02)
  --atol-newton                 double    Absolute tolerance used in newton solver
                                          (default: --atol-newton=1.e-12)
  --batchsize                   int       Batchsize the same state vector described in state file is cloned
                                          (default: --batchsize=1)
  --chemSurffile                string    Chem file name e.g., chemSurf.inp
                                          (default: --chemSurffile=chemSurf.inp)
  --chemfile                    string    Chem file name e.g., chem.inp
                                          (default: --chemfile=chem.inp)
  --dzmax                       double    Maximum dz step size
                                          (default: --dzmax=1.0e-06)
  --dzmin                       double    Minimum dz step size
                                          (default: --dzmin=1.0e-10)
  --echo-command-line           bool      Echo the command-line but continue as normal
  --help                        bool      Print this help message
  --initial_condition           bool      If true, use a newton solver to obtain initial condition of the constraint
                                          (default: --initial_condition=True)
  --inputSurffile               string    Input state file name e.g., inputSurfGas.dat
                                          (default: --inputSurffile=inputSurf.dat)
  --inputVelocityfile           string    Input state file name e.g., inputVelocity.dat
                                          (default: --inputVelocityfile=inputVelocity.dat)
  --max-newton-iterations       int       Maximum number of newton iterations
                                          (default: --max-newton-iterations=100)
  --max-z-iterations         int       Maximum number of z iterations
                                          (default: --max-z-iterations=4000)
  --output_frequency            int       save data at this iterations
                                          (default: --output_frequency=-1)
  --prefixPath                  string    prefixPath e.g.,inputs/
                                          (default: --prefixPath=data/plug-flow-reactor/X/)
  --rtol-newton                 double    Relative tolerance used in newton solver
                                          (default: --rtol-newton=1.0e-06)
  --samplefile                  string    Input state file name e.g., input.dat
                                          (default: --samplefile=sample.dat)
  --zbeg                        double    Position begin
                                          (default: --zbeg=0)
  --team-size                   int       User defined team size
                                          (default: --team-size=-1)
  --zend                        double    Position  end
                                          (default: --zend=2.5e-02)
  --thermSurffile               string    Therm file name e.g.,thermSurf.dat
                                          (default: --thermSurffile=thermSurf.dat)
  --thermfile                   string    Therm file name e.g., therm.dat
                                          (default: --thermfile=therm.dat)
  --time-iterations-per-intervalint       Number of time iterations per interval to store qoi
                                          (default: --time-iterations-per-interval=10)
  --tol-z                    double    Tolerance used for adaptive z stepping
                                          (default: --tol-z=1.0e-04)
  --transient_initial_condition bool      If true, use a transient solver to obtain initial condition of the constraint
                                          (default: --transient_initial_condition=false)
  --use_prefixPath              bool      If true, input file are at the prefix path
                                          (default: --use_prefixPath=true)
  --vector-size                 int       User defined vector size
                                          (default: --vector-size=-1)
  --verbose                     bool      If true, printout the first Jacobian values
                                          (default: --verbose=true)
Description:
  This example computes Temperature, density, mass fraction and site fraction for a plug flow reactor
```

The following shell script sets the input parameters and runs the PFR example

```
exec=$TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x
inputs=$TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas/
Area=0.00053
Pcat=0.025977239243415308
dzmin=1e-12
dzmax=1e-5
zend=0.025
tol_z=1e-8
max_z_iterations=310
max_newton_iterations=20
atol_newton=1e-12
rtol_newton=1e-8
save=1
transient_initial_condition=false
initial_condition=true

$exec --prefixPath=$inputs --initial_condition=$initial_condition --transient_initial_condition=$transient_initial_condition --Area=$Area  --Pcat=$Pcat --tol-z=$tol_z --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dzmin=$dzmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dzmax=$dzmax --zend=$zend --max-time-iterations=$max_z_iterations
```

We ran the example  in the install directory "TCHEM_INSTALL_PATH/example/runs/PlugFlowReactor/CH4-PTnogas". Thus, all the paths are relative to this directory. This script will run the executable "TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x" with the input files located at "TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas/". These files correspond to the gas-phase and surface reaction mechanisms ("chem.inp" and "chemSurf.inp") and their corresponding thermo files ("therm.dat" and "thermSurf.dat").  The operating condition at the inlet of the reactor, i.e. the gas composition, in "sample.dat", and the initial guess for the site fractions, in "inputSurf.dat", are also required. The format and description of these files are presented in [Section](#inputfiles). The gas velocity at the inlet is provided in "inputVelocity.dat".




The  "Area" [<img src="svgs/89ef0b1086da48459dd5f47ed088933b.svg?invert_in_darkmode" align=middle width=20.985647099999987pt height=26.76175259999998pt/>] is the cross area of the channel and  "Pcat" [<img src="svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/>] is the chemical active perimeter of the PFR. The step size is controled by  "dzmin",  "dzmax", and "tol_z", the simulation will end with the "z"(position) is equal to  "zend" or when it reaches the  "max_z_iterations'.  The relative and absolute tolerance in the Newton solver are set through "atol_newton" and "rtol_newton". The description of the  integration method can be found in [Section](#timeintegration). The "save" parameter sets the output frequency, in this case equal to <img src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/>, which means the information will be saved every stepin "PFRSolution.dat". The following header is saved in the output file



````
iter     t       dt      Density[kg/m3]          Pressure[Pascal]        Temperature[K] SPECIES1 (Mass Fraction) ... SPECIESN (Mass Fraction)  SURFACE_SPECIES1 (Site Fraction) ... SURFACE_SPECIESN (Site Fraction) Velocity[m/s]  
````

The inputs "transient_initial_condition" and "initial_condition" allow us to pick a method to compute an initial condition that satisfies the system of DAE equation as described in [Section](#initialconditionforpfrproblem). In this case, the simulation will use a Newton solver to find an initial surface site fraction to meet the constraint presented above.


***Results***
The gas-phase and surface mechanisms used in this example represents the catalytic combustion of methane on platinum and was developed by [Blondal and co-workers](https://pubs.acs.org/doi/10.1021/acs.iecr.9b01464). These mechanisms have 15 gas species, 20 surface species, 47 surface reactions and no gas-phase reactions. The total number of ODEs is <img src="svgs/15d851cfce799553cec908376fe8edd9.svg?invert_in_darkmode" align=middle width=16.438418699999993pt height=21.18721440000001pt/> and there are <img src="svgs/ee070bffef288cab28aad0517a35741b.svg?invert_in_darkmode" align=middle width=16.438418699999993pt height=21.18721440000001pt/> constrains.  One simulation took about 12s to complete on a MacBook Pro with a 3.1GHz Intel Core i7 processor. Time profiles for temperature, density, velocity, mass fractions and site fractions for selected species are presented in the following figures.  Scripts and jupyter notebooks for this example are located under "TCHEM_INSTALL_PATH/example/runs/PlugFlowReactor/CH4-PTnogas".


![plot of density, temperature and velocity](src/markdown/Figures/CH4-PTnogas/TempDensVelPFR.jpg)
Figure. Gas Temperature (left axis), velocity and density (both on right axis) along the PFR.   

![mass fraction of a few species](src/markdown/Figures/CH4-PTnogas/gas1.jpg)
Figure. Mass fraction of <img src="svgs/7f936bb1c1dc952b6731aaf76b41df20.svg?invert_in_darkmode" align=middle width=30.753496949999988pt height=22.465723500000017pt/>, <img src="svgs/b1ecdbb9cb88bb3bd2f7e5606b55165b.svg?invert_in_darkmode" align=middle width=25.114232549999993pt height=22.465723500000017pt/> and <img src="svgs/b4b356cc8f0001fbb1edac08b3d41b74.svg?invert_in_darkmode" align=middle width=56.68964069999999pt height=22.465723500000017pt/>.   

![mass fraction of a few species](src/markdown/Figures/CH4-PTnogas/gas2.jpg)
Figure. Mass fractions for <img src="svgs/9ccbda7b283762831af5341797828c29.svg?invert_in_darkmode" align=middle width=19.33798019999999pt height=22.465723500000017pt/>, <img src="svgs/60bca9f14ca3cdd80da13186d6202ffe.svg?invert_in_darkmode" align=middle width=36.986411549999985pt height=22.465723500000017pt/> and <img src="svgs/d05b020cfe66331f185202f0a4934fcf.svg?invert_in_darkmode" align=middle width=57.14627654999999pt height=22.465723500000017pt/>

![site fraction of a few species](src/markdown/Figures/CH4-PTnogas/surf1.jpg)
Figure. Site fractions for <img src="svgs/f2809fd51e9b0b17e492b0a02b04d50b.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> (empty space), <img src="svgs/ca17b45f895957a581477173533d4f46.svg?invert_in_darkmode" align=middle width=43.90423619999999pt height=22.465723500000017pt/> and <img src="svgs/3d91d34d076e0b330a185fb1afcebd5a.svg?invert_in_darkmode" align=middle width=37.44303089999999pt height=22.465723500000017pt/>.

![site fraction of a few species](src/markdown/Figures/CH4-PTnogas/surf2.jpg)
Figure. Site fractions for <img src="svgs/27685d3014878a9b9183ce59dbfd814f.svg?invert_in_darkmode" align=middle width=24.65759669999999pt height=22.465723500000017pt/>, <img src="svgs/6fb5d226e2c01a382831002fe9cadc7b.svg?invert_in_darkmode" align=middle width=36.52977569999999pt height=22.465723500000017pt/>, <img src="svgs/91b7f323325c356aa9678bd264cd611f.svg?invert_in_darkmode" align=middle width=48.85857404999999pt height=22.465723500000017pt/>.

***Parametric study***
The executable "TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x" can be also used with more than one sample. In this example, we ran it with eight samples. The inputs for this run are located at "TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas_SA". A script and a jupyter-notebook to reproduce this example are placed under "TCHEM_INSTALL_PATH/example/runs/PlugFlowReactor/CH4-PTnogas_SA".

These samples correspond to combination of values for the molar fraction of  <img src="svgs/ae893ddc6a1df993f14669218c18b5e1.svg?invert_in_darkmode" align=middle width=33.14158649999999pt height=22.465723500000017pt/>, <img src="svgs/d4589f7200e3fcdef519ed20d7c01f9e.svg?invert_in_darkmode" align=middle width=82.19200604999999pt height=24.65753399999998pt/>, inlet gas temperature, <img src="svgs/c143d9d00975e5aeff727a1ae57f2c91.svg?invert_in_darkmode" align=middle width=81.27876734999998pt height=24.65753399999998pt/> [K], and velocity, <img src="svgs/499b56b9cbe31d7851a2b0d03e1c29e6.svg?invert_in_darkmode" align=middle width=115.06884509999999pt height=24.65753399999998pt/> [m/s]. The bash script to run this problem is listed below

The bash script to run this problem is :

````
exec=$TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x
use_prefixPath=false
inputs=$TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas/
inputs_conditions=inputs/

chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
chemSurffile=$inputs"chemSurf.inp"
thermSurffile=$inputs"thermSurf.dat"
samplefile=$inputs_conditions"sample.dat"
inputSurffile=$inputs_conditions"inputSurf.dat"
inputVelocityfile=$inputs_conditions"inputVelocity.dat"

save=1
dzmin=1e-12
dzmax=1e-5
zend=0.025
max_newton_iterations=100
max_z_iterations=2000
atol_newton=1e-12
rtol_newton=1e-8
tol_z=1e-8
Area=0.00053
Pcat=0.025977239243415308
transient_initial_condition=true
initial_condition=false

$exec --use_prefixPath=$use_prefixPath --chemfile=$chemfile --thermfile=$thermfile --chemSurffile=$chemSurffile --thermSurffile=$thermSurffile --samplefile=$samplefile --inputSurffile=$inputSurffile --inputVelocityfile=$inputVelocityfile --initial_condition=$initial_condition --transient_initial_condition=$transient_initial_condition --Area=$Area  --Pcat=$Pcat --tol-z=$tol_z --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dzmin=$dzmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dzmax=$dzmax --zend=$zend --max-time-iterations=$max_z_iterations
````

In the above script we did not use a prefix path ("use_prefixPath=false") instead we provided the name of the inputs files: "chemfile", "thermfile",  "chemSurffile", "thermSurffile", "samplefile", "inputSurffile", "inputVelocityfile". The files for the reaction mechanism ("chem.inp" and "chemSurf.inp") and the thermo files ("therm.dat" and "thermSurf.dat") are located under "TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas/". The files with the inlet conditions ("sample.dat", "inputSurf.dat" and"inputVelocity.dat") are located in the "input" directory, located under the run directory. One can set a different path for the input files with the command-line option "use_prefixPath". Additionally, one can also use the option "transient_initial_condition=true", to activate the transient solver to find initial condition for the [PFR](#initialconditionforpfrproblem).

The following figures show temperature, gas-phase species mass fractions and surface species site fractions corresponding to the example presented above.

![Temperature](src/markdown/Figures/CH4-PTnogas_SA/TempSamplesPFR.jpg)
Figure. Gas temperature for 8 sample.

![<img src="svgs/07e86e45b7ebe272704b69abf2775d41.svg?invert_in_darkmode" align=middle width=19.09133654999999pt height=22.465723500000017pt/>](src/markdown/Figures/CH4-PTnogas_SA/O2SamplesPFR.jpg)
Figure. Mass fraction <img src="svgs/07e86e45b7ebe272704b69abf2775d41.svg?invert_in_darkmode" align=middle width=19.09133654999999pt height=22.465723500000017pt/> for 8 sample.

![<img src="svgs/ab1eb97d31e59bc83bdd3563750ef4ae.svg?invert_in_darkmode" align=middle width=42.69636194999999pt height=22.465723500000017pt/>](src/markdown/Figures/CH4-PTnogas_SA/CH4SamplesPFR.jpg)
Figure. Mass fraction <img src="svgs/ae893ddc6a1df993f14669218c18b5e1.svg?invert_in_darkmode" align=middle width=33.14158649999999pt height=22.465723500000017pt/> for 8 sample.

![<img src="svgs/f74275c195a72099e0b06f584fd5489a.svg?invert_in_darkmode" align=middle width=25.920062849999987pt height=22.465723500000017pt/>](src/markdown/Figures/CH4-PTnogas_SA/COSamplesPFR.jpg)
Figure. Mass fraction <img src="svgs/f74275c195a72099e0b06f584fd5489a.svg?invert_in_darkmode" align=middle width=25.920062849999987pt height=22.465723500000017pt/> for 8 sample.

![<img src="svgs/2f38b966661ff4345cf5e316b9f242fc.svg?invert_in_darkmode" align=middle width=27.99541469999999pt height=22.465723500000017pt/>](src/markdown/Figures/CH4-PTnogas_SA/OHSamplesPFR.jpg)
Figure. Mass fraction <img src="svgs/2f38b966661ff4345cf5e316b9f242fc.svg?invert_in_darkmode" align=middle width=27.99541469999999pt height=22.465723500000017pt/> for 8 sample.

![<img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/>](src/markdown/Figures/CH4-PTnogas_SA/XSamplesPFR.jpg)
Figure. Site fraction <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> for 8 sample.

![<img src="svgs/9f3d75b8ba9e0e89cce5bd92a28d04c6.svg?invert_in_darkmode" align=middle width=27.904113599999988pt height=22.465723500000017pt/>](src/markdown/Figures/CH4-PTnogas_SA/OXSamplesPFR.jpg)
Figure. Site fraction <img src="svgs/9f3d75b8ba9e0e89cce5bd92a28d04c6.svg?invert_in_darkmode" align=middle width=27.904113599999988pt height=22.465723500000017pt/> for 8 sample.

![<img src="svgs/219aa3e038365b48be98d9ff4c5f718f.svg?invert_in_darkmode" align=middle width=51.052504799999994pt height=22.465723500000017pt/>](src/markdown/Figures/CH4-PTnogas_SA/CH4XSamplesPFR.jpg)
Figure. Site fraction <img src="svgs/8a5024844db7df08da90ba8fe00b632e.svg?invert_in_darkmode" align=middle width=48.87218819999999pt height=22.465723500000017pt/> for 8 sample.

<a name="initialconditionforpfrproblem"></a>

#### 6.3.4\. Initial Condition for PFR Problem
The initial condition for the PFR problem must satisfy the algebraic constraint in the DAE system. Thus, an appropriate initial condition must be provided. To solve this problem, TChem first solves a system that accounts for the constraint only. The gas-phase species mass fractions and temperature are kept constant. The constraint component can be solved either by evolving an equivalent time-dependent formulation to steady-state or by directly solving the non-linear problem directly. a steady state or a time dependent formulation. In one method, the following equation is resolved in time until the system reaches stable state. In the second method, a newton solver is used to directly resolver the constraint part(<img src="svgs/07504631eadd80ba7c0b6c51fb64bcaf.svg?invert_in_darkmode" align=middle width=45.930262949999985pt height=21.95701200000001pt/>).

<p align="center"><img src="svgs/7e995e66eac58643829660e29b7eb60b.svg?invert_in_darkmode" align=middle width=316.3236417pt height=33.81208709999999pt/></p>

 In the [first method](#cxx-api-SimpleSurface), the ODE system is solved until reaches steady state. This is presented at "TCHEM_REPOSITORY_PATH/src/example/TChem_SimpleSurface.cpp". The following figure shows three surface species, the other species have values lower than 1e-4. This result shows the time to reach stable state is only of 1e-4 s. In the PFR example presented above, this option can be used setting  "transient_initial_condition=true" and  "initial_condition=false".


 ![site fraction of a few species](src/markdown/Figures/X/SimpleSurface.jpg)
 Figure.Site fractions for <img src="svgs/f2809fd51e9b0b17e492b0a02b04d50b.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> (empty space), <img src="svgs/27685d3014878a9b9183ce59dbfd814f.svg?invert_in_darkmode" align=middle width=24.65759669999999pt height=22.465723500000017pt/> and <img src="svgs/aa11c6d8353532b8b48a16d5c8e9f740.svg?invert_in_darkmode" align=middle width=36.52977569999999pt height=22.465723500000017pt/>. We start this simulation with an empty surface (<img src="svgs/a330a58ca19e6bf0df2960a30083d29c.svg?invert_in_darkmode" align=middle width=42.46563914999999pt height=22.465723500000017pt/>).


The example produces an output file ("InitialConditionPFR.dat") with the last iteration. This file can be used in the PFR problem as the"inputSurf.dat" file. The inputs for this example are located at "TCHEM_INSTALL_PATH/example/runs/InitialConditionPFR".

In the [second method](#cxx-api-InitialConditionSurface), we used a newton solver to find a solution to the constraint.  One code example of this alternative is presented at "TCHEM_REPOSITORY_PATH/src/example/TChem_InitialCondSurface.cpp". In the PFR example presented above, the default option is"initial_condition=true", if both option are set true, the code will execute the transient initial condition first and then the newton initial condition.

<a name="applicationprogramminginterface"></a>

## 7\. Application Programming Interface

<a name="c++"></a>

### 7.1\. C++

TChem provides two types of interface so called ``runHostBatch`` and ``runDeviceBatch`` for solving many problem instances in parallel. The ``runHostBatch`` use ``Kokkos::DefaultHostExecutionSpace`` and give data lives on host memory. On the other hand, ``runDeviceBatch`` dispatch works to ``Kokkos::DefaultExecutionSpace`` which is configured in Kokkos. In general, the default execution space is configured as OpenMP or Cuda upon its availability. When we use Cuda device, data should be transferred to the device memory using ``Kokkos::deep_copy``. An example in the below illustrates how to compute reaction rates of many samples. It reads kinetic model data and a collection of input state vectors. As the data files are available on the host memory, the input data is copied to the device memory. After computation is done, one can copy the device memory to the host memory to print the output.

```
#include "TChem_Util.hpp"
#include "TChem_ReactionRates.hpp"
#include "TChem_KineticModelData.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

int main() {
  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string periodictableFile("periodictable.dat");
  std::string inputFile("input.dat");
  std::string outputFile("omega.dat");

  Kokkos::initialize(argc, argv);
  {
    /// kinetic model is constructed and an object is constructed on host
    TChem::KineticModelData kmd(chemFile, thermFile, periodictableFile);

    /// kinetic model data is transferred to the device memory
    const auto kmcd = kmd.createConstData<TChem::exec_space>();

    /// input file includes the number of samples and the size of the state vector
    ordinal_type nBatch, stateVectorSize;
    TChem::readNumberOfSamplesAndStateVectorSize(inputFile, nBatch, stateVectorSize);

    /// create a 2d array storing the state vectors
    real_type_2d_view state("StateVector", nBatch, stateVectorSize);
    auto state_host = Kokkos::create_mirror_view(state);

    /// read the input file and store them into the host array
    TChem::readStateVectors(inputFile, state_host);
    /// if execution space is host execution space, this deep copy is a soft copy
    Kokkos::deep_copy(state, state_host);

    /// output: reaction rates (omega)
    real_type_2d_view omega("ReactionRates", nBatch, kmcd.nSpec);

    /// create a parallel policy with workspace
    /// for better performance, team size must be tuned instead of using AUTO
    Kokkos::TeamPolicy<TChem::exec_space>
      policy(TChem::exec_space(), nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = TChem::ReactionRates::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_scratch  =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    /// computes net production rates
    TChem::NetProductionRatePerMass::runDeviceBatch
      (policy,
       state,
       omega,
       kmcd);
    TChem::exec_space().fence();

    /// optionally, one can move the omega to host memory
    auto omega_host = Kokkos::create_mirror_view(omega);
    Kokkos::deep_copy(omega_host, omega);

    /// one may want to print omega_host
    for (ordinal_type s=0;s<nBatch;++s) {
      std::cout << "Sample ID = " << s << std::endl;
      for (ordinal_type k=0;k<kmcd.nSpec;++k)
         std::cout << omega_host(s, k) << std::endl;
    }
  }
  Kokkos::finalize();

  return 0;
}
```

This pattern can be applied for the other similar functions.
* [SpecificHeatCapacityPerMass](cxx-api-SpecificHeatCapacityPerMass)
* [EnthalpyMass](cxx-api-EnthalpyMass)
* [ReactionRates](cxx-api-ReactionRates)


Time ODE and DAE systems require a different workflow from the above example. It needs a time advance object including the range of time integration, time step sizes, newton solver tolerence, etc. The following example shows parameters that a user can set for their own problems.
```
#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_IgnitionZeroD.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using time_advance_type = TChem::time_advance_type;

using real_type_0d_view = TChem::real_type_0d_view;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

using time_advance_type_0d_view = TChem::time_advance_type_0d_view;
using time_advance_type_1d_view = TChem::time_advance_type_1d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

using time_advance_type_0d_view_host = TChem::time_advance_type_0d_view_host;
using time_advance_type_1d_view_host = TChem::time_advance_type_1d_view_host;

int main(int argc, char *argv[]) {
  /// input files
  std::string chemFile("chem.inp");
  std::string thermFile("therm.dat");
  std::string periodictableFile("periodictable.dat");
  std::string inputFile("input.dat");

  /// time stepping parameters
  /// the range of time begin and end
  real_type tbeg(0), tend(1);  
  /// min and max time step size
  real_type dtmin(1e-11), dtmax(1e-6);
  /// maximum number of time iterations computed in a single kernels launch
  ordinal_type num_time_iterations_per_interval(1);
  /// adaptive time stepping tolerance which is compared with the error estimator
  real_type tol_time(1e-8);
  /// new ton solver absolute and relative tolerence
  real_type atol_newton(1e-8), rtol_newton(1e-5);
  /// max number of newton iterations
  ordinal_Type max_num_newton_iterations(100);
  /// max number of time ODE kernel launch
  ordinal_type max_num_time_iterations(1e3);

  Kokkos::initialize(argc, argv);
  {
    /// kinetic model is constructed and an object is constructed on host
    TChem::KineticModelData kmd(chemFile, thermFile, periodictableFile);

    /// kinetic model data is transferred to the device memory
    const auto kmcd = kmd.createConstData<TChem::exec_space>();

    /// input file includes the number of samples and the size of the state vector
    ordinal_type nBatch, stateVectorSize;
    TChem::readNumberOfSamplesAndStateVectorSize(inputFile, nBatch, stateVectorSize);

    /// create a 2d array storing the state vectors
    real_type_2d_view state("StateVector", nBatch, stateVectorSize);
    auto state_host = Kokkos::create_mirror_view(state);

    /// read the input file and store them into the host array
    TChem::readStateVectors(inputFile, state_host);
    /// if execution space is host execution space, this deep copy is a soft copy
    Kokkos::deep_copy(state, state_host);

    /// create time advance objects
    time_advance_type tadv_default;
    tadv_default._tbeg = tbeg;
    tadv_default._tend = tend;
    tadv_default._dt = dtmin;
    tadv_default._dtmin = dtmin;
    tadv_default._dtmax = dtmax;
    tadv_default._tol_time = tol_time;
    tadv_default._atol_newton = atol_newton;
    tadv_default._rtol_newton = rtol_newton;
    tadv_default._max_num_newton_iterations = max_num_newton_iterations;
    tadv_default._num_time_iterations_per_interval = num_time_iterations_per_interval;

    /// each sample is time-integrated independently
    time_advance_type_1d_view tadv("tadv", nBatch);
    Kokkos::deep_copy(tadv, tadv_default);

    /// for print the time evolution of species, we need a host mirror view
    auto tadv_host = Kokkos::create_mirror_view(tadv);
    auto state_host = Kokkos::create_mirror_view(state);

    /// create a parallel execution policy with workspace
    Kokkos::TeamPolicy<TChem::exec_space>
      policy(TChem::exec_space(), nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = TChem::IgnitionZeroD::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_scratch  =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    for (; iter < max_num_time_iterations && tsum <= tend; ++iter) {
      /// in each kernel launch, it computes the number of time iterations per
      /// interval
      TChem::IgnitionZeroD::runDeviceBatch
        (policy,
         tadv, state, /// input
         t, dt, state, /// output
         kmcd);
      Kokkos::fence();

      /// terminate this loop when all samples reach the time end
      tsum = zero;
      Kokkos::parallel_reduce(
        Kokkos::RangePolicy<TChem::exec_space>(0, nBatch),
        KOKKOS_LAMBDA(const ordinal_type &i, real_type &update) {
          tadv(i)._tbeg = t(i);
          tadv(i)._dt = dt(i);
          update += t(i);
        },
        tsum);
      Kokkos::fence();
      tsum /= nBatch;

      /// to store or print the state vectors, the data must be transferred to
      /// host memory
      Kokkos::deep_copy(tadv_host, tadv);
      Kokkos::deep_copy(state_host, state);
      UserDefinedPrintStateVector(tadv_host, state_host);
    }
  }
  Kokkos::finalize();
}
```
A similar pattern can be applied for the following functions.
* [IgnitionZeroD](cxx-api-IgnitionZeroD)
* [PlugFlowReactor](cxx-api-PlugFlowReactor)

<a name="functionlist"></a>

#### 7.1.1\. Function List

This section lists all top-level function interface. Here, so-called top-level interface means that the function launches a parallel kernel with a given parallel execution policy.

<a name="cxx-api-SpecificHeatCapacityPerMass"></a>
<a name="specificheatcapacitypermass"></a>

##### 7.1.1.1\. SpecificHeatCapacityPerMass
```
/// Specific heat capacity per mass
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CpMass - rank 2d array sized by nBatch x nSpec storing Cp per species
///   [out] CpMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_SpecificHeatCapacityPerMass.hpp"
TChem::SpecificHeatCapacityPerMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &CpMass,
   const real_type_1d_view &CpMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-SpecificHeatCapacityConsVolumePerMass"></a>
<a name="specificheatcapacityconsvolumepermass"></a>

##### 7.1.1.2\. SpecificHeatCapacityConsVolumePerMass
```
/// Specific heat capacity per mass
/// ===============================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] CvMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_SpecificHeatCapacityPerMass.hpp"
TChem::SpecificHeatCapacityPerMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_1d_view &CpMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-EnthalpyMass"></a>
<a name="enthalpymass"></a>

##### 7.1.1.3\. EnthalpyMass
```
/// Enthalpy per mass
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EnthalpyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EnthalpyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_EnthalpyMass.hpp"
TChem::EnthalpyMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &EnthalpyMass,
   const real_type_1d_view &EnthalpyMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-EntropyMass"></a>
<a name="entropymass"></a>

##### 7.1.1.4\. EntropyMass
```
/// Entropy per mass
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] EntropyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] EntropyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_EntropyMass.hpp"
TChem::EntropyMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &EntropyMass,
   const real_type_1d_view &EntropyMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-InternalEnergyMass"></a>
<a name="internalenergymass"></a>

##### 7.1.1.5\. InternalEnergyMass
```
/// Internal Energy per mass
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] InternalEnergyMass - rank 2d array sized by nBatch x nSpec storing enthalpy per species
///   [out] InternalEnergyMixMass - rank 1d array sized by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_InternalEnergyMass.hpp"
TChem::InternalEnergyMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &InternalEnergyMass,
   const real_type_1d_view &InternalEnergyMixMass,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-ReactionRates"></a>
<a name="netproductionratespermass"></a>

##### 7.1.1.6\. NetProductionRatesPerMass
```
/// Net Production Rates per mass
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_NetProductionRatePerMass.hpp"
TChem::NetProductionRatePerMass::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &omega,
   const KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-ReactionRatesMole"></a>
<a name="netproductionratespermole"></a>

##### 7.1.1.7\. NetProductionRatesPerMole
```
/// Net Production Rates per mole
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] omega - rank 2d array sized by nBatch x nSpec storing reaction rates
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_NetProductionRatePerMole.hpp"
TChem::NetProductionRatePerMole::runDeviceBatch
  (const team_policy_type &policy,
   const real_type_2d_view &state,
   const real_type_2d_view &omega,
   cosnt KineticModelConstDataDevice &kmcd);
```

<a name="cxx-api-ReactionRatesSurface"></a>
<a name="netproductionratesurfacepermass"></a>

##### 7.1.1.8\. NetProductionRateSurfacePerMass
We need to update this interface in the code: OD
````
/// Net Production Rates Surface per mass
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in device memory (Surface phase)
TChem::NetProductionRateSurfacePerMass::runDeviceBatch
  (const real_type_2d_view &state,
   const real_type_2d_view &zSurf,
   const real_type_2d_view &omega,
   const real_type_2d_view &omegaSurf,
   const KineticModelConstDataDevice &kmcd,
   const KineticSurfModelConstDataDevice &kmcdSurf);

````

<a name="cxx-api-ReactionRatesSurfaceMole"></a>
<a name="netproductionratesurfacepermole"></a>

##### 7.1.1.9\. NetProductionRateSurfacePerMole
We need to update this interface in the code: OD
````
/// Net Production Rates Surface per mole
/// ==============
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank 2d array sized by nBatch x nSpec(Surface)
///   [out] omega - rank 2d array sized by nBatch x nSpec(Gas) storing reaction rates gas species
///   [out] omegaSurf - rank 2d array sized by nBatch x nSpec(Surface) storing reaction rates surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory(gas phase)
///   [in] kmcdSurf -  a const object of kinetic model storing in device memory (Surface phase)
TChem::NetProductionRateSurfacePerMole::runDeviceBatch
  (const real_type_2d_view &state,
   const real_type_2d_view &zSurf,
   const real_type_2d_view &omega,
   const real_type_2d_view &omegaSurf,
   const KineticModelConstDataDevice &kmcd,
   const KineticSurfModelConstDataDevice &kmcdSurf);

````

<a name="cxx-api-IgnitionZeroD"></a>
<a name="ignition0d"></a>

##### 7.1.1.10\. Ignition 0D
```
/// Ignition 0D
/// ===========
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [in] kmcd -  a const object of kinetic model storing in device memory
###inclue "TChem_IgnitionZeroD.hpp"
TChem::IgnitionZeroD::runDeviceBatch
  (const team_policy_type &policy,
   const time_advance_type_1d_view &tadv,
   const real_type_2d_view &state,
   const real_type_1d_view &t_out,
   const real_type_1d_view &dt_out,
   const real_type_2d_view &state_out,
   cosnt KineticModelConstDataDevice &kmcd);
```
<a name="cxx-api-PlugFlowReactor"></a>
<a name="plugflowreactor"></a>

##### 7.1.1.11\. PlugFlowReactor
```
/// Plug Flow Reactor
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank2d array by nBatch x number of surface specues
///   [in] velociy -rank1d array by nBatch
///   [out] t_out - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt_out - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] state_out - rank 2d array sized by nBatch x stateVectorSize storing updated state vectors
///   [out] z_out - rank2d array by nBatch x number of surface specues
///   [out] velocity_out -rank1d array by nBatch
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
///   [in] area - cross-sectional area
///   [in] pcat - chemically active perimeter
###inclue "TChem_PlugFlowReactor.hpp"
TChem::PlugFlowReactor::runDeviceBatch
  (const team_policy_type &policy,
   const time_advance_type_1d_view &tadv,
   const real_type_2d_view &state,
   const real_type_2d_view &z_surf,
   const real_type_1d_view &velocity,
   const real_type_1d_view &t_out,
   const real_type_1d_view &dt_out,
   const real_type_2d_view &state_out,
   const real_type_2d_view &z_out,
   const real_type_1d_view &velocity_out,
   const KineticModelConstDataDevice &kmcd,
   const KineticSurfModelConstDataDevice &kmcdSurf,
   const real_type area,
   const real_type pcat);
```

<a name="cxx-api-SimpleSurface"></a>
<a name="simplesurface"></a>

##### 7.1.1.12\. SimpleSurface

```
/// Simple surface
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] tadv - rank 1d array sized by nBatch storing time stepping data structure
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] siteFraction - rank2d array by nBatch x number of surface species
///   [out] t - rank 1d array sized by nBatch storing time when exiting the function
///   [out] dt - rank 1d array sized by nBatch storing time step size when exiting the function
///   [out] siteFraction_out - rank2d array by nBatch x number of surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
#include "TChem_SimpleSurface.hpp"
TChem::SimpleSurface::runDeviceBatch
(const team_policy_type &policy,
 const time_advance_type_1d_view &tadv,
 const real_type_2d_view &state,
 const real_type_2d_view &siteFraction,
 const real_type_1d_view &t,
 const real_type_1d_view &dt,
 const real_type_2d_view  &siteFraction_out,
 const KineticModelConstDataDevice &kmcd,
 const KineticSurfModelConstDataDevice &kmcdSurf);

```

<a name="cxx-api-InitialConditionSurface"></a>
<a name="initialconditionsurface"></a>

##### 7.1.1.13\. InitialConditionSurface
```
/// InitialConditionSurface
/// =================
///   [in] policy - Kokkos parallel execution policy; league size must be nBatch
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] siteFraction - rank2d array by nBatch x number of surface species
///   [out] siteFraction_out - rank2d array by nBatch x number of surface species
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory
#include "TChem_InitialCondSurface.hpp"
TChem::InitialCondSurface::runDeviceBatch
(const team_policy_type &policy,
 const real_type_2d_view &state,
 const real_type_2d_view &siteFraction,
 const real_type_2d_view  &siteFraction_out,
 const KineticModelConstDataDevice &kmcd,
const KineticSurfModelConstDataDevice &kmcdSurf);
```

<a name="cxx-api-RateOfProgress"></a>
<a name="rateofprogress"></a>

##### 7.1.1.14\. RateOfProgress
```
/// RateOfProgress
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [out] RoPFor - rank2d array by nBatch x number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_RateOfProgress.hpp"
TChem::RateOfProgress::runDeviceBatch
( const ordinal_type nBatch,
  const real_type_2d_view& state,
  const real_type_2d_view& RoPFor,
  const real_type_2d_view& RoPRev,
  const KineticModelConstDataDevice& kmcd);
```

<a name="cxx-api-SourceTerm"></a>
<a name="sourceterm"></a>

##### 7.1.1.15\. SourceTerm
```
/// SourceTerm
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] SourceTerm - rank2d array by nBatch x number of species + 1 (temperature)
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_SourceTerm.hpp"
  TChem::SourceTerm::runDeviceBatch
  (const ordinal_type nBatch,
  const real_type_2d_view& state,
  const real_type_2d_view& SourceTerm,
  const KineticModelConstDataDevice& kmcd);
```
<a name="cxx-api-Smatrix"></a>
<a name="smatrix"></a>

##### 7.1.1.16\. Smatrix
```
/// S Matrix
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Smatrix - rank3d array by nBatch  x number of species + 1 x twice the number of reaction in gas phase
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_Smatrix.hpp"
TChem::Smatrix::runDeviceBatch
  (const ordinal_type nBatch,
   const real_type_2d_view& state,
   const real_type_3d_view& Smatrix,
   const KineticModelConstDataDevice& kmcd);
```

<a name="cxx-api-IgnitionZeroDNumJacobian"></a>
<a name="ignitionzerodnumjacobian"></a>

##### 7.1.1.17\. IgnitionZeroDNumJacobian
```
/// IgnitionZeroDNumJacobian
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] jac - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [out] fac - rank 2d array by nBatch  x number of species + 1
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_IgnitionZeroDNumJacobian.hpp"
TChem::IgnitionZeroDNumJacobian::runDeviceBatch
  (const ordinal_type nBatch,
  const real_type_2d_view& state,
  const real_type_3d_view& jac,
  const real_type_2d_view& fac,
  const KineticModelConstDataDevice& kmcd);
```       

<a name="cxx-api-JacobianReduced"></a>
<a name="jacobianreduced"></a>

##### 7.1.1.18\. JacobianReduced
```
/// JacobianReduced
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [out] Jacobian - rank 3d array by nBatch  x number of species + 1 x number of species + 1  
///   [in] kmcd -  a const object of kinetic model storing in device memory
#include "TChem_JacobianReduced.hpp"
TChem::JacobianReduced::runDeviceBatch
(const ordinal_type nBatch,
const real_type_2d_view& state,
const real_type_3d_view& Jacobian,
const KineticModelConstDataDevice& kmcd);
```
<a name="cxx-api-PlugFlowReactorRHS"></a>
<a name="plugflowreactorrhs"></a>

##### 7.1.1.19\. PlugFlowReactorRHS
```
/// Plug Flow Reactor RHS
/// =================
///   [in] nBatch - number of samples
///   [in] state - rank 2d array sized by nBatch x stateVectorSize
///   [in] zSurf - rank 2d array by nBatch x number of surface species
///   [in] velocity - rank 2d array sized by nBatch x stateVectorSize
///   [in] kmcd -  a const object of kinetic model storing in device memory
///   [in] kmcdSurf -  a const object of surface kinetic model storing in device memory

#include "TChem_PlugFlowReactorRHS.hpp"
TChem::PlugFlowReactorRHS::runDeviceBatch
(const ordinal_type nBatch,
 const real_type_2d_view& state,
 const real_type_2d_view& zSurf,
 const real_type_2d_view& velocity,
 const real_type_2d_view& rhs,
 const KineticModelConstDataDevice& kmcd,
 const KineticSurfModelConstDataDevice& kmcdSurf);
```    

<a name="on-goingandfutureworks"></a>

## 8\. On-going and Future Works

* YAML and HDF5 IO interface
* GPU performance optimization
* Exploring exponential time integrator 
<a name="acknowledgement"></a>

## 9\. Acknowledgement

This work is supported as part of the Computational Chemical Sciences Program funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Chemical Sciences, Geosciences and Biosciences Division.

Award number: 0000232253
