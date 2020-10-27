# Introduction

TChem is an open source software library for solving complex computational chemistry problems and analyzing detailed chemical kinetic models. The software provides support for
* the support of complex kinetic models for gas-phase and surface chemistry,
* thermodynamic properties based on NASA polynomials,
* species production/consumption rates,
* stable time integrator for solving stiff time ordinary differential equations,
* reactor models such as homogenous gas-phase ignition (with analytical Jacobians), continuously stirred tank reactor, plug-flow reactor.


This toolkit builds upon earlier versions that were written in C and featured tools for gas-phase chemistry only. The current version of the software was completely refactored in C++, uses an object-oriented programming model, and adopts [Kokkos](https://github.com/kokkos) as its portability layer to make it ready for the next generation computing architectures i.e., multi/many core computing platforms with GPU accelerators. We have expanded the range of kinetic models to include surface chemistry and have added examples pertaining to Continuously Stirred Tank Reactors (CSTR) and Plug Flow Reactor (PFR) models to complement the homogenous ignition examples present in the earlier versions. To exploit the massive parallelism available from modern computing platforms, the current software interface is designed to evaluate samples in parallel, which enables large scale parametric studies, e.g. for sensitivity analysis and model calibration.



## Citing

* Kyungjoo Kim, Oscar Diaz-Ibarra, Cosmin Safta, and Habib Najm, TChem v2.0 - A Software Toolkit for the Analysis of Complex Kinetic Models, Sandia National Laboratories, SAND 2020-10762, 2020.*

## Nomenclature

In the table below, $ro$ stands for reaction order, for the forward and reverse paths, respectively.

Symbol|Description|Units
--|--|--
$N_{spec}$ | Number of species | -
$N_{reac}$ | Number of reactions | -
$N_{spec}^g$ | number of gas-phase species |  -
$N_{spec}^s$ | number of surface species   |  -
$N_{spec}^{s,n}$ | number of surface species in phase $n$ |  -
$\rho$ | gas-phase density | kg/m$^3$
$P$ | thermodynamic pressure | Pa
$T$ | Temperature | K
$C_p$| Heat capacity at constant pressure| J/(K.kmol) |
$C_{p,k}$ |  for species $k$| J/(K.kmol) |
$c_{p}$ |   specific| J/(K.kg) |
$c_{p,k}$ |  specific, for species $k$| J/(K.kg) |
$H$ | Molar enthalpy of a mixture | J/kmol
$S$ | Molar entropy of a mixture | J/(kmol.K)
$Y_k$ | Mass fraction of species $k$ | -
$X_k$ | Mole fraction of species $k$ | -
$C_{p,k}$ | Heat capacity at constant pressure for species $k$ | J/(kmol.K)
$H_{k}$ | Molar enthalpy of $k$ species | J/kmol
$H_{k}$ | for species $k$ | J/kmol|
$h_{p}$ |  specific| J/kg|
$h_{p,k}$ |  specific, for species $k$| J/kg|
$S_{k}$ | Molar entropy of $k$ species | J/(kmol.K)
$S_{k}$ |  for species $k$| J/(K.kmol)|
$s$ |  specific| J/(K.kg)|
$s_{k}$ |  specific, for species $k$| J/(K.kg)|
$G_{k}$ | Gibbs free energy of $k$ species | J/kmol
$G_{k}$ | for species $k$ | J/kmol|
$g$ |  specific| J/kg|
$g_{k}$ |  specific, for species $k$| J/kg|
$\mathfrak{X}_k$ | Molar concentration of species $k$ | kmol/m$^3$
$Y_k$ |  mass fraction of species $k$  | -|
$X_k$ |  mole fraction of species $k$  | -|
$Z_k$ |  site fraction of species $k$  | -|
$Z_k^{(n)}$ | for species $k$ in phase $n$ | -|
$\Gamma_n$ | surface site density of phase $n$ | kmol/m$^2$|
$\sigma_{k}(n)$ |site occupancy by species $k$ in phase $n$| -|
$W$ | mixture molecular weight | kg/kmol|
$W_{k}$ | for species $k$ | kg/kmol|
$R$ | universal gas constant | J/(kmol.K)|
$k_{fi}$ | Forward rate constant of $i$ reaction | $\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}$
$k_{ri}$ | Reverse rate constant of $i$ reaction | $\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}$
$R$ | Universal gas constant | J/(kmol.K) |
$\dot{q}_{i}$ | Rate of progress of $i$ reaction | kmol/(m$^3$.s)
$\gamma_{i}$| sticking coefficient for reaction $i$ | $\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}$|
$\dot{\omega}_{k}$ | Production rate of $k$ species | kmol/(m$^3$.s)
$\dot{s}_{k}$ | surface molar production rate of species $k$ | kmol/(m$^2$.s)|
