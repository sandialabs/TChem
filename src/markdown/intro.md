# Introduction

The TChem open-source software is a toolkit for computing thermodynamic properties, source term, and source term's Jacobian matrix for chemical kinetic models that involve gas and surface reactions. The software provides support for


* complex kinetic models for gas-phase and surface chemistry,
* thermodynamic properties based on NASA polynomials,
* parser for Chemkin/Cantera-YAML input files,
* species production/consumption rates,
* canonical reactor models such as constant pressure homogeneous gas-phase ignition, constant volume homogeneous gas-phase ignition, transient continuously stirred tank reactor, and plug-flow reactor,
* automatic evaluation of source term's Jacobian matrix using either finite difference schemes or automatic differentiation via the SACADO library.

TChem v3 is written in C++ and is based on [Kokkos](https://github.com/kokkos) for portability to heterogenous computing platforms i.e., multi/many core computing platforms with GPU (Graphics Processing Unit) accelerators. The toolkit includes gas-phase and surface chemistry models and several canonical reactor model examples including homogenous ignition, Plug Flow Reactor (PFR), and Transient Continuously Stirred Tank Reactors (T-CSTR). To exploit the massive parallelism available from modern computing platforms, the current software interface is designed to evaluate samples in parallel, which enables large scale parametric studies, e.g., for sensitivity analysis and model calibration. TChem v3 adds new interfaces for Python, C, and Fortran as well as kinetic model specifications in YAML format to facilitate the use of TChem in a broader range of applications.


## Citing

* Kyungjoo Kim, Oscar Diaz-Ibarra, Cosmin Safta, and Habib Najm, TChem v3.0 - A Software Toolkit for the Analysis of Complex Kinetic Models, Sandia National Laboratories, SAND 2021-14064, 2021.

## Nomenclature

In the table below, $ro$ stands for reaction order, for the forward and reverse paths, respectively.

Symbol|Description|Units
--|--|--
$\> N_{spec}$ | number of species   |-
$\> \> N_{spec}^g$ | number of gas-phase species |-
$\> \> N_{spec}^s$ | number of surface species   |-
$\> \> N_{spec}^{s,n}$ | number of surface species in phase $n$ |-
$\>N_{reac}$ | number of reactions |-
$\>\rho$ | gas-phase density | kg/m$^3$
$\>P$   |  thermodynamic pressure |  Pa
$\>T$   |  temperature  | K
$\>C_p$ |  mixture heat capacity at constant pressure | J/(K.kmol)
$\> \> C_{p,k}$ | for species $k$ | J/(K.kmol)
$\> \> c_{p}$ |  specific | J/(K.kg)
$\> \> c_{p,k}$ |  specific, for species $k$ | J/(K.kg)
$\>H$ | mixture molar enthalpy | J/kmol
$\> \>H_{k}$ | for species $k$ | J/kmol
$\> \>h_{p}$ |  specific| J/kg
$\> \>h_{p,k}$ |  specific, for species $k$| J/kg
$\> S   \> \> $ | mixture molar entropy  | J/(kmol.K)
$\> \> S_{k} \> $|   for species $k$ | J/(K.kmol)
$\> \> s \>$   |specific | J/(K.kg)
$\> \> s_{k} \>$ | specific, for species $k$ | J/(K.kg)
$\> G   \> \>$  |Gibbs free energy for the mixture | J/kmol
$\> \> G_{k} \> $ |for species $k$ | J/kmol
$\> \> g \>  $ |specific| J/kg
$\> \> g_{k} \>  $ | specific, for species $k$| J/kg
$\> Y_k \> \> $| mass fraction of species $k$  | -
$\> X_k \> \> $| mole fraction of species $k$  | -
$\> Z_k \> \> $| site fraction of species $k$  | -
$\> \> Z_k^{(n)} \> $| for species $k$ in phase $n$ | -
$\> \mathfrak{X}_k \> \> $|molar concentration of species $k$  | kmol/m$^3$
$\> \Gamma_n \> \> $|surface site density of phase $n$ | kmol/m$^2$
$\> \sigma_{k}(n) \> \>$|site occupancy by species $k$ in phase $n$|-
$\> W \> \> $|mixture molecular weight | kg/kmol
$\> \> W_{k} \> $| for species $k$ | kg/kmol
$\> R \> \> $|universal gas constant | J/(kmol.K)
$\> k_{fi} \> \>$| forward rate constant of reaction $i$ | $\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}$
$\> k_{ri} \> \>$| reverse rate constant of reaction $i$ | $\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}$
$\> \dot{q}_{i} \> \> $|rate of progress of reaction $i$ | kmol/(m$^3$.s)
$\> \gamma_{i} \> \> $|sticking coefficient for reaction $i$ | $\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}$
$\> \dot{\omega}_{k} \> \> $|molar production rate of species $k$ | kmol/(m$^3$.s)
$\> \dot{s}_{k} \> \> $|surface molar production rate of species $k$ | kmol/(m$^2$.s)
