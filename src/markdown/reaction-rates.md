# Reaction Rates

In this chapter we present reaction rate expressions for gas-phase reactions in [Section](#gas-phasechemistry) and for surface species or between surface and gas-phase species in [Section](#surface chemistry).


## [Gas-Phase Chemistry](#cxx-api-ReactionRates)

The production rate for species $k$ in molar units is written as
$$
\dot{\omega}_k=\sum_{i=1}^{N_{reac}}\nu_{ki}q_i,\,\,\, \nu_{ki}=\nu''_{ki}-\nu'_{ki},
$$
where $N_{reac}$ is the number of reactions and $\nu'_{ki}$ and $\nu''_{ki}$ are the stoichiometric coefficients of species $k$ in reaction $i$ for the reactant and product side of the reaction, respectively. The rate-of-progress of reaction $i$ is $q_i=\mathcal{C}_i\mathcal{R}_i$, with

$\mathcal{C}_i$|Reaction Type
--|--|--
$1$ | basic reaction
$\mathfrak{X}_i$ | 3-rd body enhanced, no pressure dependence
$\frac{\Pr_i}{1+\Pr_i}F_i$ | unimolecular/recombination fall-off reactions
$\frac{1}{1+\Pr_i}F_i$ | chemically activated bimolecular reactions

and
$$\mathcal{R}_i={k_f}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu'_{ji}}-
{k_r}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu''_{ji}}$$

The above expressions are detailed below.

### Forward and Reverse Rate Constants

The forward rate constant has typically an Arrhenius expression,
$$
{k_f}_i=A_iT^{\beta_i}\exp\left(-\frac{E_i}{RT}\right),
$$
where $A_i$, $\beta_i$, and $E_i$ are the pre-exponential factor, temperature exponent, and activation energy, respectively, for reaction $i$. For reactions with reverse Arrhenius parameters specified, the reverse rate constant ${k_r}_i$ is computed similar to ${k_f}_i$. If the reverse Arrhenius parameters are not specified, ${k_r}_i$ is computed as
$$
{k_r}_i={k_f}_i/{K_c}_i,
$$
where ${K_c}_i$ is the equilibrium constant (in concentration units) for reaction $i$
$$
{K_c}_i=\left(\frac{P_{atm}}{RT}\right)^{\sum_{k=1}^{N_{spec}}\nu_{ki}}{K_p}_i,\,\,\,
{K_p}_i=\exp\left(\sum_{k=1}^{N_{spec}}\nu_{ki}\left(\frac{S_k}{R}-\frac{H_k}{RT}\right)\right).
$$
When computing the equilibrium constant, the atmospheric pressure, $P_{atm}=1$atm, and the universal gas constant $R$ are converted to cgs units, dynes/cm$^2$ and erg/(mol.K), respectively.

Note: If a reaction is irreversible, $k_r=0$.

### Concentration of the "Third-Body"   

If the expression "+M" is present in the reaction string, some of the species might have custom efficiencies for their contribution in the mixture. For these reactions, the mixture concentration is computed as


$$
\mathfrak{X}_i=\sum_{j=1}^{N_{spec}}\alpha_{ij}\mathfrak{X}_j,
$$
where $\alpha_{ij}$ is the efficiency of species $j$ in reaction $i$ and $\mathfrak{X}_j$ is the concentration of species $j$. $\alpha_{ij}$ coefficients are set to 1 unless specified in the kinetic model description.

### Pressure-dependent Reactions

* Reduced pressure $\mathrm{Pr}_i$. If expression "(+M)" is used to describe a reaction, then $\mathrm{Pr}_i=\frac{{k_0}_i}{{k_\infty}_i}\mathfrak{X}_i$.
* For reactions that contain expressions like "(+$\Upsilon_m$)" ($\Upsilon_m$ is the name of species $m$), the reduced pressure is computed as $\mathrm{Pr}_i=\frac{{k_0}_i}{{k_\infty}_i}\mathfrak{X}_m$.

For *unimolecular/recombination fall-off reactions* the Arrhenius parameters for the high-pressure limit rate constant $k_\infty$ are given on the reaction line, while the parameters for the low-pressure limit rate constant $k_0$ are given on the auxiliary reaction line that contains the keyword **LOW**. For *chemically activated bimolecular reactions* the parameters for $k_0$ are given on the reaction line while the parameters for $k_\infty$ are given on the auxiliary reaction line that contains the keyword **HIGH**.

The following expressions are employed to compute the $F_i$:
$F_i$|Reaction Type
--|--
$1$ | Lindemann reaction
$F_{cent}^{1/\left(1+(A/B)^2\right)}$ | Troe reaction
$dT^e\bigl(a\exp\left(-\frac{b}{T}\right)+\exp\left(-\frac{T}{c}\right)\bigr)^X$ | SRI reaction

* For the Troe form, $F_{cent}$, $A$, and $B$ are
$$
F_{cent} = (1-a)\exp\left(-\frac{T}{T^{***}}\right)+a\exp\left(-\frac{T}{T^{*}}\right)
+\exp\left(-\frac{T^{**}}{T}\right),
$$
$$
A=\log_{10} \mathrm{Pr}_i-0.67\log_{10} F_{cent}-0.4,\,\,\, B=0.806-1.1762\log_{10} F_{cent} -0.14\log_{10}\mathrm{Pr}_i
$$
Parameters $a$, $T^{***}$, $T^{*}$, and $T^{**}$ are provided (in this order) in the kinetic model description for each Troe-type reaction. If $T^{**}$ is omitted, only the first two terms are used to compute $F_{cent}$.
* For the SRI form exponent $X$ is computed as
$$X=\left(1+\left(\log_{10}\Pr_i\right)^2\right)^{-1}.$$
Parameters $a$, $b$, $c$, $d$, and $e$ are provided in the kinetic model description for each SRI-type reaction. If $d$ and $e$ are omitted, these parameters are set to $d=1$ and $e=0$.

Miller~\cite{PLOGprinceton} has developed an alternative expression for the pressure dependence for pressure fall-off reactions that cannot be fitted with a single Arrhenius rate expression. This approach employs linear interpolation of $\log {k_f}_i$ as a function of pressure for reaction $i$ as follows
$$
\log {k_f}_i(T) = \log {k_f}_{i,l}(T)+(\log p-\log p_l)\frac{\log {k_f}_{i,l+1}(T)-\log {k_f}_{i,l}(T)}{\log p_{l+1}-\log p_l}
$$
Here, ${k_f}_{i,l}(T)=A_{i,l}T^{\beta_{i,l}}\exp\left(-\frac{E_{i,l}}{R T}\right)$ is the Arrhenius rate corresponding to pressure $p_l$. For $p<p_1$ the Arrhenius rate is set to ${k_f}_i={k_f}_{i,1}$, and similar for $p>p_{N_i}$ where $N_i$ is the number of pressures for which the Arrhenius factors are provided, for reaction $i$. This formulation can be combined with 3$^{\mathrm{rd}}$ body information, e.g. $\mathcal{C}_i=\mathfrak{X}_i$. For PLOG reactions for which there are multiple PLOG entries for each pressure value, the forward rate constants are evaluated as
$$
{k_f}_{i,l}(T)=\sum_{j=1}^{M_l} A_{i,l}^{(j)}T^{\beta_{i,l}^{(j)}}\exp\left(-\frac{E_{i,l}^{(j)}}{R T}\right)
$$
where $j=1,\ldots,M_l$ is the index for the entries corresponding to pressure $p_l$.

### Note on Units for Net Production rates

In most cases, the kinetic models input files contain parameters that are based on *calories, cm, moles, kelvin, seconds*. The mixture temperature and species molar concentrations are necessary to compute the reaction rate. Molar concentrations are computed as above are in [kmol/m$^3$]. For the purpose of reaction rate evaluation, the concentrations are transformed to [mol/cm$^3$]. The resulting reaction rates and species production rates are in [mol/(cm$^3$.s)]. In the last step these are converted to SI units [kg/(m$^3$.s)].

### Example

The production rate for species $k$ in mass units (kg/m$^3$/s) ($\dot{\omega}_k W_k$) is computed with the [function call](#cxx-api-ReactionRates) and in mole units ($\dot{\omega}_k$ kmol/m$^3$/s) with [function call](#cxx-api-ReactionRatesMole). A example is located at src/example/TChem_NetProductionRatesPerMass.cpp. This example computes the production rate in mass units for any type of gas reaction mechanism.

## [Surface Chemistry](#cxx-api-ReactionRatesSurface)

The production rate for gas and surface species $k$ in molar/$m^2$ units is written as

$$
\dot{s}_k=\sum_{i=1}^{N_{reac}}\nu_{ki}q_i,\,\,\, \nu_{ki}=\nu''_{ki}-\nu'_{ki},
$$

where $N_{reac}$ is the number of reactions on the surface phase and $\nu'_{ki}$ and $\nu''_{ki}$ are the stoichiometric coefficients of species $k$ in reaction $i$ for the reactant and product side of the reaction, respectively.

The rate of progress $q_i$ of the $ith$ surface reaction is equal to:

$$q_i={k_f}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu'_{ji}}-
{k_r}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu''_{ji}}$$


Where $\mathfrak{X}_j$ is the concentration of the species $j$. If the species $j$ is a gas species, this is the molar concentration ($\mathfrak{X}_j=\frac{Y_j \rho}{W_j}$). If, on the other hand, the species $j$ is a surface species, it the surface molar concentration computed by $\mathfrak{X}=\frac{Z_k\Gamma_n}{\sigma_{j,n}}$ is . $Z_j$ is site fraction, $\Gamma_n$ is density of surface site of the phase $n$, and $\sigma_{j,n}$ is the site occupancy number (We assume $\sigma_{j,n}=1$ ).

### Forward and Reverse Rate Constants

The forward rate constant is computed as we describe in the gas section. If parameters are not specified for reverse rate, this rate is computed with equilibrium constant defined by:

$$k_{r,i}=\frac{k_{f,i}}{K_{c,i}}$$

The equilibrium constant for the surface reaction $i$ is computed as

$$K_{c,i} = K_{p,i} \Big( \frac{p^o}{RT} \Big)^{\sum_k=1 ^{Kg}\nu_{ki}} \prod_{n=N_s^f}^{N_s^l} (\Gamma_n^o)^{\Delta \sigma_(n,i)} $$  

Here, $N_{spec}^g$ and $N_{spec}^s$ represent the number of gas-phase and surface species, respectively, and $p^o=1$atm. TChem currently assumes the surface site density $\Gamma_n$ for all phases to be constant. The equilibrium constant in pressure units is computed as

$$K_{p,i} = \exp\left(\frac{\Delta S^o_i}{R} -  \frac{\Delta H^o_i}{RT} \right)$$

based on entropy and enthalpy changes from reactants to products (including gas-phase and surface species). The net change for surface of the site occupancy number for phase $n$ for reaction $i$ is given by

$$ \Delta \sigma_(n,i)=\sum_{k=1}^{N_{spec}^{s,n}}\nu_{ki}\sigma_k(n)$$

## Sticking Coefficients

The reaction rate for some surface reactions are described in terms of the probability that a collision results in a reaction. For these reaction, the forward rate is computed as

$$k_{r,i} =\frac{\gamma_i}{(\Gamma_{Tot})^m} \sqrt{\frac{RT}{2 \pi W}}  A_iT^{\beta_i}\exp\left(-\frac{E_i}{RT}\right)  $$

where $\gamma_i$ is the sticking coefficient, $W$ is the molecular weight of the gas-phase mixture, $R$ is the universal gas constant, $\Gamma_{tot}$ is the total surface site concentration over all phases, and $m$ is the sum of stoichiometric coefficients for all surface species in reaction $i$.

### Note on Units for surface production rates

The units of the surface and gas species concentration presented above are in units of kmol/m$^2$ (surface species) or kmol/$m^3$ (gas species). To match the units of the kinetic model and compute the rate constants, we transformed the concentration units to mol/cm$^3$ or mol/cm$^2$. The resulting rate constant has units of mol/cm$^2$. In the last step these are converted to SI units [kg/(m$^2$.s)].

### Example

The production rate for species $k$ in mass units (kg/m$^2$/s) ($\dot{s}_k W_k$) is computed with the [function call](#cxx-api-ReactionRatesSurface), in molar units ( $\dot{s}_k$ kmole/m$^2$/s) with [function call](#cxx-api-ReactionRatesSurfaceMole). A example is located at src/example/TChem_NetProductionSurfacePerMass.cpp. In this example, we compute the production rates of gas phase and also the production rate of the surface phase in mass units.
