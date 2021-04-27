# Thermodynamic Properties

We first present conversion formulas and the gas-phase equation of state, followed by a description of molar and mass-based expression for several thermodynamic properties.

## Mass-Molar Conversions

The molar mass of the mixture, $W$ is computed as
$$
W=\sum_{k=1}^{N_{spec}} X_k W_k
=1\Bigg/\left(\sum_{k=1}^{N_{spec}} \frac{Y_k}{W_k}\right)
$$
where $X_k$ and $Y_k$ are the mole and mass fractions, respectively, of species $k$, and $W_k$ is the molecular weight of species $k$. Mass and mole fractions can be computed from each other as
$$
X_k=Y_k W/W_k,\,\, Y_k=X_kW_k/W
$$
The the molar concentration of species $k$ is given by $\mathfrak{X}_k=\rho Y_k/W_k=\rho X_k/W$, and the molar concentration of the mixture is given by
$$
\sum_{k=1}^{N_{spec}}\mathfrak{X}_k=\rho/W
$$


For problems that include heterogenous chemistry, the site fractions $Z_k$ describe the composition of species on the surface. The number of surface phases is denoted by $N_{phase}$ and the site fractions are normalized with respect to each phase.
$$
  \sum_{k=1}^{N_{spec}^{s,n}}Z_k^{(n)}=1,\,\,\, \mathrm{for }n=1,\ldots N_{phase}.
$$

Here, $N_{spec}^{s,n}$ is the number of species on surface phase $n$. TChem currently handles $1$ surface phase only, $N_{phase}=1$. The surface concentration of surface species $k$ is given by

$$
  \mathfrak{X}_k=Z_k^{(n)} \Gamma_n\big/\sigma_k(n)
$$

where $\Gamma_n$ is the surface site density of surface phase $n$ and $\sigma_k(n)$ is the site occupancy number for species $k$. $\sigma_k(n)$ represents the number of sites in phase $n$ occupied by species $k$.


## Equation of State    

The ideal gas equation of state is used throughout the library,
$$
P=\rho\frac{R}{\sum_{k=1}^{N_{spec}} X_kW_k}T=\rho R\left(\sum_{k=1}^{N_{spec}}\frac{Y_k}{ W_k}\right)T=\rho\frac{R}{W}T=\left(\sum_{k=1}^{N_{spec}}\mathfrak{X}_k\right)R T
$$
where $P$ is the thermodynamic pressure, $W$ and $W_k$ are the molecular weights of the mixture and of species $k$, respectively, $T$ is the temperature, and $\mathfrak{X}_k$ is the molar concentration of species $k$.


## Gas-Phase Properties

The standard-state thermodynamic properties for a thermally perfect gas are computed based on NASA polynomials \cite{McBride:1993}. The molar heat capacity at constant pressure for species $k$ is computed as
$$
\frac{C_{p,k}}{R}=a_{0,k}+T(a_{1,k}+T(a_{2,k}+T\left(a_{3,k}+a_{4,k}T
\right)))
$$
where $R$ the universal gas constant. The molar enthalpy is computed as
$$
\frac{{H}_k}{R}=\int_{T_0}^T C_{p,k}dT+H_{k,T_0}
=T \left(a_{0,k}+T\left(\frac{a_{1,k}}{2}+T\left(\frac{a_{2,k}}{3}
+T\left(\frac{a_{3,k}}{4}+\frac{a_{4,k}}{5}T\right)\right)\right)\right)+a_{5,k}
$$

The molar entropy is given by
$$
\frac{S_k^0}{R}=\int_{T_0}^T\frac{C_{p,k}}{T}dT+S_{k,T_0}
=a_{0,k}\ln T+T\left(a_{1,k}+T\left(\frac{a_{2,k}}{2}
+T\left(\frac{a_{3,k}}{3}+\frac{a_{4,k}}{4}T\right)\right)\right)+a_{6,k}
$$
The temperature units are Kelvin in the polynomial expressions above. Other thermodynamics properties are computed based on the polynomial fits above. The molar heat capacity at constant volume $C_{v,k}$, the internal energy $U_{k}$, and the Gibbs free energy $G_{k}$:
$$
C_{v,k}=C_{p,k}-R,\,\,\,U_{k}=H_{k} - R T,\,\,\, G_k^0 = H_k-T S_{k}^0
$$

The mixture properties in molar units are given by
$$
C_p= \sum_{k=1} ^{N_{spec}} X_k C_{p,k},\,\,\,C_v= \sum_{k=1} ^{N_{spec}} X_k C_{v,k},\,\,\,
H= \sum_{k=1} ^{N_{spec}} X_k {H}_k,\,\,\, U= \sum_{k=1} ^{N_{spec}} X_k {U}_k
$$
where $X_k$ the mole fraction of species $k$. The entropy and Gibbs free energy for species $k$ account for the entropy of mixing and thermodynamic pressure
$$
S_k=S_k^0-R\ln (X_k\frac{P}{P_{atm}}),\,\,\, G_k=S_k-T S_k
$$
The mixture values for these properties are computed as above
$$
S=\sum_{k=1}^{N_{spec}} X_k S_k,\,\,\,G=\sum_{k=1}^{N_{spec}} X_k G_k
$$

The specific thermodynamic properties in mass units are obtained by dividing the above expression by the species molecular weight, $W_k$,
$$
c_{p,k}= C_{p,k}/W_k,\,\,\,c_{v,k}=C_{v,k}/W_k,\,\,\,h_k=H_k/W_k,\,\,\,u_k=U_k/W_k,\,\,\,
s_k^0= S_k^0/W_k,\,\,\,
g_k^0=G_k^0/W_k
$$
and
$$
s_k= S_k/W_k,\,\,\,g_k=G_k/W_k
$$
For the thermodynamic properties in mass units the mixture properties are given by
$$
c_p= \sum_{k=1}^{N_{spec}} Y_k c_{p,k},\,\,\, c_v=\sum_{k=1}^{N_{spec}} Y_k c_{v,k},\,\,\,
h=\sum_{k=1}^{N_{spec}} Y_k{h}_k,\,\,\, u=\sum_{k=1}^{N_{spec}} Y_k u_k,\,\,\,
s=\sum_{k=1}^{N_{spec}} Y_k s_k,\,\,\,g=\sum_{k=1} ^{N_{spec}} Y_k g_k
$$
where $Y_k$ the mass fraction of species $k$.

The mixture properties in mass units can also be evaluated from the equivalent molar properties as
$$
c_p=C_p/W,\,\,\,c_v=C_v/W,\,\,\,h_k=H/W,\,\,\,u=U/W,\,\,\,
s=S/W,\,\,\,g=G/W
$$
where $W$ is the molecular weight of the mixture.
## Examples

A example to compute $c_{p}$ and $h$ in mass base is at "example/TChem_ThermalProperties.cpp". Enthalpy per species and the mixture enthalpy are computed with this [function call](#cxx-api-EnthalpyMass). Heat capacity per species and mixture with this [function call](#cxx-api-SpecificHeatCapacityPerMass). This example can be used in bath mode, and several sample are compute in one run. The next two figures were compute with 40000 samples changing temperature and equivalent ratio for methane/air mixtures.

![Enthalpy](Figures/gri3.0_OneSample/MixtureEnthalpy.jpg)
Figure. Mixture Enthalpy compute with gri3.0 mechanism.

![SpecificHeatCapacity](Figures/gri3.0_OneSample/MixtureSpecificHeatCapacity.jpg)
Figure.  Mixutre Specific Heat Capacity $C_p$ compute with gri3.0 mechanism.


## Surface Species Properties

The thermal properties of the surface species are computed with the same equation used by the gas phase describe above.

<!-- ## Examples -->
