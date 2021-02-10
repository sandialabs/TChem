# [Homogenous Batch Reactors](#cxx-api-IgnitionZeroD)

## Problem Definition

In this example we consider a transient zero-dimensional constant-pressure problem where temperature $T$ and species mass fractions for $N_{spec}$ gas-phase species are resolved in a batch reactor. In this problem an initial condition is set and a time integration solver will evolve the solution until a time provided by the user.

For an open batch reactor the system of ODEs solved by TChem are given by:
* ***Energy equation***

$$
\frac{dT}{dt}= -\frac{1}{\rho c_p}\sum_{k=1}^{N_{spec}}\dot{\omega_{k}} W_k h_k = S_T
$$

* ***Species equation***

$$
\frac{dY_k}{dt}=\frac{1}{\rho}\dot{\omega_{k}}W_k=S_{Y_k},\,\,\,k=1\ldots N_{spec}
$$

where $\rho$ is the density, $c_p$ is the specific heat at constant pressure for the mixture, $\dot{w_{k}}$ is the molar production rate of species $k$, $W_k$ is its molecular weight, and $h_k$ is the specific enthalpy.

## Jacobian Formulation

Efficient integration and accurate analysis of the stiff system of ODEs shown above requires the Jacobian matrix of the *rhs* vector. In this section we will derive the Jacobian matrix components.

Let

$$
\Phi=\left\{T,Y_1,Y_2,\ldots,Y_{N_{spec}}\right\}^T
$$

by the denote the variables in the *lhs* of the 0D system and let

$$
\tilde{\Phi}=\left\{\rho,P,T,Y_1,Y_2,\ldots,Y_{N_{spec}}\right\}^T
$$

be the extended state vector. The 0D system can be written in compact form as

$$
\frac{d\Phi}{dt}=f(\Phi)\,\,\,\mathrm{and}\,\,\,\frac{d\tilde{\Phi}}{dt}=\tilde{f}(\tilde{\Phi})
$$

where $f=\{S_T,S_{Y_1},\ldots S_{Y_{N_{spec}}}\}^T$ and $\tilde{f}=\{S_\rho,S_P,S_T,S_{Y_1},\ldots S_{Y_{N_{spec}}}\}^T$. The thermodynamic pressure $P$ was introduced for completeness. For open batch reactors $P$ is constant and $S_P\equiv 0$. The source term $S_\rho$ is computed considering the ideal gas equation of state

$$
P=\rho R \sum \frac{Y_i}{W_i} T
$$

with P=const and using the expressions above for $S_T$ and $S_{Y_k}$,

$$
S_\rho=-W\sum_{k=1}^{N_{spec}}\dot{\omega_k}+\frac{1}{c_p T}\sum_{k=1}^{N_{spec}}\dot{\omega_{k}} W_k h_k
$$

Let $\tilde{J}$ and $J$ be the Jacobian matrices corresponding to $\tilde{f}(\tilde{\Phi})$ and $f(\Phi)$, respectively. Chain-rule differentiation leads to

$$
\frac{\partial f_u}{\partial v}=\frac{\partial \tilde{f}_u}{\partial v}+\frac{\partial \tilde{f}_u}{\partial \rho}\frac{\partial \rho}{\partial v}
$$

Note that each component $u$ of $\Phi$ is also a component of $\tilde{\Phi}$ and the corresponding *rhs* components are also the same, $f_u(\Phi)=\tilde{f}_u(\tilde{\Phi})$.

### Evaluation of $\tilde{J}$ Components

We first identify the dependencies on the elements of $\tilde{\Phi}$ for each of the components of $\tilde{f}$

* $\tilde{f}_1=S_\rho$. We postpone the discussion for this component.

* $\tilde{f}_2=S_P=0$

* $\tilde{f}_3=S_T$. $S_T$ is defined above. Here we highlight its dependencies on the elements of $\tilde{\Phi}$

$$
c_p=\sum_{k=1}^{N_{spec}} Y_k{c_p}_k(T),\,\,\, h_k=h_k(T),\,\,\,\text{ and }\,\,\, \dot{\omega}_k=\dot{\omega}_k(T,\mathfrak{X}_1,\mathfrak{X}_2,\ldots,\mathfrak{X}_{N_{spec}}),
$$

where $\mathfrak{X}_k$ is the molar concentration of species $k$, $\mathfrak{X}_k=\rho Y_k/W_k$.

$$
\tilde{J}_{3,1}=\frac{\partial\tilde{f}_3}{\partial\rho}=\frac{1}{\rho c_p}\sum h_k\left(\frac{\dot{\omega}_k}{\rho}-\frac{\partial\dot{\omega}_k}{\partial\rho}\right),\,\,\,
\tilde{J}_{3,2}=0,\\
\tilde{J}_{3,3}=\frac{\partial\tilde{f}_3}{\partial T}
=\frac{1}{\rho c_p^2}\frac{d c_p}{d T}\sum h_k\dot{\omega}_k
-\frac{1}{\rho c_p}\sum {c_p}_k \dot{\omega}_k-\frac{1}{\rho c_p}\sum h_k\frac{\partial\dot{\omega}_k}{\partial T},\\
\tilde{J}_{3,3+j}= \frac{\partial\tilde{f}_3}{\partial Y_j}=\frac{1}{\rho c_p^2}{c_p}_j\sum h_k\dot{\omega}_k
-\frac{1}{\rho c_p}\sum h_k\frac{\partial\dot{\omega}_k}{\partial Y_j},\,\,\, j=1,2,\ldots,{N_{spec}}
$$

* $\tilde{f}_{3+k}=S_{Y_k}$

$$
\tilde{J}_{3+k,1}=\frac{\partial\tilde{f}_{3+k}}{\partial\rho}= \frac{W_k}{\rho}\left(\frac{\partial\dot{\omega}_k}{\partial\rho}-\frac{\dot{\omega}_k}{\rho}\right),\,\,\,
\tilde{J}_{3+k,2}=\frac{\partial\tilde{f}_{3+k}}{\partial P}\equiv 0,\\
\tilde{J}_{3+k,3}=\frac{\partial\tilde{f}_{3+k}}{\partial T}=\frac{W_k}{\rho}\frac{\partial\dot{\omega}_k}{\partial T},\,\,\,
\tilde{J}_{3+k,3+j}=\frac{\partial\tilde{f}_{3+k}}{\partial Y_j}=\frac{W_k}{\rho}\frac{\partial\dot{\omega}_k}{\partial Y_j},\,\,\, j,k=1,2,\ldots,{N_{spec}}
$$

The values for heat capacities and their derivatives are computed based on the NASA polynomial fits as

$$
\frac{\partial c_p}{\partial Y_k}= {c_p}_k,\,\,\,
\frac{\partial c_p}{\partial T}= \sum Y_k \frac{d{c_p}_k}{dT},\,\,\,
\frac{d{c_p}_k}{dT}=R_k\Bigl(a_{1,k}+T\bigl(2a_{2,k}+T\left(3a_{3,k}+4a_{4,k}T\right)\bigr)\Bigr)
$$

The partial derivatives of the species production rates,   $\dot{\omega}_k(T,\mathfrak{X}_1,\mathfrak{X}_2,\ldots)$, are computed as
as

$$
\left.\frac{\partial\dot{\omega}_k}{\partial\rho}\right\vert_{T,Y}=\sum_{l=1}^{N_{spec}}\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_l}\frac{\partial\mathfrak{X}_l}{\partial\rho}
+\frac{\partial\dot{\omega}_k}{\partial T}\underbrace{\frac{\partial T}{\partial\rho}}_{0}+\underbrace{\frac{\partial\dot{\omega}_k}{\partial\rho}}_{0}\frac{\partial\rho}{\partial\rho}
=\sum_{l=1}^{N_{spec}}\frac{Y_l}{W_l}\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_l},\\
\left.\frac{\partial\dot{\omega}_k}{\partial Y_j}\right\vert_{\rho,T,Y_{\neq j}}
=\sum_{l=1}^{N_{spec}}\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_l}\frac{\partial\mathfrak{X}_l}{\partial Y_j}
+\frac{\partial\dot{\omega}_k}{\partial T}\underbrace{\frac{\partial T}{\partial Y_j}}_{0}+\frac{\partial\dot{\omega}_k}{\partial\rho}\underbrace{\frac{\partial\rho}{\partial Y_j}}_{0}
=\frac{\rho}{W_j}\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_j}
$$

The steps for the calculation of $\frac{\partial\dot{\omega}_k}{\partial T}$ and $\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_l}$ are itemized below

* Derivatives of production rate $\dot{\omega}_k$ of species $k$

$$
\dot{\omega}_k=\sum_{i=1}^{N_{reac}}\nu_{ki}q_i \Rightarrow
\frac{\partial\dot{\omega}_k}{\partial T}=\sum_{i=1}^{N_{reac}}\nu_{ki}\frac{\partial q_i}{\partial T},\,\,\,
\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_l}=\sum_{i=1}^{N_{reac}}\nu_{ki}\frac{\partial q_i}{\partial\mathfrak{X}_l}
$$

* Derivatives of rate-of-progress variable $q_i$ of reaction $i$

$$
q_i=\mathcal{C}_i\mathcal{R}_i \Rightarrow \frac{\partial q_i}{\partial T} = \frac{\partial\mathcal{C}_i}{\partial T}\mathcal{R}_i+\mathcal{C}_i\frac{\partial\mathcal{R}_i}{\partial T},\,\,\,
\frac{\partial q_i}{\partial\mathfrak{X}_l} = \frac{\partial\mathcal{C}_i}{\partial \mathfrak{X}_l}\mathcal{R}_i+\mathcal{C}_i\frac{\partial\mathcal{R}_i}{\partial\mathfrak{X}_l}
$$

* Derivatives of $\mathcal{C}_i$

Basic reactions $\mathcal{C}_i = 1$: $\frac{\partial\mathcal{C}_i}{\partial T}\equiv \frac{\partial\mathcal{C}_i}{\partial\mathfrak{X}_l}\equiv 0$

3-rd body-enhanced reactions $\mathcal{C}_i = \mathfrak{X}_i$: $\frac{\partial\mathcal{C}_i}{\partial T}\equiv 0$, $\frac{\partial\mathcal{C}_i}{\partial\mathfrak{X}_l}=\alpha_{il}$

Unimolecular/recombination fall-off reactions $\mathcal{C}_i = \frac{\Pr_i}{1+\Pr_i}F_i$

$$
\frac{\partial\mathcal{C}_i}{\partial T}=\frac{1}{\left(1+\Pr_i\right)^2}\frac{\partial\Pr_i}{\partial T}F_i+\frac{\Pr_i}{1+\Pr_i}\frac{\partial F_i}{\partial T} \\
\frac{\partial\mathcal{C}_i}{\partial\mathfrak{X}_l}=\frac{1}{\left(1+\Pr_i\right)^2}\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}F_i+\frac{\Pr_i}{1+\Pr_i}\frac{\partial F_i}{\partial\mathfrak{X}_l}
$$

$\Pr_i=\frac{{k_0}_i}{{k_\infty}_i}\mathfrak{X}_i \Rightarrow \frac{\partial\Pr_i}{\partial T}=\frac{{k'_0}_i{k_\infty}_i-{k_0}_i{k'_\infty}_i}{{k_\infty^2}_i}\mathfrak{X}_i,\,\,\,
\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}=\frac{{k_0}_i}{{k_\infty}_i}\alpha_{il}$.

$\Pr_i=\frac{{k_0}_i}{{k_\infty}_i}\mathfrak{X}_m \Rightarrow
\frac{\partial\Pr_i}{\partial T}=\frac{{k'_0}_i{k_\infty}_i-{k_0}_i{k'_\infty}_i}{{k_\infty^2}_i}\mathfrak{X}_m,\,\,\,
\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}=\frac{{k_0}_i}{{k_\infty}_i}\delta_{lm}$, where $\delta_{lm}$ is Kroenecker delta symbol.

For Lindemann form $F_i=1 \Rightarrow \frac{\partial F_i}{\partial T}\equiv \frac{\partial F_i}{\partial\mathfrak{X}_l}\equiv 0$.

For Troe form

$$
\frac{\partial F_i}{\partial T}=\frac{\partial F_i}{\partial F_{cent}}\frac{\partial F_{cent}}{\partial T}+\frac{\partial F_i}{\partial \Pr_i}\frac{\partial \Pr_i}{\partial T},\\
\frac{\partial F_i}{\partial\mathfrak{X}_l}=\frac{\partial F_i}{\partial F_{cent}}\underbrace{\frac{\partial F_{cent}}{\partial\mathfrak{X}_l}}_{0}+\frac{\partial Fi}{\partial\Pr_i}\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}
=\frac{\partial F_i}{\partial\Pr_i}\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}\\
\frac{\partial F_i}{\partial F_{cent}}=\frac{F}{F_{cent}\left(1+\left(\frac{A}{B}\right)^2\right)}
-F\ln F_{cent}\left(\frac{2A}{B^3}\right)
\frac{A_FB-B_FA}{\left(1+\left(\frac{A}{B}\right)^2\right)^2}\\
\frac{\partial F_i}{\partial\Pr_i}=F\ln F_{cent}\left(\frac{2A}{B^3}\right)
\frac{A_{\Pr}B-B_{\Pr}A}{\left(1+\left(\frac{A}{B}\right)^2\right)^2}
$$

where

$$
A_F=\frac{\partial A}{\partial F_{cent}}=-\frac{0.67}{F_{cent}\ln 10},\,\,\,
B_F=\frac{\partial B}{\partial F_{cent}}=-\frac{1.1762}{F_{cent}\ln 10} \\
A_{\Pr}=\frac{\partial A}{\partial \Pr_i}=\frac{1}{\Pr_i \ln 10},\,\,\,
B_{\Pr}=\frac{\partial B}{\partial \Pr_i}=-\frac{0.14}{\Pr_i\ln 10} \\
\frac{\partial F_{cent}}{\partial T}=-\frac{1-a}{T^{***}}\exp\left(-\frac{T}{T^{***}}\right)
-\frac{a}{T^*}\exp\left(-\frac{T}{T^{*}}\right)+\frac{T^{**}}{T^2}\exp\left(-\frac{T^{**}}{T}\right)
$$

For SRI form

$$
\frac{\partial F_i}{\partial T} = F\Biggl(\frac{e}{T}+\frac{\partial X}{\partial\Pr_i}\frac{\partial\Pr_i}{\partial T}\ln\left(a\exp\left(-\frac{b}{T}\right)
+\exp\left(-\frac{T}{c}\right)\right)\Biggr.
+\Biggl.X\frac{\frac{ab}{T^2}\exp\left(-\frac{b}{T}\right)-\frac{1}{c}\exp\left(-\frac{T}{c}\right)}
{a\exp\left(-\frac{b}{T}\right)+\exp\left(-\frac{T}{c}\right)}\Biggr) \\
\frac{\partial F_i}{\partial\mathfrak{X}_l} = F\ln\left(a\exp\left(-\frac{b}{T}\right)+\exp\left(-\frac{T}{c}\right)\right)
\frac{\partial X}{\partial \Pr_i}\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}\\
\frac{\partial X}{\partial\Pr_i} =-X^2\frac{2\log_10 \Pr_i}{\Pr_i\ln 10}
$$

Chemically activated bimolecular reactions: $\mathcal{C}_i = \frac{1}{1+\Pr_i}F_i$

$$
\frac{\partial\mathcal{C}_i}{\partial T}=-\frac{1}{\left(1+\Pr_i\right)^2}\frac{\partial\Pr_i}{\partial T}F_i+\frac{1}{1+\Pr_i}\frac{\partial F_i}{\partial T} \\
\frac{\partial\mathcal{C}_i}{\partial\mathfrak{X}_l}=-\frac{1}{\left(1+\Pr_i\right)^2}\frac{\partial\Pr_i}{\partial\mathfrak{X}_l}F_i+\frac{1}{1+\Pr_i}\frac{\partial F_i}{\partial\mathfrak{X}_l}
$$

Partial derivatives of $\Pr_i$ and $F_i$ are computed similar to the ones above.

* Derivatives of $\mathcal{R}_i$

$$
\frac{\partial \mathcal{R}_i}{\partial T}={k'_f}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu'_{ji}}-
{k'_r}_i\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu''_{ji}} \\
\frac{\partial\mathcal{R}_i}{\partial\mathfrak{X}_l}=\frac{{k_f}_i\nu'_{li}\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu'_{ji}}}{\mathfrak{X}_l}
-\frac{{k_r}_i\nu''_{li}\prod_{j=1}^{N_{spec}}\mathfrak{X}_j^{\nu''_{ji}}}{\mathfrak{X}_l}
$$

${k_f}_i=A_iT^{\beta_i}\exp\left(-\frac{E_i}{R T}\right)
=A_i\exp\left(\beta_i\ln T-\frac{{T_a}_i}{T}\right)$, where ${T_a}_i=E_i/R$. The derivative with respect to temperature can be calculated as ${k'_f}_i=\frac{{k_f}_i}{T}\left(\beta_i+\frac{{T_a}_i}{T}\right)$

if reverse Arrhenius parameters are provided, ${k'_r}_i$ is computed similar to above. If ${k_r}_i$ is computed based on ${k_f}_i$ and the equilibrium constant ${K_c}_i$, then its derivative is computed as

$$
{k_r}_i=\frac{{k_f}_i}{{K_c}_i}\Rightarrow
{k'_r}_i=\frac{{k'_f}_i {K_c}_i-{k_f}_i {K_c}'_i}{{K_c}_i^2}=
\frac{\frac{{k_f}_i}{T}\left(\beta_i+\frac{{T_a}_i}{T}\right)}{{K_c}_i}
-\frac{{k_f}_i}{{K_c}_i}\frac{{K_c} '_i}{{K_c}_i}\\
={k_r}_i\left(\frac{1}{T}\left(\beta_i+\frac{{T_a}_i}{T}\right)-\frac{{K_c} '_i}{{K_c}_i}\right).
$$

Since
${K_c}_i=\left(\frac{p_{atm}}{\Re}\right)^{\sum_{k=1}^{N_{spec}}\nu_{ki}}
\exp\left(\sum_{k=1}^{N_{spec}}\nu_{ki}g_k\right)\Rightarrow
\frac{{K_c}'_i}{{K_c}_i}=\sum_{k=1}^{N_{spec}}\nu_{ki}g'_k$. It follows that

$$
{k'_r}_i = {k_r}_i\left(\frac{1}{T}\left(\beta_i+\frac{{T_a}_i}{T}\right)-\sum_{k=1}^{N_{spec}}\nu_{ki}g'_k\right)
$$

where $g'_k$ is computed based on NASA polynomial fits as

$$
g'_k=\frac{1}{T}\left(a_{0,k}-1+\frac{a_{5,k}}{T}\right)+\frac{a_{1,k}}{2}
+T\left(\frac{a_{2,k}}{3}+T\left(\frac{a_{3,k}}{4}+\frac{a_{4,k}}{5}T\right)\right)
$$

####  Efficient Evaluation of the $\tilde{J}$ Terms

* Step 1:

$$
\tilde{J}_{3+k,2}\equiv 0,\\
\tilde{J}_{3+k,3}=\frac{W_k}{\rho}\frac{\partial\dot{\omega}_k}{\partial T}=\frac{W_k}{\rho}\left[\sum_{j=1}^{N_{reac}}\nu_{kj}\frac{\partial\mathcal{C}_j}{\partial T}
\left({\mathcal{R}_f}_j-{\mathcal{R}_r}_j\right)+\sum_{j=1}^{N_{reac}}\nu_{kj}\mathcal{C}_j\left({\mathcal{R}_f}_j\frac{{k'_f}_j}{{k_f}_j}
-{\mathcal{R}_r}_j\frac{{k'_r}_j}{{k_r}_j}\right)\right],\\
\tilde{J}_{3+k,3+i}=\frac{W_k}{\rho}\frac{\partial\dot{\omega}_k}{\partial Y_i}=\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_i}
=\frac{W_k}{W_i}\left[\sum_{j=1}^{N_{reac}}\nu_{kj}\frac{\partial\mathcal{C}_k}{\partial\mathfrak{X}_i}
\left({\mathcal{R}_f}_j-{\mathcal{R}_r}_j\right)+\sum_{j=1}^{N_{reac}}\nu_{kj}\mathcal{C}_j
\frac{{\mathcal{R}_f}_j\nu'_{kj}-{\mathcal{R}_r}_j\nu''_{kj}}{\mathfrak{X}_i}\right],\\
i=1,2,\ldots,{N_{spec}}
$$

Here ${\mathcal{R_f}}_j$ and ${\mathcal{R}_r}_j$ are the forward and reverse parts, respectively of $\mathcal{R}_j$:

$$
{\mathcal{R}_f}_j={k_f}_j\prod_{i=1}^{N_{spec}}\mathfrak{X}_i^{\nu'_{ij}},\,\,\,
{\mathcal{R}_r}_j={k_r}_j\prod_{i=1}^{N_{spec}}\mathfrak{X}_i^{\nu''_{ij}}
$$

* Step 2: Once $\tilde{J}_{3+k,3+i}$ are evaluated for all $i$, then $\tilde{J}_{3+k,1}$ is computed as

$$
\tilde{J}_{3+k,1}=\frac{W_k}{\rho}\left(\frac{\partial\dot{\omega}_k}{\partial\rho}-\frac{\dot{\omega}_k}{\rho}\right)
=\frac{W_k}{\rho}\left(-\frac{\dot{\omega}_k}{\rho}+\sum_{i=1}^{N_{spec}}Y_i\frac{\partial\dot{\omega}_k}{\partial\mathfrak{X}_i}\right)
=\frac{1}{\rho}\left(-\frac{W_k\dot{\omega}_k}{\rho}+\sum_{i=1}^{N_{spec}}Y_i\tilde{J}_{3+k,3+i}\right)
$$

* Step 3:

$$
\tilde{J}_{3,1}=\frac{1}{\rho c_p}\sum_{i=1}^{N_{spec}} W_i h_i\left(\frac{\dot{\omega}_i}{\rho}-\frac{\partial\dot{\omega}_i}{\partial\rho}\right)
=-\frac{1}{c_p}\sum_{i=1}^{N_{spec}} h_i \tilde{J}_{3+i,1},\,\,\,
\tilde{J}_{3,2}\equiv 0 \\
\tilde{J}_{3,3}=\frac{1}{\rho c_p}\left[\frac{1}{c_p}\frac{\partial c_p}{\partial T}\sum_{i=1}^{N_{spec} } W_i h_i\dot{\omega}_i
-\sum_{i=1}^{N_{spec}} W_i {c_p}_i \dot{\omega}_i\right]-\frac{1}{\rho c_p}\sum_{i=1}^{N_{spec}} W_i h_i\frac{\partial\dot{\omega_i}}{\partial T} \\
=\frac{1}{\rho c_p}\left[\frac{1}{c_p}\frac{c_p}{T}\sum_{i=1}^{N_{spec}} W_i h_i\dot{\omega}_i
-\sum_{i=1}^{N_{spec}} W_i {c_p}_i \dot{\omega}_i\right]-\frac{1}{c_p}
\sum_{i=1}^{N_{spec}} h_i\tilde{J}_{3+i,3}
$$

### Evaluation of $J$ Components

* *Temperature equation*

$$
J_{1,1}=\tilde{J}_{3,3}+\tilde{J}_{3,1}\frac{\partial\rho}{\partial T},\,\,\,
J_{1,1+k}=\tilde{J}_{3,3+k}+\tilde{J}_{3,1}\frac{\partial \rho}{\partial Y_k}
$$

* *Species equations*

$$
J_{i,1}  =\tilde{J}_{1+i,3}+\tilde{J}_{i+1,1}\frac{\partial\rho}{\partial T}, \\
J_{i,1+k}=\tilde{J}_{i+1,3+k}+\tilde{J}_{i+1,1}\frac{\partial\rho}{\partial Y_k},\,\,\, k=1,2,\ldots,{N_{spec}}
$$

For $P=const$ density is a dependent variable, calculated based on
the ideal gas equation of state:

$$
\rho=\frac{P}{R T\sum_{k=1}^{N_{spec}}\frac{Y_k}{W_k}}
$$

The partial derivaties of density with respect to the independent variables are computed as

$$
\frac{\partial\rho}{\partial P} = \frac{\rho}{P},\,\,\, \frac{\partial\rho}{\partial T} =-\frac{\rho}{T},\,\,\,
\frac{\partial\rho}{\partial Y_k}=-\frac{\rho W}{W_k}.
$$

## Running the 0D Ignition Utility
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

@exec --inputsPath=@inputs --tol-time=@tol_time --atol-newton=@atol_newton --rtol-newton=@rtol_newton --dtmin=@dtmin --max-newton-iterations=@max_newton_iterations --output_frequency=@save --dtmax=@dtmax --tend=@tend --max-time-iterations=@max_time_iterations
````

In the above bash script the "inputs" variables is the path to where the inputs files are located in this case (\verb|TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0|). In this directory, the gas reaction mechanism is defined in "chem.inp" and the thermal properties in "therm.dat". Additionally, "sample.dat" contains the initial conditions for the simulation.

The parameters "dtmin" and "dtmax" control the size of the time steps in the solver. The decision on increase or decrease time step depends on the parameter "tol\_time". This parameter controls the error in each time iteration, thus, a bigger value will allow the solver to increase the time step while a smaller value will result in smaller time steps. The time-stepping will end when the time reaches "tend". The simulation will also end when the number of time steps reache  "max\_time\_iterations".  The absolute and relative tolerances in the Newton solver in each iteration are set with "atol\_newton" and "rtol\_newton", respectively, and the maximum number of Newton solver iterations is set with "max\_newton\_iterations".

The user can specify how often a solution is saved with the parameter "save". Thus, a solution will be saved at every iteration for this case. The default value of this input is $-1$, which means no output will be saved. The simulation results are saved in "IgnSolution.dat", with the following format:


````
iter     t       dt      Density[kg/m3]          Pressure[Pascal]        Temperature[K] SPECIES1 ... SPECIESN  
````  
where MF\_SPECIES1 respresents the mass fraction of species \#1, and so forth. Finally, we provide two methods to compute the ignition delay time. In the first approach, we save the time where the gas temperature reaches a threshold temperature. This temperature is set by default to $1500$K. In the second approach, save the location of the inflection point for the temperature profile as a function of time, also equivalent to the time when the second derivative of temperature with respect to time is zero. The result of these two methods are saved in files "IgnitionDelayTimeTthreshold.dat" and "IgnitionDelayTime.dat", respectively.



* **GRIMech 3.0 Results**
 The results presented below are obtained by running "TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDSA.x" with an initial temperature of $1000$K, pressure of $1$atm and a stoichiometric equivalence ratio ($\phi$) for methane/air mixtures. The input files are located at "TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/" and selected parameters were presented above. The outputs of the simulation were saved every iteration in "IgnSolution.dat". Time profiles for temperature and mass fractions for selected species are presented the following figures.



![Temperature and $CH_4$, $O_2$, $CO$ mass fraction](src/markdown/Figures/gri3.0_OneSample/TempMassFraction2.jpg)

![Temperature and $OH$, $H$, $H2$ mass fraction](src/markdown/Figures/gri3.0_OneSample/TempMassFraction3.jpg)

The ignition delay time values based on the two alternative computations discussed above are $1.100791$s and $1.100854$s, respectively. The scripts to setup and run this example and the jupyter-notebook used to create these figures can be found under "TCHEM_INSTALL_PATH/example/runs/gri3.0_IgnitionZeroD".

* **GRIMech 3.0 results parametric study**

The following figure shows the ignition delay time as a function of the initial temperature and equivalence ratio values. These results are based on settings provided in "TCHEM_INSTALL_PATH/example/runs/gri3.0_IgnDelay" and correspond to 100 samples. "TChem\_IgnitionZeroDSA.x" runs these samples in parallel. The wall-time is between $200-300s$ on a 3.1GHz Intel Core i7 cpu.

We also provide a jupyter-notebook to produce the sample file "sample.dat" and to generate the figure presented above.

![Ignition Delta time](src/markdown/Figures/gri3.0_IgnDelay/Gri3IgnDelayTime.jpg)
Figure. Ignition delay times [s] at P=1 atm for several CH$_4$/air equivalence ratio $\phi$ and initial temperature values. Results are based on the GRI-Mech v3.0 kinetic model.



## Ignition Delay Time Parameter Study for IsoOctane


We present a parameter study for several equivalence ratio, pressure, and initial temperature values for iso-Octane/air mixtures. The iso-Octane reaction mechanism used in this study consists of 874 species and 3796 elementary reactions~\cite{W. J. Pitz M. Mehl, H. J. Curran and C. K. Westbrook ref 5 }. We selected four pressure values, $\{10,16,34,45\}$ [atm]. For each case we ran a number of simulations that span a grid of $30$ initial conditions each for the equivalence ratio and temperature resulting in 900 samples for each pressure value. Each sample was run on a test bed with a Dual-Socket Intel Xeon Platinum architecture.


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
