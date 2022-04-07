Nomenclature
============

In the table below, $ro$ stands for reaction order, for the forward and reverse paths, respectively.

============================ ======================================================== =========================================================
Symbol                       Description                                              Units
============================ ======================================================== =========================================================
 :math:`N_{spec}`            number of species                                        :math:`-`          
 :math:`N_{reac}`            number of reactions                                      :math:`-`          
 :math:`N_{spec}^g`          number of gas-phase species                              :math:`-`          
 :math:`N_{spec}^s`          number of surface species                                :math:`-`          
 :math:`N_{spec}^{s,n}`      number of surface species in phase *n*                   :math:`-`          
 :math:`\rho`                gas-phase density                                        :math:`kg/m^3`     
 :math:`P`                   thermodynamic pressure                                   :math:`Pa`         
 :math:`T`                   temperature                                              :math:`K`          
 :math:`C_p`                 heat capacity at constant pressure                       :math:`J/(K.kmol)` 
 :math:`C_{p,k}`             heat capacity for species *k*                            :math:`J/(K.kmol)` 
 :math:`c_{p}`               specific heat capacity                                   :math:`J/(K.kg)`   
 :math:`c_{p,k}`             specific heat capacity for species *k*                   :math:`J/(K.kg)`   
 :math:`H`                   molar enthalpy of a mixture                              :math:`J/kmol`     
 :math:`S`                   molar entropy of a mixture                               :math:`J/(kmol.K)` 
 :math:`Y_k`                 mass fraction of species *k*                             :math:`-`          
 :math:`X_k`                 mole fraction of species *k*                             :math:`-`          
 :math:`H_{k}`               molar enthalpy of *k* species                            :math:`J/kmol`
 :math:`h_{p}`               specific molar enthalpy                                  :math:`J/kg`
 :math:`h_{p,k}`             specific molar enthalpy for species *k*                  :math:`J/kg`
 :math:`S_{k}`               molar entropy of *k* species                             :math:`J/(kmol.K)`
 :math:`s`                   specific molar entropy                                   :math:`J/(K.kg)`
 :math:`s_{k}`               specific molar entropy for species *k*                   :math:`J/(K.kg)`
 :math:`G_{k}`               Gibbs free energy of *k* species                         :math:`J/kmol`
 :math:`g`                   specific Gibbs free energy                               :math:`J/kg`
 :math:`g_{k}`               specific Gibbs free energy for species *k*               :math:`J/kg`
 :math:`\mathfrak{X}_k`      molar concentration of species *k*                       :math:`kmol/m^3`
 :math:`Z_k`                 site fraction of species *k*                             :math:`-`
 :math:`Z_k^{(n)}`           site fraction of species *k* in phase *n*                :math:`-`
 :math:`\Gamma_n`            surface site density of phase *n*                        :math:`kmol/m^2`
 :math:`\sigma_{k}(n)`       site occupancy by species *k* in phase *n*               :math:`-`
 :math:`W`                   mixture molecular weight                                 :math:`kg/kmol`
 :math:`W_{k}`               for species *k*                                          :math:`kg/kmol`
 :math:`R`                   universal gas constant                                   :math:`J/(kmol.K)`
 :math:`k_{fi}`              Forward rate constant of *i* reaction                    :math:`\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}`
 :math:`k_{ri}`              Reverse rate constant of *i* reaction                    :math:`\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}`
 :math:`R`                   Universal gas constant                                   :math:`J/(kmol.K)`
 :math:`\dot{q}_{i}`         Rate of progress of *i* reaction                         :math:`kmol/(m^3.s)`
 :math:`\gamma_{i}`          sticking coefficient for reaction *i*                    :math:`\frac{(\textrm{kmol/m}^3)^{(1-ro)}}{\textrm{s}}`
 :math:`\dot{\omega}_{k}`    Production rate of *k* species                           :math:`kmol/(m^3.s)`
 :math:`\dot{s}_{k}`         surface molar production rate of species *k*             :math:`kmol/(m^2.s)`
============================ ======================================================== =========================================================

.. autosummary::
   :toctree: generated

   tchem
