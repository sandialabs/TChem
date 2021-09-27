/* =====================================================================================
TChem version 2.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the licese is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */
#ifndef __TCHEM_DRIVER__
#define __TCHEM_DRIVER__

#include "TChem.hpp"

namespace TChem {
  struct Driver {
  public:
    /// this policy is for time integration which can be repeatedly reused
    using policy_type = typename TChem::UseThisTeamPolicy<exec_space>::type;

  private:
    std::string _chem_file, _therm_file;
    
    TChem::KineticModelData _kmd;
    TChem::KineticModelGasConstData<interf_device_type> _kmcd_device;
    TChem::KineticModelGasConstData<interf_host_device_type> _kmcd_host;

    // model variation
    Kokkos::View<TChem::KineticModelData*, Kokkos::HostSpace> _kmds;
    Kokkos::View<TChem::KineticModelGasConstData<interf_device_type>*,interf_device_type> _kmcds_device;
    Kokkos::View<TChem::KineticModelGasConstData<interf_host_device_type>*,interf_host_device_type> _kmcds_host;

    bool _is_gasphase_kmcd_created;
    
    ordinal_type _n_sample;

    ///
    /// dual views are used for communication between host and device
    /// in particular, these dual views are used for pre/post processing
    /// otherwise, view should be used
    ///

    /// variables
    real_type_2d_dual_view _state;

    // Forward and reverse constants
    real_type_2d_dual_view _kforward;
    real_type_2d_dual_view _kreverse;
    
    // enthalpy mix
    real_type_2d_dual_view _enthalpy_mass;
    real_type_1d_dual_view _enthalpy_mix_mass;

    // gas phase net production rates
    real_type_2d_dual_view _net_production_rate_per_mass;

    //  homogeneous gas reactor
    real_type_3d_dual_view _jacobian_homogeneous_gas_reactor;
    real_type_2d_dual_view _rhs_homogeneous_gas_reactor;

    /// time integations
    time_advance_type_1d_view _tadv;
    real_type_2d_view _tol_time ;
    real_type_1d_view _tol_newton ;
    real_type_2d_view _fac ;

    /// t and dt are used for post processing
    real_type_1d_dual_view _t ;
    real_type_1d_dual_view _dt ;

    /// this is for time integration where the policy is reused
    policy_type _policy  ;

    void createTeamExecutionPolicy(const ordinal_type& per_team_extent);
    void createTimeAdvance(const ordinal_type& number_of_ODEs, const ordinal_type & number_of_equations );
    bool isTimeAdvanceCreated() const;
    void setTimeAdvance(const TChem::time_advance_type& tadv_default,
                        const real_type& atol_newton, const real_type&rtol_newton,
                        const real_type& atol_time, const real_type& rtol_time);
    void freeTimeAdvance();

  public:
    Driver();
    Driver(const std::string& chem_file, const std::string& therm_file);

    void freeAll();
    ~Driver();

    
    ///
    /// kinetic model interface; new kinetic model will free all views
    ///
    void createGasKineticModel(const std::string& chem_file, const std::string& therm_file);
    bool isGasKineticModelCreated() const;

    void cloneGasKineticModel();
    bool isGasKineticModelCloned() const;

    void modifyGasArrheniusForwardParameters(const ordinal_type_1d_view_host & reac_indices,
						  const real_type_3d_view_host & factors);
    void createGasKineticModelConstData();
    bool isGasKineticModelConstDataCreated() const;
    
    void freeGasKineticModel();

    ordinal_type getNumberOfSpecies() const;
    ordinal_type getNumberOfReactions() const;

    ///
    /// batch parallel; if a new sample size will free all views
    ///
    void setNumberOfSamples(const ordinal_type n_sample);
    ordinal_type getNumberOfSamples() const;

    ///
    /// state vector setup helper
    ///
    ordinal_type getLengthOfStateVector() const;
    ordinal_type getSpeciesIndex(const std::string& species_name) const;
    ordinal_type getStateVariableIndex(const std::string& var_name) const;

    ///
    /// arrhenius forward parameter accessor
    ///

    /// from the reference const object
    real_type getGasArrheniusForwardParameter(const ordinal_type reac_index,
					      const ordinal_type param_index);
    void getGasArrheniusForwardParameter(const ordinal_type_1d_view_host & reac_indices,
					 const ordinal_type param_index,
					 const real_type_1d_view_host & params);
    void getGasArrheniusForwardParameter(const ordinal_type_1d_view_host & reac_indices,
					 const real_type_2d_view_host & params);

    /// from the array object
    real_type getGasArrheniusForwardParameter(const ordinal_type imodel,
					      const ordinal_type reac_index,
					      const ordinal_type param_index);
    void getGasArrheniusForwardParameter(const ordinal_type imodel,
					 const ordinal_type_1d_view_host & reac_indices,
					 const ordinal_type param_index,
					 const real_type_1d_view_host & params);
    void getGasArrheniusForwardParameter(const ordinal_type imodel,
					 const ordinal_type_1d_view_host & reac_indices,
					 const real_type_2d_view_host & params);
    
    ///
    /// state vector
    ///
    bool isStateVectorCreated() const; /// why
    void createStateVector();
    void freeStateVector();
    void setStateVectorHost(const ordinal_type i, const real_type_1d_view_host& state_at_i);
    void setStateVectorHost(const real_type_2d_view_host& state);
    void getStateVectorHost(const ordinal_type i, real_type_1d_const_view_host& view);
    void getStateVectorHost(real_type_2d_const_view_host& view);
    void getStateVectorNonConstHost(const ordinal_type i, real_type_1d_view_host& view);
    void getStateVectorNonConstHost(real_type_2d_view_host& view);

    ///
    /// reaction rate constant
    ///
    bool isGasReactionRateConstantsCreated() const;
    void createGasReactionRateConstants();
    void freeGasReactionRateConstants();
    void getGasReactionRateConstantsHost(const ordinal_type i,
					      real_type_1d_const_view_host& view1,
					      real_type_1d_const_view_host& view2) ;
    void getGasReactionRateConstantsHost(real_type_2d_const_view_host& view1,
					      real_type_2d_const_view_host& view2);
    void computeGasReactionRateConstantsDevice();
    
    ///
    /// enthalphy mass
    ///
    bool isGasEnthapyMassCreated() const;
    void createGasEnthapyMass();
    void freeGasEnthapyMass();

    real_type getGasEnthapyMixMassHost(const ordinal_type i);
    void getGasEnthapyMixMassHost(real_type_1d_const_view_host& view);
    
    void getGasEnthapyMassHost(const ordinal_type i, real_type_1d_const_view_host& view);    
    void getGasEnthapyMassHost(real_type_2d_const_view_host& view);

    void computeGasEnthapyMassDevice() ;
    
    ///
    /// net production rate
    ///
    bool isGasNetProductionRatePerMassCreated() const;
    void createGasNetProductionRatePerMass();
    void freeGasNetProductionRatePerMass();
    void getGasNetProductionRatePerMassHost(const ordinal_type i, real_type_1d_const_view_host& view);
    void getGasNetProductionRatePerMassHost(real_type_2d_const_view_host& view);


    void computeGasNetProductionRatePerMassDevice();

    ///
    /// homogeneous gas reactor
    ///
    bool isJacobianHomogeneousGasReactorCreated() const;
    void createJacobianHomogeneousGasReactor();
    void freeJacobianHomogeneousGasReactor();
    void getJacobianHomogeneousGasReactorHost(const ordinal_type i, real_type_2d_const_view_host& view) ;
    void getJacobianHomogeneousGasReactorHost(real_type_3d_const_view_host& view);
    void computeJacobianHomogeneousGasReactorDevice() ;

    // RHS homogeneous gas reactor
    bool isRHS_HomogeneousGasReactorCreated() const;
    void createRHS_HomogeneousGasReactor();
    void freeRHS_HomogeneousGasReactor();
    void getRHS_HomogeneousGasReactorHost(const ordinal_type i, real_type_1d_const_view_host& view) ;
    void getRHS_HomogeneousGasReactorHost(real_type_2d_const_view_host& view);
    void computeRHS_HomogeneousGasReactorDevice() ;


    ///
    /// time integration
    ///
    void setTimeAdvanceHomogeneousGasReactor(const real_type & tbeg,
					     const real_type & tend,
					     const real_type & dtmin,
					     const real_type & dtmax,
                                             const ordinal_type & jacobian_interval,
					     const ordinal_type & max_num_newton_iterations,
					     const ordinal_type & num_time_iterations_per_interval,
					     const real_type& atol_newton, const real_type&rtol_newton,
					     const real_type& atol_time, const real_type& rtol_time);
    real_type computeTimeAdvanceHomogeneousGasReactorDevice();

    /// t and dt accessor
    ///
    void getTimeStepHost(real_type_1d_const_view_host& view);
    void getTimeStepSizeHost(real_type_1d_const_view_host& view);

    void createAllViews();
    void freeAllViews();
    void showViews(const std::string& label);
  };
}

/// C and Fortran interface
#ifdef __cplusplus
extern "C" {
#endif
  void TChem_createGasKineticModel(const char * chem_file, const char * therm_file);
  bool TChem_isKineticModelCreated();
  void TChem_createGasKineticModelConstData();
  bool TChem_isGasKineticModelConstDataCreated();
  void TChem_freeGasKineticModel();

  int  TChem_getNumberOfSpecies();
  int  TChem_getNumberOfReactions();

  void TChem_setNumberOfSamples(const int n_sample);
  int  TChem_getNumberOfSamples();

  int  TChem_getLengthOfStateVector();
  int  TChem_getSpeciesIndex(const char * species_name);
  int  TChem_getStateVariableIndex(const char * var_name);

  bool TChem_isStateVectorCreated();
  void TChem_createStateVector();
  void TChem_freeStateVector();
  void TChem_setSingleStateVectorHost(const int i, real_type * state_at_i);
  void TChem_setAllStateVectorHost(real_type * state);
  void TChem_getSingleStateVectorHost(const int i, real_type * view);
  void TChem_getAllStateVectorHost(real_type * view);

  bool TChem_isGasNetProductionRatePerMassCreated();
  void TChem_createGasNetProductionRatePerMass();
  void TChem_freeGasNetProductionRatePerMass();
  void TChem_getSingleGasNetProductionRatePerMassHost(const int i, real_type * view);
  void TChem_getAllGasNetProductionRatePerMassHost(real_type * view);

  void TChem_computeGasNetProductionRatePerMassDevice();

  void TChem_setTimeAdvanceHomogeneousGasReactor(const real_type tbeg,
						 const real_type tend,
						 const real_type dtmin,
						 const real_type dtmax,
                                                 const int jacobian_interval,
						 const int max_num_newton_iterations,
						 const int num_time_iterations_per_interval,
						 const real_type atol_newton, const real_type rtol_newton,
						 const real_type atol_time, const real_type rtol_time);
  real_type TChem_computeTimeAdvanceHomogeneousGasReactorDevice();

  void TChem_getTimeStepHost(real_type * view);
  void TChem_getTimeStepSizeHost(real_type * view);

  void TChem_createAllViews();
  void TChem_showAllViews(const char * label);
  void TChem_freeAllViews();

#ifdef __cplusplus
}
#endif

#endif
