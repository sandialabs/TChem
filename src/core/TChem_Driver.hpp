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

    bool _kmd_created;
    std::string _chem_file, _therm_file;
    TChem::KineticModelData _kmd;
    TChem::KineticModelConstData<interf_device_type> _kmcd_device;
    TChem::KineticModelConstData<interf_host_device_type> _kmcd_host;

    ordinal_type _n_sample;

    ///
    /// dual views are used for communication between host and device
    /// in particular, these dual views are used for pre/post processing
    /// otherwise, view should be used
    ///

    /// variables
    real_type_2d_dual_view _state;

    real_type_2d_dual_view _net_production_rate_per_mass;
    //  jacobian homogeneous gas reactor
    real_type_3d_dual_view _jacobian_homogeneous_gas_reactor;

    // rhs homogeneous gas reactor
    real_type_2d_dual_view _rhs_homogeneous_gas_reactor;

    // enthalpy mix
    real_type_2d_dual_view _enthalpy_mass;
    real_type_1d_dual_view _enthalpy_mix_mass;

    // Forward and reverse constants
    real_type_2d_dual_view _kforward;
    real_type_2d_dual_view _kreverse;
    policy_type _policy_KForRev;

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
    policy_type _policy_enthalpy;

    /// with reactor set function, this flag is on and time advance is ready
    ordinal_type _is_time_advance_set;

    void createTeamExecutionPolicy(const ordinal_type& per_team_extent);
    void createTimeAdvance(const ordinal_type& number_of_ODEs, const ordinal_type & number_of_equations );
    void setTimeAdvance(const TChem::time_advance_type& tadv_default,
                        const real_type& atol_newton, const real_type&rtol_newton,
                        const real_type& atol_time, const real_type& rtol_time);
    void freeTimeAdvance();

  public:
    Driver();
    Driver(const std::string& chem_file, const std::string& therm_file);
    ~Driver();

    ///
    /// kinetic model interface; new kinetic model will free all views
    ///
    void createKineticModel(const std::string& chem_file, const std::string& therm_file);
    void freeKineticModel();
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
    /// net production rate
    ///
    bool isNetProductionRatePerMassCreated() const;
    void createNetProductionRatePerMass();
    void freeNetProductionRatePerMass();
    void getNetProductionRatePerMassHost(const ordinal_type i, real_type_1d_const_view_host& view);
    void getNetProductionRatePerMassHost(real_type_2d_const_view_host& view);


    void computeNetProductionRatePerMassDevice();

    // Jacobina homogeneous gas reactor
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

    // k forward and reverse
    bool isReactionRateConstantsCreated() const;
    void createReactionRateConstants();
    void freeReactionRateConstants();
    void getReactionRateConstantsHost(const ordinal_type i,
                                      real_type_1d_const_view_host& view1,
                                      real_type_1d_const_view_host& view2) ;
    void getReactionRateConstantsHost(real_type_2d_const_view_host& view1,
                                      real_type_2d_const_view_host& view2);
    void computeReactionRateConstantsDevice() ;

    // enthalpy mix
    bool isEnthapyMassCreated() const;
    void createEnthapyMass();
    void freeEnthapyMass();
    // void getEnthapyMassHost(const ordinal_type i, real_type_1d_const_view_host& view) ;
    void getEnthapyMassHost(real_type_2d_const_view_host& view);
    void getEnthapyMixMassHost(real_type_1d_const_view_host& view);
    void computeEnthapyMassDevice() ;

    ///
    /// time integration
    ///
    void unsetTimeAdvance();
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

    ///
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
  void TChem_createKineticModel(const char * chem_file, const char * therm_file);
  bool TChem_isKineticModelCreated();
  void TChem_freeKineticModel();
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

  bool TChem_isNetProductionRatePerMassCreated();
  void TChem_createNetProductionRatePerMass();
  void TChem_freeNetProductionRatePerMass();
  void TChem_getSingleNetProductionRatePerMassHost(const int i, real_type * view);
  void TChem_getAllNetProductionRatePerMassHost(real_type * view);

  void TChem_computeNetProductionRatePerMassDevice();

  void TChem_unsetTimeAdvance();
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
