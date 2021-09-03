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
#include "TChem_Driver.hpp"

namespace TChem {
  void
  Driver::
  createTeamExecutionPolicy(const ordinal_type& per_team_extent)
  {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");

    const auto exec_space_instance = exec_space();
    _policy = policy_type(exec_space_instance, _n_sample, Kokkos::AUTO());

    const ordinal_type level = 1;
    const ordinal_type per_team_scratch =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    _policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  }

  void
  Driver::
  createTimeAdvance(const ordinal_type& number_of_ODEs, const ordinal_type & number_of_equations ){

    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");

    _tadv = time_advance_type_1d_view("time advance", _n_sample);
    _tol_time = real_type_2d_view( "tol time", number_of_ODEs, 2);
    _tol_newton = real_type_1d_view("tol newton", 2);
    _fac = real_type_2d_view ("fac", _n_sample, number_of_equations);

    _t = real_type_1d_dual_view("time", _n_sample);

    _dt = real_type_1d_dual_view("delta time", _n_sample);
  }

  void
  Driver::
  setTimeAdvance(const TChem::time_advance_type& tadv_default,
                 const real_type& atol_newton, const real_type&rtol_newton,
                 const real_type& atol_time, const real_type& rtol_time) {
    TCHEM_CHECK_ERROR(isTimeAdvanceCreated(), "Time adance first needs to be created; use createTimeAdvance()");

    Kokkos::deep_copy(_tadv, tadv_default);
    Kokkos::deep_copy(_t.view_device(), tadv_default._tbeg);     _t.modify_device();
    Kokkos::deep_copy(_dt.view_device(), tadv_default._dtmin);    _dt.modify_device();

    {
      auto tol_time = Kokkos::create_mirror_view(Kokkos::HostSpace(), _tol_time);
      auto tol_newton = Kokkos::create_mirror_view(Kokkos::HostSpace(), _tol_newton);
      Kokkos::parallel_for
	(Kokkos::RangePolicy<host_exec_space>(0, _tol_time.extent(0)),
	 [=](const ordinal_type &i) {
	   tol_time(i,0) = atol_time;
	   tol_time(i,1) = rtol_time;
	   if (i == 0) {
	     tol_newton(0) = atol_newton;
	     tol_newton(1) = rtol_newton;
	   }
	 });
      Kokkos::deep_copy(_tol_time, tol_time);
      Kokkos::deep_copy(_tol_newton, tol_newton);
    }
  }

  void
  Driver::
  freeTimeAdvance() {
    _tadv = time_advance_type_1d_view();
    _tol_time = real_type_2d_view();
    _tol_newton = real_type_1d_view();
    _fac = real_type_2d_view();
    _dt = real_type_1d_dual_view();
    _t = real_type_1d_dual_view();
  }

  Driver::
  Driver() :
    _kmd_created(false),
    _chem_file(),
    _therm_file(),
    _kmd(),
    _kmcd_device(),
    _kmcd_host(),
    _n_sample(0),
    _state(),
    _net_production_rate_per_mass(),
    _tadv(),
    _t(),
    _dt(),
    _tol_time(),
    _tol_newton(),
    _fac(),
    _policy(),
    _is_time_advance_set(),
    //  jacobian homogeneous gas reactor
    _jacobian_homogeneous_gas_reactor(),
    // rhs homogeneous gas reactor
    _rhs_homogeneous_gas_reactor(),
    // enthalpy mix
    _enthalpy_mass(),
    _enthalpy_mix_mass(),
    _policy_enthalpy(),
    // k forward and reverse
    _kforward(),
    _kreverse(),
    _policy_KForRev()
    {
    }

  Driver::
  Driver(const std::string& chem_file, const std::string& therm_file) :
    _kmd_created(),
    _chem_file(chem_file),
    _therm_file(therm_file),
    _kmd(),
    _kmcd_device(),
    _kmcd_host(),
    _n_sample(0),
    _state(),
    _net_production_rate_per_mass(),
    _tadv(),
    _t(),
    _dt(),
    _tol_time(),
    _tol_newton(),
    _fac(),
    _policy(),
    _is_time_advance_set(),
    //  jacobian homogeneous gas reactor
    _jacobian_homogeneous_gas_reactor(),
    // rhs homogeneous gas reactor
    _rhs_homogeneous_gas_reactor(),
    // enthalpy mix
    _enthalpy_mass(),
    _enthalpy_mix_mass(),
    _policy_enthalpy(),
    // k forward and reverse
    _kforward(),
    _kreverse(),
    _policy_KForRev()
    {
      createKineticModel(chem_file, therm_file);
    }

  Driver::
  ~Driver() {
    freeKineticModel();
    freeAllViews();
    freeTimeAdvance();
  }

  void
  Driver::
  createKineticModel(const std::string& chem_file, const std::string& therm_file) {
    _chem_file = chem_file;
    _therm_file = therm_file;

    _kmd = KineticModelData(_chem_file, _therm_file);
    _kmcd_device = _kmd.createConstData<interf_device_type>();
    _kmcd_host = _kmd.createConstData<interf_host_device_type>();

    _kmd_created = true;

    freeAllViews();
    freeTimeAdvance();

    _is_time_advance_set = false;
  }

  void
  Driver::
  freeKineticModel() {
    _chem_file = std::string();
    _therm_file = std::string();

    _kmd = KineticModelData();
    _kmcd_device = KineticModelConstData<interf_device_type>();
    _kmcd_host = KineticModelConstData<interf_host_device_type>();

    _kmd_created = false;

    freeAllViews();
    freeTimeAdvance();

    _is_time_advance_set = false;
  }

  ordinal_type
  Driver::
  getNumberOfSpecies() const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    return _kmcd_host.nSpec;
  }

  ordinal_type
  Driver::
  getNumberOfReactions() const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    return _kmcd_host.nReac;
  }


  void
  Driver::
  setNumberOfSamples(const ordinal_type n_sample) {
    /// when a new samples are set, it needs to reset all
    if (_n_sample == n_sample) {
      /// do nothing
    } else {
      _n_sample = n_sample;
      freeAllViews();
    }
  }

  ordinal_type
  Driver::
  getNumberOfSamples() const {
    return _n_sample;
  }

  ordinal_type
  Driver::
  getLengthOfStateVector() const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    return Impl::getStateVectorSize(_kmcd_device.nSpec);
  }

  ordinal_type
  Driver::
  getSpeciesIndex(const std::string& species_name) const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    for (ordinal_type i = 0; i < _kmcd_host.nSpec; i++)
      if (strncmp(&_kmcd_host.speciesNames(i, 0),
                  (species_name).c_str(),
                  LENGTHOFSPECNAME) == 0) {
        return i;
    }
    return -1;
  }

  ordinal_type
  Driver::
  getStateVariableIndex(const std::string& var_name) const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");

  if ((var_name == "Temperature") || (var_name == "T") || (var_name == "Temp")   )
  {
    return 2;
  } else if ((var_name == "Density") || (var_name == "D") )
  {
    return 0;
  } else if ((var_name == "Pressure") || (var_name == "P")  )
  {
    return 1;
  }
  return getSpeciesIndex(var_name) + 3;
  }

  bool
  Driver::
  isStateVectorCreated() const {
    return (_state.span() > 0);
  }

  void
  Driver::
  createStateVector() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const ordinal_type len = Impl::getStateVectorSize(_kmcd_device.nSpec);
    _state = real_type_2d_dual_view("state dev", _n_sample, len);
  }

  void
  Driver::
  freeStateVector() {
    _state = real_type_2d_dual_view();
  }

  void
  Driver::
  setStateVectorHost(const ordinal_type i, const real_type_1d_view_host& state_at_i) {
    if (!isStateVectorCreated()) {
      createStateVector();
    }
    const auto src = state_at_i;
    const auto tgt = Kokkos::subview(_state.view_host(), i, Kokkos::ALL());
    Kokkos::deep_copy(tgt, src);
    _state.modify_host();
  }

  void
  Driver::
  setStateVectorHost(const real_type_2d_view_host& state) {
    if (!_state.span()) {
      createStateVector();
    }
    using range_type = Kokkos::pair<ordinal_type,ordinal_type>;
    const auto src = state;
    const auto tgt = Kokkos::subview(_state.view_host(), range_type(0,src.extent(0)), range_type(0,src.extent(1)));
    Kokkos::deep_copy(tgt, src);
    _state.modify_host();
  }

  void
  Driver::
  getStateVectorHost(const ordinal_type i, real_type_1d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_state.span() == 0, "State vector should be constructed");
    _state.sync_host();
    auto hv = _state.view_host();
    view = real_type_1d_const_view_host(&hv(i,0), hv.extent(1));
  }

  void
  Driver::
  getStateVectorHost(real_type_2d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_state.span() == 0, "State vector should be constructed");
    _state.sync_host();
    auto hv = _state.view_host();
    view = real_type_2d_const_view_host(&hv(0,0), hv.extent(0), hv.extent(1));
  }

  void
  Driver::
  getStateVectorNonConstHost(const ordinal_type i, real_type_1d_view_host& view) {
    TCHEM_CHECK_ERROR(_state.span() == 0, "State vector should be constructed");
    _state.sync_host();
    auto hv = _state.view_host();
    view = real_type_1d_view_host(&hv(i,0), hv.extent(1));
    _state.modify_host();
  }

  void
  Driver::
  getStateVectorNonConstHost(real_type_2d_view_host& view) {
    TCHEM_CHECK_ERROR(_state.span() == 0, "State vector should be constructed");
    _state.sync_host();
    auto hv = _state.view_host();    
    view = real_type_2d_view_host(&hv(0,0), hv.extent(0), hv.extent(1));
    _state.modify_host();
  }

  bool
  Driver::
  isNetProductionRatePerMassCreated() const {
    return (_net_production_rate_per_mass.span() > 0);
  }

    void
  Driver::
  createNetProductionRatePerMass() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const ordinal_type len = _kmcd_device.nSpec;
    _net_production_rate_per_mass = real_type_2d_dual_view("net_production_rate_per_mass dev", _n_sample, len);
  }


  void
  Driver::
  freeNetProductionRatePerMass() {
    _net_production_rate_per_mass = real_type_2d_dual_view();
  }

  void
  Driver::
  getNetProductionRatePerMassHost(const ordinal_type i, real_type_1d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_net_production_rate_per_mass.span() == 0, "State vector should be constructed");
    _net_production_rate_per_mass.sync_host();
    auto hv = _net_production_rate_per_mass.view_host();
    view = real_type_1d_const_view_host(&hv(i,0), hv.extent(1));
  }

  void
  Driver::
  getNetProductionRatePerMassHost(real_type_2d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_net_production_rate_per_mass.span() == 0, "State vector should be constructed");
    _net_production_rate_per_mass.sync_host();
    auto hv = _net_production_rate_per_mass.view_host();
    view = real_type_2d_const_view_host(&hv(0,0), hv.extent(0), hv.extent(1));
  }

  void
  Driver::
  computeNetProductionRatePerMassDevice() {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");
    _state.sync_device();
    
    if (_net_production_rate_per_mass.span() == 0) {
      createNetProductionRatePerMass();
    }
    NetProductionRatePerMass::runDeviceBatch
      (_n_sample, _state.view_device(), _net_production_rate_per_mass.view_device(), _kmcd_device);
    _net_production_rate_per_mass.modify_device();
  }

  bool
  Driver::
  isJacobianHomogeneousGasReactorCreated() const {
    return (_jacobian_homogeneous_gas_reactor.span() > 0);
  }

  void
  Driver::
  createJacobianHomogeneousGasReactor() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const ordinal_type len = _kmcd_device.nSpec + 1;

    _jacobian_homogeneous_gas_reactor = real_type_3d_dual_view("jacobian homogeneous gas reactor dev", _n_sample, len, len);
  }

  void
  Driver::
  freeJacobianHomogeneousGasReactor() {
    _jacobian_homogeneous_gas_reactor = real_type_3d_dual_view();
  }
  
  void
  Driver::
  getJacobianHomogeneousGasReactorHost(const ordinal_type i, real_type_2d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_jacobian_homogeneous_gas_reactor.span() == 0, "Jacobian of homogeneous gas reactor should be constructed");
    _jacobian_homogeneous_gas_reactor.sync_host();
    auto hv = _jacobian_homogeneous_gas_reactor.view_host();
    view = real_type_2d_const_view_host(&hv(i,0,0), hv.extent(1), hv.extent(2));
  }

  void
  Driver::
  getJacobianHomogeneousGasReactorHost(real_type_3d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_jacobian_homogeneous_gas_reactor.span() == 0, "State vector should be constructed");
    _jacobian_homogeneous_gas_reactor.sync_host();
    auto hv = _jacobian_homogeneous_gas_reactor.view_host();
    view = real_type_3d_const_view_host(&hv(0,0,0), hv.extent(0), hv.extent(1), hv.extent(2));
  }

  void
  Driver::
  computeJacobianHomogeneousGasReactorDevice() {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");
    _state.sync_device();
    
    if (_jacobian_homogeneous_gas_reactor.span() == 0) {
      createJacobianHomogeneousGasReactor();
    }

    real_type_2d_view workspace;
    JacobianReduced::runDeviceBatch
      ( _state.view_device(), _jacobian_homogeneous_gas_reactor.view_device(), workspace, _kmcd_device);
    _jacobian_homogeneous_gas_reactor.modify_device();
  }

  bool
  Driver::
  isRHS_HomogeneousGasReactorCreated() const {
    return (_rhs_homogeneous_gas_reactor.span() > 0);
  }

  void
  Driver::
  createRHS_HomogeneousGasReactor() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const ordinal_type len = _kmcd_device.nSpec + 1;
    _rhs_homogeneous_gas_reactor = real_type_2d_dual_view("jacobian homogeneous gas reactor dev", _n_sample, len);
  }

  void
  Driver::
  freeRHS_HomogeneousGasReactor() {
    _rhs_homogeneous_gas_reactor = real_type_2d_dual_view();
  }

  void
  Driver::
  getRHS_HomogeneousGasReactorHost(const ordinal_type i, real_type_1d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_rhs_homogeneous_gas_reactor.span() == 0, "RHS of homogeneous gas reactor should be constructed");
    _rhs_homogeneous_gas_reactor.sync_host();
    auto hv = _rhs_homogeneous_gas_reactor.view_host();
    view = real_type_1d_const_view_host(&hv(i,0), hv.extent(1));
  }

  void
  Driver::
  getRHS_HomogeneousGasReactorHost(real_type_2d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_rhs_homogeneous_gas_reactor.span() == 0, "State vector should be constructed");
    _rhs_homogeneous_gas_reactor.sync_host();
    auto hv = _rhs_homogeneous_gas_reactor.view_host();
    view = real_type_2d_const_view_host(&hv(0,0), hv.extent(0), hv.extent(1));
  }

  void
  Driver::
  computeRHS_HomogeneousGasReactorDevice() {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");
    _state.sync_device();

    if (_rhs_homogeneous_gas_reactor.span() == 0) {
      createRHS_HomogeneousGasReactor();
    }

    real_type_2d_view workspace;
    SourceTerm::runDeviceBatch(_state.view_device(), _rhs_homogeneous_gas_reactor.view_device(), workspace, _kmcd_device);
  }

  bool
  Driver::
  isReactionRateConstantsCreated() const {
    return (_kforward.span() > 0);
  }

  void
  Driver::
  createReactionRateConstants() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const ordinal_type len = _kmcd_device.nReac;
    _kforward = real_type_2d_dual_view("Forward rate constant dev", _n_sample, len);
    _kreverse = real_type_2d_dual_view("reverse rate constant dev", _n_sample, len);

    // KK: Why do we create policy here ?
    // create policy for enthalpy
    const auto exec_space_instance = exec_space();
    _policy_KForRev = policy_type(exec_space_instance, _n_sample, Kokkos::AUTO());
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = TChem::KForwardReverse::getWorkSpaceSize(_kmcd_device);
    const ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    _policy_KForRev.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  }

  void
  Driver::
  freeReactionRateConstants() {
    _kforward = real_type_2d_dual_view();
    _kreverse = real_type_2d_dual_view();
  }

  /// KK: why do we name view1 view2 instead of forward reverse names ? 
  void
  Driver::
  getReactionRateConstantsHost(const ordinal_type i, 
                               real_type_1d_const_view_host& view1,
                               real_type_1d_const_view_host& view2) {
    TCHEM_CHECK_ERROR(_kforward.span() == 0, "Forward rate constant should be constructed");
    _kforward.sync_host();
    {
      auto hv = _kforward.view_host();
      view1 = real_type_1d_const_view_host(&hv(i,0), hv.extent(1));
    }
    TCHEM_CHECK_ERROR(_kreverse.span() == 0, "Reverse rate constant should be constructed");
    _kreverse.sync_host();
    {
      auto hv = _kreverse.view_host();
      view2 = real_type_1d_const_view_host(&hv(i,0), hv.extent(1));
    }
  }

  void
  Driver::
  getReactionRateConstantsHost(real_type_2d_const_view_host& view1,
                               real_type_2d_const_view_host& view2) {
    TCHEM_CHECK_ERROR(_kforward.span() == 0, "Forward rate constant should be constructed");
    _kforward.sync_host();
    {
      auto hv = _kforward.view_host();
      view1 = real_type_2d_const_view_host(&hv(0,0), hv.extent(0), hv.extent(1));
    }
    TCHEM_CHECK_ERROR(_kreverse.span() == 0, "Reverse rate constant should be constructed");
    _kreverse.sync_host();
    {
      auto hv = _kreverse.view_host();
      view2 = real_type_2d_const_view_host(&hv(0,0), hv.extent(0), hv.extent(1));
    }
  }

  void
  Driver::
  computeReactionRateConstantsDevice() {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");
    _state.sync_device();

    if (_kforward.span() == 0 || _kreverse.span() == 0) {
      createReactionRateConstants();
    }
    
    KForwardReverse::runDeviceBatch( _policy_KForRev,
                                     _state.view_device(),
                                     _kforward.view_device(),
                                     _kreverse.view_device(),
                                     _kmcd_device);
    _kforward.modify_device();
    _kreverse.modify_device();
  }

  // enthalpy mass
  bool
  Driver::
  isEnthapyMassCreated() const {
    return (_enthalpy_mass.span() > 0);
  }

  void
  Driver::
  createEnthapyMass() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const ordinal_type len = _kmcd_device.nSpec;
    _enthalpy_mass = real_type_2d_dual_view("jacobian homogeneous gas reactor dev", _n_sample, len);
    _enthalpy_mix_mass = real_type_1d_dual_view("jacobian homogeneous gas reactor dev", _n_sample);

    // create policy for enthalpy
    const auto exec_space_instance = exec_space();
    _policy_enthalpy = policy_type(exec_space_instance, _n_sample, Kokkos::AUTO());
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = TChem::EnthalpyMass::getWorkSpaceSize(_kmcd_device);
    const ordinal_type per_team_scratch =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    _policy_enthalpy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
  }

  void
  Driver::
  freeEnthapyMass() {
    _enthalpy_mass = real_type_2d_dual_view();
    _enthalpy_mix_mass = real_type_1d_dual_view();
  }

  void
  Driver::
  getEnthapyMixMassHost(real_type_1d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_enthalpy_mix_mass.span() == 0, "enthalpy-mix view should be constructed");
    _enthalpy_mix_mass.sync_host();
    auto hv = _enthalpy_mix_mass.view_host();
    view = real_type_1d_const_view_host(&hv(0), hv.extent(0));
  }

  void
  Driver::
  getEnthapyMassHost(real_type_2d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_enthalpy_mass.span() == 0, "enthalpy view should be constructed");
    _enthalpy_mass.sync_host();
    auto hv = _enthalpy_mass.view_host();
    view = real_type_2d_const_view_host(&hv(0,0), hv.extent(0), hv.extent(1));
  }

  void
  Driver::
  computeEnthapyMassDevice() {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");
    _state.sync_device();

    if (_enthalpy_mass.span() == 0) {
      createEnthapyMass();
    }

    TChem::EnthalpyMass::runDeviceBatch(_policy_enthalpy,
                                        _state.view_device(),
                                        _enthalpy_mass.view_device(),
                                        _enthalpy_mix_mass.view_device(),
                                        _kmcd_device);
  }

  void
  Driver::
  unsetTimeAdvance() {
    _is_time_advance_set = false;
  }
  
  void
  Driver::
  setTimeAdvanceHomogeneousGasReactor(const real_type & tbeg,
				      const real_type & tend,
				      const real_type & dtmin,
				      const real_type & dtmax,
                                      const ordinal_type & jacobian_interval,
				      const ordinal_type & max_num_newton_iterations,
				      const ordinal_type & num_time_iterations_per_interval,
				      const real_type& atol_newton, const real_type&rtol_newton,
				      const real_type& atol_time, const real_type& rtol_time) {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");
    const ordinal_type worksize = TChem::IgnitionZeroD::getWorkSpaceSize(_kmcd_device);
    createTeamExecutionPolicy(worksize);

    using problem_type = TChem::Impl::IgnitionZeroD_Problem<real_type, interf_device_type>;
    createTimeAdvance(problem_type::getNumberOfTimeODEs(_kmcd_device),
		      problem_type::getNumberOfEquations(_kmcd_device));
    
    TChem::time_advance_type tadv_default;
    tadv_default._tbeg = tbeg;
    tadv_default._tend = tend;
    tadv_default._dt = dtmin;
    tadv_default._dtmin = dtmin;
    tadv_default._dtmax = dtmax;
    tadv_default._max_num_newton_iterations = max_num_newton_iterations;
    tadv_default._num_time_iterations_per_interval = num_time_iterations_per_interval;
    tadv_default._jacobian_interval = jacobian_interval;
    setTimeAdvance(tadv_default, atol_newton, rtol_newton, atol_time, rtol_time);

    _is_time_advance_set = true;
  }

  real_type
  Driver::
  computeTimeAdvanceHomogeneousGasReactorDevice() {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");
    TCHEM_CHECK_ERROR(_is_time_advance_set, "invoke setTimeAdvanceHomogeneousGasReactor first");

    _state.sync_device();
    _t.sync_device();
    _dt.sync_device();

    TChem::IgnitionZeroD::runDeviceBatch(
      _policy, _tol_newton, _tol_time, _fac, _tadv,
      _state.view_device(), _t.view_device(), _dt.view_device(), _state.view_device(), _kmcd_device);
    Kokkos::fence();

    _state.modify_device();
    _t.modify_device();
    _dt.modify_device();
    
    /// to avoid implicitly capturing "this" pointer
    auto tadv = _tadv;
    auto t_dv = _t.view_device();
    auto dt_dv = _dt.view_device();
    real_type tsum(0);
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<exec_space>(0, _n_sample),
      KOKKOS_LAMBDA(const ordinal_type& i, real_type& update) {
        tadv(i)._tbeg = t_dv(i);
        tadv(i)._dt = dt_dv(i);
        update += t_dv(i);
      },
      tsum);

    tsum /= _n_sample;
    //printf("computeTimeAdvanceHomogeneousGasReactorDevice current average time %e\n", tsum);
    return  tsum;
  }

  void
  Driver::
  getTimeStepHost(real_type_1d_const_view_host& view) {
    _t.sync_host();
    auto hv = _t.view_host();
    view = real_type_1d_const_view_host(&hv(0), hv.extent(0));
  }

  void
  Driver::
  getTimeStepSizeHost(real_type_1d_const_view_host& view) {
    _dt.sync_host();
    auto hv = _dt.view_host();
    view = real_type_1d_const_view_host(&hv(0), hv.extent(0));
  }

  void
  Driver::
  createAllViews() {
    createStateVector();
    createNetProductionRatePerMass();
    createJacobianHomogeneousGasReactor();
    createRHS_HomogeneousGasReactor();
    createReactionRateConstants();
    createEnthapyMass();
  }

  void
  Driver::
  freeAllViews() {
    freeStateVector();
    freeNetProductionRatePerMass();
    freeJacobianHomogeneousGasReactor();
    freeRHS_HomogeneousGasReactor();
    freeReactionRateConstants();
    freeEnthapyMass();
  }

  void
  Driver::
  showViews(const std::string& label) {
    auto show_2d_view_info = [](std::string name, real_type_2d_dual_view view) {
      if (view.span() > 0) {
        auto hv = view.view_host();
	std::cout << name
		  << ", (" << hv.extent(0) << "," << hv.extent(1) << "), ";
	if (view.need_sync_host()) {
          std::cout << "NeedSyncHost";
        } else if (view.need_sync_device()) { 
          std::cout << "NeedSyncToDevice";
        } else {
          std::cout << "Sync'ed";
        }
	std::cout << "\n";
      }
    };
    std::cout << label << std::endl;
    show_2d_view_info("StateVector", _state);
    show_2d_view_info("NetProductionRatePerMass", _net_production_rate_per_mass);
    std::cout << std::endl;
  }
}


/// C and Fotran interface
static TChem::Driver * g_tchem = nullptr;

void TChem_createKineticModel(const char * chem_file, const char * therm_file) {
  if (!Kokkos::is_initialized())
    Kokkos::initialize();

  if (g_tchem != nullptr) {
    g_tchem->freeAllViews();
    g_tchem->freeKineticModel();
    delete g_tchem;
  }

  g_tchem = new TChem::Driver(chem_file, therm_file);
}

bool TChem_isKineticModelCreated() {
  return g_tchem != nullptr;
}

void TChem_freeKineticModel() {
  if (g_tchem != nullptr) {
    g_tchem->freeAllViews();
    g_tchem->freeKineticModel();
    delete g_tchem;
  }
  Kokkos::finalize();
  g_tchem = nullptr;
}

int  TChem_getNumberOfSpecies() {
  return g_tchem == nullptr ? -1 : g_tchem->getNumberOfSpecies();
}

int  TChem_getNumberOfReactions() {
  return g_tchem == nullptr ? -1 : g_tchem->getNumberOfReactions();
}

void TChem_setNumberOfSamples(const int n_sample) {
  if (g_tchem != nullptr) g_tchem->setNumberOfSamples(n_sample);
}

int  TChem_getNumberOfSamples() {
  return g_tchem == nullptr ? -1 : g_tchem->getNumberOfSamples();
}

int  TChem_getLengthOfStateVector() {
  return g_tchem == nullptr ? -1 : g_tchem->getLengthOfStateVector();
}

int  TChem_getSpeciesIndex(const char * species_name) {
  return g_tchem == nullptr ? -1 : g_tchem->getSpeciesIndex(species_name);
}

int  TChem_getStateVariableIndex(const char * var_name) {
  return g_tchem == nullptr ? -1 : g_tchem->getStateVariableIndex(var_name);
}

bool TChem_isStateVectorCreated() {
  return g_tchem == nullptr ? -1 : g_tchem->isStateVectorCreated();
}

void TChem_createStateVector() {
  if (g_tchem != nullptr) g_tchem->createStateVector();
}

void TChem_freeStateVector() {
  if (g_tchem != nullptr) g_tchem->freeStateVector();
}

void TChem_setSingleStateVectorHost(const int i, real_type * state_at_i) {
  if (g_tchem != nullptr) {
    const int m = TChem_getLengthOfStateVector();
    g_tchem->setStateVectorHost(i, real_type_1d_view_host(state_at_i, m));
  }
}

void TChem_setAllStateVectorHost(real_type * state) {
  if (g_tchem != nullptr) {
    const int m0 = TChem_getNumberOfSamples();
    const int m1 = TChem_getLengthOfStateVector();
    g_tchem->setStateVectorHost(real_type_2d_view_host(state, m0, m1));
  }
}

void TChem_getSingleStateVectorHost(const int i, real_type * view) {
  if (g_tchem != nullptr) {
    TChem::real_type_1d_const_view_host const_view;
    g_tchem->getStateVectorHost(i, const_view);
    memcpy(view, const_view.data(), sizeof(real_type)*const_view.span());
  }
}

void TChem_getAllStateVectorHost(real_type * view) {
  if (g_tchem != nullptr) {
    TChem::real_type_2d_const_view_host const_view;
    g_tchem->getStateVectorHost(const_view);
    memcpy(view, const_view.data(), sizeof(real_type)*const_view.span());
  }
}

bool TChem_isNetProductionRatePerMassCreated() {
  return g_tchem == nullptr ? -1 : g_tchem->isNetProductionRatePerMassCreated();
}

void TChem_createNetProductionRatePerMass() {
  if (g_tchem != nullptr) {
    g_tchem->createNetProductionRatePerMass();
  }
}

void TChem_freeNetProductionRatePerMass() {
  if (g_tchem != nullptr) {
    g_tchem->freeNetProductionRatePerMass();
  }
}

void TChem_getSingleNetProductionRatePerMassHost(const int i, real_type * view) {
  if (g_tchem != nullptr) {
    TChem::real_type_1d_const_view_host const_view;
    g_tchem->getNetProductionRatePerMassHost(i, const_view);
    memcpy(view, const_view.data(), sizeof(real_type)*const_view.span());
  }
}

void TChem_getAllNetProductionRatePerMassHost(real_type * view) {
  if (g_tchem != nullptr) {
    TChem::real_type_2d_const_view_host const_view;
    g_tchem->getNetProductionRatePerMassHost(const_view);
    memcpy(view, const_view.data(), sizeof(real_type)*const_view.span());
  }
}

void TChem_computeNetProductionRatePerMassDevice() {
  if (g_tchem != nullptr) {
    g_tchem->computeNetProductionRatePerMassDevice();
  }
}

void TChem_unsetTimeAdvance() {
  if (g_tchem != nullptr) {
    g_tchem->unsetTimeAdvance();
  }
}

void TChem_setTimeAdvanceHomogeneousGasReactor(const real_type tbeg,
					       const real_type tend,
					       const real_type dtmin,
					       const real_type dtmax,
                                               const int jacobian_interval,
					       const int max_num_newton_iterations,
					       const int num_time_iterations_per_interval,
					       const real_type atol_newton, const real_type rtol_newton,
					       const real_type atol_time, const real_type rtol_time) {
  if (g_tchem != nullptr) {
    g_tchem->setTimeAdvanceHomogeneousGasReactor(tbeg, tend,
						 dtmin, dtmax,
                                                 jacobian_interval,
						 max_num_newton_iterations,
						 num_time_iterations_per_interval,
						 atol_newton, rtol_newton,
						 atol_time, rtol_time);
  }
}

real_type TChem_computeTimeAdvanceHomogeneousGasReactorDevice() {
  return g_tchem == nullptr ? -1 : g_tchem->computeTimeAdvanceHomogeneousGasReactorDevice();
}

void TChem_getTimeStepHost(real_type * view) {
  if (g_tchem != nullptr) {
    TChem::real_type_1d_const_view_host const_view;
    g_tchem->getTimeStepHost(const_view);
    memcpy(view, const_view.data(), sizeof(real_type)*const_view.span());
  }
}

void TChem_getTimeStepSizeHost(real_type * view) {
  if (g_tchem != nullptr) {
    TChem::real_type_1d_const_view_host const_view;
    g_tchem->getTimeStepSizeHost(const_view);
    memcpy(view, const_view.data(), sizeof(real_type)*const_view.span());
  }
}

void TChem_createAllViews() {
  if (g_tchem != nullptr) {
    g_tchem->createAllViews();
  }
}

void TChem_showAllViews(const char * label) {
  if (g_tchem != nullptr) {
    g_tchem->showViews(label);
  }
}

void TChem_freeAllViews() {
  if (g_tchem != nullptr) {
    g_tchem->freeAllViews();
  }
}
