/* =====================================================================================
TChem version 2.1.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

This file is part of TChem. TChem is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
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
  initialize() {
    // if (Kokkos::is_initialized()) {
    //   _this_initialize_kokkos = false;
    // } else {
    //   Kokkos::initialize();
    //   _this_initialize_kokkos = true;
    // }
  }

  void
  Driver::
  finalize() {
    // if (_this_initialize_kokkos) {
    //   Kokkos::finalize();
    // }
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
    _state_need_sync(0),
    _state(),
    _net_production_rate_per_mass_need_sync(0),
    _net_production_rate_per_mass() {
    initialize();
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
    _state_need_sync(0),
    _state(),
    _net_production_rate_per_mass_need_sync(0),
    _net_production_rate_per_mass() {
    initialize();
    createKineticModel(chem_file, therm_file);
  }

  Driver::
  ~Driver() {
    freeKineticModel();
    freeAllViews();
    finalize();
  }

  void
  Driver::
  createKineticModel(const std::string& chem_file, const std::string& therm_file) {
    _chem_file = chem_file;
    _therm_file = therm_file;
    
    _kmd = KineticModelData(_chem_file, _therm_file);
    _kmcd_device = _kmd.createConstData<exec_space>();
    _kmcd_host = _kmd.createConstData<host_exec_space>();

    _kmd_created = true;
  }

  void
  Driver::
  freeKineticModel() {
    _chem_file = std::string();
    _therm_file = std::string();
    
    _kmd = KineticModelData();
    _kmcd_device = KineticModelConstData<exec_space>();
    _kmcd_host = KineticModelConstData<host_exec_space>();

    _kmd_created = false;
  }

  void
  Driver::
  setNumberOfSamples(const int n_sample) {
    _n_sample = n_sample;
    freeAllViews();
  }

  int
  Driver::
  getNumberOfSamples() const {
    return _n_sample;
  }

  int
  Driver::
  getLengthOfStateVector() const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    return Impl::getStateVectorSize(_kmcd_device.nSpec);
  }

  int
  Driver::
  getNumerOfSpecies() const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    return _kmcd_device.nSpec;
  }

  int
  Driver::
  getNumerOfReactions() const {
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    return _kmcd_device.nReac;
  }

  bool
  Driver::
  isStateVectorCreated() const {
    return (_state._dev.span() > 0);
  }
  
  void
  Driver::
  createStateVector() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");    
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const int len = Impl::getStateVectorSize(_kmcd_device.nSpec);
    _state._dev = real_type_2d_view("state dev", _n_sample, len);
    _state._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _state._dev);
    _state_need_sync = NoNeedSync;
  }

  void
  Driver::
  freeStateVector() {
    _state._dev = real_type_2d_view();
    _state._host = real_type_2d_view_host();
    _state_need_sync = NoNeedSync;
  }

  void
  Driver::
  setStateVectorHost(const int i, const real_type_1d_view_host& state_at_i) {
    if (!_state._dev.span()) {
      createStateVector();
    }
    const auto src = state_at_i;
    const auto tgt = Kokkos::subview(_state._host, i, Kokkos::ALL());
    Kokkos::deep_copy(tgt, src);

    _state_need_sync = NeedSyncToDevice;    
  }
  
  void
  Driver::
  setStateVectorHost(const real_type_2d_view_host& state) {
    if (!_state._dev.span()) {
      createStateVector();
    }
    using range_type = Kokkos::pair<int,int>;
    const auto src = state;
    const auto tgt = Kokkos::subview(_state._host, range_type(0,src.extent(0)), range_type(0,src.extent(1)));
    Kokkos::deep_copy(tgt, src);

    _state_need_sync = NeedSyncToDevice;    
  }

  void
  Driver::
  getStateVectorHost(const int i, real_type_1d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_state._dev.span() == 0, "State vector should be constructed");    
    if (_state_need_sync == NeedSyncToHost) {
      Kokkos::deep_copy(_state._host, _state._dev);
      _state_need_sync = NoNeedSync;
    }
    view = real_type_1d_const_view_host(&_state._host(i,0), _state._host.extent(1));
  }

  void
  Driver::
  getStateVectorHost(real_type_2d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_state._dev.span() == 0, "State vector should be constructed");    
    if (_state_need_sync == NeedSyncToHost) {
      Kokkos::deep_copy(_state._host, _state._dev);
      _state_need_sync = NoNeedSync; 
    }
    view = real_type_2d_const_view_host(&_state._host(0,0), _state._host.extent(0), _state._host.extent(1));
  }

  void
  Driver::
  getStateVectorNonConstHost(const int i, real_type_1d_view_host& view) {
    TCHEM_CHECK_ERROR(_state._dev.span() == 0, "State vector should be constructed");    
    if (_state_need_sync == NeedSyncToHost) {
      Kokkos::deep_copy(_state._host, _state._dev);
      _state_need_sync = NoNeedSync;
    }
    view = real_type_1d_view_host(&_state._host(i,0), _state._host.extent(1));
    _state_need_sync = NeedSyncToDevice;
  }

  void
  Driver::
  getStateVectorNonConstHost(real_type_2d_view_host& view) {
    TCHEM_CHECK_ERROR(_state._dev.span() == 0, "State vector should be constructed");    
    if (_state_need_sync == NeedSyncToHost) {
      Kokkos::deep_copy(_state._host, _state._dev);
      _state_need_sync = NoNeedSync; 
    }
    view = real_type_2d_view_host(&_state._host(0,0), _state._host.extent(0), _state._host.extent(1));
    _state_need_sync = NeedSyncToDevice;
  }

  bool
  Driver::
  isNetProductionRatePerMassCreated() const {
    return (_net_production_rate_per_mass._dev.span() > 0);
  }
  
  void
  Driver::
  createNetProductionRatePerMass() {
    TCHEM_CHECK_ERROR(_n_sample <= 0, "# of samples should be nonzero");    
    TCHEM_CHECK_ERROR(!_kmd_created, "Kinetic mode first needs to be created");
    const int len = _kmcd_device.nSpec;
    _net_production_rate_per_mass._dev = real_type_2d_view("net_production_rate_per_mass dev", _n_sample, len);
    _net_production_rate_per_mass._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _net_production_rate_per_mass._dev);
    _net_production_rate_per_mass_need_sync = NoNeedSync;
  }

  void
  Driver::
  freeNetProductionRatePerMass() {
    _net_production_rate_per_mass._dev = real_type_2d_view();
    _net_production_rate_per_mass._host = real_type_2d_view_host();
    _net_production_rate_per_mass_need_sync = NoNeedSync;
  }
  
  void
  Driver::
  getNetProductionRatePerMassHost(const int i, real_type_1d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_net_production_rate_per_mass._dev.span() == 0, "State vector should be constructed");    
    if (_net_production_rate_per_mass_need_sync == NeedSyncToHost) {
      Kokkos::deep_copy(_net_production_rate_per_mass._host, _net_production_rate_per_mass._dev);
      _net_production_rate_per_mass_need_sync = NoNeedSync;
    }
    view = real_type_1d_const_view_host(&_net_production_rate_per_mass._host(i,0), _net_production_rate_per_mass._host.extent(1));
  }

  void
  Driver::
  getNetProductionRatePerMassHost(real_type_2d_const_view_host& view) {
    TCHEM_CHECK_ERROR(_net_production_rate_per_mass._dev.span() == 0, "State vector should be constructed");    
    if (_net_production_rate_per_mass_need_sync == NeedSyncToHost) {
      Kokkos::deep_copy(_net_production_rate_per_mass._host, _net_production_rate_per_mass._dev);
      _net_production_rate_per_mass_need_sync = NoNeedSync; 
    }
    view = real_type_2d_const_view_host(&_net_production_rate_per_mass._host(0,0), _net_production_rate_per_mass._host.extent(0), _net_production_rate_per_mass._host.extent(1));
  }

  void
  Driver::
  computeNetProductionRatePerMassDevice() {
    TCHEM_CHECK_ERROR(_kmd_created, "Kinetic mode first needs to be created");     
    if (_state_need_sync == NeedSyncToDevice) {
      Kokkos::deep_copy(_state._dev, _state._host);
      _state_need_sync = NoNeedSync;
    }
    if (_net_production_rate_per_mass._dev.span() == 0) {
      createNetProductionRatePerMass();
    }
    NetProductionRatePerMass::runDeviceBatch
      (_n_sample, _state._dev, _net_production_rate_per_mass._dev, _kmcd_device);
    _net_production_rate_per_mass_need_sync = NeedSyncToHost;
  }

  void
  Driver::
  createAllViews() {
    createStateVector();
    createNetProductionRatePerMass();
  }

  void
  Driver::
  freeAllViews() {
    freeStateVector();
    freeNetProductionRatePerMass();
  }

  void
  Driver::
  showViews(const std::string& label) {
    auto show_2d_view_info = [](std::string name, real_type_2d_view_host view, int sync) {
      if (view.span() > 0) {
	std::cout << name 
		  << ", (" << view.extent(0) << "," << view.extent(1) << "), ";
	if (sync == NoNeedSync) std::cout << "NoNeedSync";
	if (sync == NeedSyncToDevice) std::cout << "NeedSyncToDevice";
	if (sync == NeedSyncToHost) std::cout << "NeedSyncToHost";
	std::cout << "\n";
      }
    };
    std::cout << label << std::endl;
    show_2d_view_info("StateVector", _state._host, _state_need_sync);
    show_2d_view_info("NetProductionRatePerMass", _net_production_rate_per_mass._host, _net_production_rate_per_mass_need_sync); 
    std::cout << std::endl;
  }
}
