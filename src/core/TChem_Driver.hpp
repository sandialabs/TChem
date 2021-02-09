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


#ifndef __TCHEM_DRIVER__
#define __TCHEM_DRIVER__

#include "TChem.hpp"

namespace TChem {
  struct Driver {
  public:
    using exec_space = TChem::exec_space;
    using host_exec_space = TChem::host_exec_space;
  
    using real_type = TChem::real_type;

    using real_type_0d_view = TChem::real_type_0d_view;
    using real_type_1d_view = TChem::real_type_1d_view;
    using real_type_2d_view = TChem::real_type_2d_view;

    using real_type_0d_view_host = typename real_type_0d_view::HostMirror;
    using real_type_1d_view_host = typename real_type_1d_view::HostMirror;
    using real_type_2d_view_host = typename real_type_2d_view::HostMirror;

    using real_type_0d_const_view_host = typename real_type_0d_view_host::const_type;
    using real_type_1d_const_view_host = typename real_type_1d_view_host::const_type;
    using real_type_2d_const_view_host = typename real_type_2d_view_host::const_type;

    template<typename ViewType>
    struct DualViewType {
      ViewType _dev;
      typename ViewType::HostMirror _host;
    };

    using real_type_0d_dual_view = DualViewType<real_type_0d_view>;
    using real_type_1d_dual_view = DualViewType<real_type_1d_view>;
    using real_type_2d_dual_view = DualViewType<real_type_2d_view>;
  
  private:
    enum {
      NeedSyncToDevice = 1,
      NeedSyncToHost = -1,
      NoNeedSync = 0
    };

    bool _this_initialize_kokkos;

    bool _kmd_created;
    std::string _chem_file, _therm_file;
    TChem::KineticModelData _kmd;
    TChem::KineticModelConstData<exec_space> _kmcd_device;
    TChem::KineticModelConstData<host_exec_space> _kmcd_host;
  
    int _n_sample;

    /// input
    int _state_need_sync;
    real_type_2d_dual_view _state;

    /// output
    int _net_production_rate_per_mass_need_sync;
    real_type_2d_dual_view _net_production_rate_per_mass;

    void initialize();
    void finalize();
    
  public:
    Driver();
    Driver(const std::string& chem_file, const std::string& therm_file);
    ~Driver();
    
    void createKineticModel(const std::string& chem_file, const std::string& therm_file);
    void freeKineticModel();

    void setNumberOfSamples(const int n_sample);
    int getNumberOfSamples() const;

    int getLengthOfStateVector() const;    
    int getNumerOfSpecies() const;
    int getNumerOfReactions() const;    

    bool isStateVectorCreated() const;
    void createStateVector();
    void freeStateVector();
    void setStateVectorHost(const int i, const real_type_1d_view_host& state_at_i);
    void setStateVectorHost(const real_type_2d_view_host& state); 
    void getStateVectorHost(const int i, real_type_1d_const_view_host& view);
    void getStateVectorHost(real_type_2d_const_view_host& view);  
    void getStateVectorNonConstHost(const int i, real_type_1d_view_host& view);
    void getStateVectorNonConstHost(real_type_2d_view_host& view);  

    bool isNetProductionRatePerMassCreated() const;    
    void createNetProductionRatePerMass();
    void freeNetProductionRatePerMass();
    void getNetProductionRatePerMassHost(const int i, real_type_1d_const_view_host& view);
    void getNetProductionRatePerMassHost(real_type_2d_const_view_host& view);
    void computeNetProductionRatePerMassDevice();

    void createAllViews();
    void freeAllViews();
    void showViews(const std::string& label);
  };
}

#endif
