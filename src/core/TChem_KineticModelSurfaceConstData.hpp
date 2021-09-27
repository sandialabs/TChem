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
#ifndef __TCHEM_KINETIC_MODEL_SURFACE_CONST_DATA_HPP__
#define __TCHEM_KINETIC_MODEL_SURFACE_CONST_DATA_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"

namespace TChem {

  template<typename DeviceType>
  struct KineticModelSurfaceConstData {
  public:
    using device_type = DeviceType;

    /// non const
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
    using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;

    using ordinal_type_1d_view = Tines::value_type_1d_view<ordinal_type,device_type>;
    using ordinal_type_2d_view = Tines::value_type_2d_view<ordinal_type,device_type>;

    using string_type_1d_view_type = Tines::value_type_1d_view<char [LENGTHOFSPECNAME + 1],device_type>;

    using kmcd_coverage_modification_type = Tines::value_type_1d_view<CoverageModification, device_type>;

    //// const views
    using kmcd_ordinal_type_1d_view = ConstUnmanaged<ordinal_type_1d_view>;
    using kmcd_ordinal_type_2d_view = ConstUnmanaged<ordinal_type_2d_view>;

    using kmcd_real_type_1d_view = ConstUnmanaged<real_type_1d_view_type>;
    using kmcd_real_type_2d_view = ConstUnmanaged<real_type_2d_view_type>;
    using kmcd_real_type_3d_view = ConstUnmanaged<real_type_3d_view_type>;

    using kmcd_string_type_1d_view = ConstUnmanaged<string_type_1d_view_type>;

    using kmcd_coverage_modification_type_1d_view = ConstUnmanaged<kmcd_coverage_modification_type>;


    kmcd_coverage_modification_type_1d_view coverageFactor;

    kmcd_string_type_1d_view speciesNames;

    ordinal_type nSpec;
    ordinal_type nReac;
    real_type sitedensity;

    kmcd_real_type_3d_view cppol;
    real_type Runiv;
    // real_type Rcal;
    // real_type Rcgs;
    real_type TthrmMin;
    real_type TthrmMax;
    kmcd_real_type_1d_view Tmi;
    kmcd_real_type_2d_view reacArhenFor;
    kmcd_ordinal_type_1d_view isStick;
    // kmcd_real_type_2d_view reacArhenRev;

    kmcd_ordinal_type_1d_view isRev;

    ordinal_type maxSpecInReac;

    kmcd_ordinal_type_1d_view reacNreac;
    kmcd_ordinal_type_1d_view reacNprod;
    kmcd_ordinal_type_2d_view reacSsrf;
    kmcd_ordinal_type_2d_view reacNuki;
    kmcd_ordinal_type_2d_view reacSidx;

    kmcd_ordinal_type_2d_view vki;
    kmcd_ordinal_type_2d_view vsurfki;

    real_type TChem_reltol;
    real_type TChem_abstol;
  };

  // surface combustion
  template<typename DT>
  KineticModelSurfaceConstData<DT> createSurfaceKineticModelConstData(const KineticModelData & kmd) {
    KineticModelSurfaceConstData<DT> data;
    
    data.speciesNames = kmd.TCsurf_sNames_.template view<DT>();
    data.nSpec = kmd.TCsurf_Nspec_; //
    data.nReac = kmd.TCsurf_Nreac_;
    data.sitedensity = kmd.TCsurf_siteden_;

    data.cppol = kmd.TCsurf_cppol_.template view<DT>();

    data.Runiv = kmd.Runiv_;
    // data.Rcal = kmd.Rcal_;
    // data.Rcgs = kmd.Rcgs_;
    data.TthrmMin = kmd.TCsurf_TthrmMin_;
    data.TthrmMax = kmd.TCsurf_TthrmMax_;
    data.Tmi = kmd.TCsurf_Tmi_.template view<DT>();
    data.reacArhenFor = kmd.TCsurf_reacArhenFor_.template view<DT>();
    data.isStick = kmd.TCsurf_isStick_.template view<DT>();
    data.isRev = kmd.TCsurf_isRev_.template view<DT>();

    data.maxSpecInReac = kmd.TCsurf_maxSpecInReac_;

    data.reacNreac = kmd.TCsurf_reacNreac_.template view<DT>(); //  /* no of reactants only */
    data.reacNprod = kmd.TCsurf_reacNprod_.template view<DT>(); // no of products per reaction
    data.reacSsrf = kmd.TCsurf_reacSsrf_.template view<DT>(); // gas species: 0 surface specie:1
    data.reacNuki = kmd.TCsurf_reacNuki_.template view<DT>(); // Stoichiometric coefficients
    data.reacSidx = kmd.TCsurf_reacSidx_.template view<DT>(); // specie index in gas list or surface list

    /* determine machine precision parameters (for numerical Jac) */
    const real_type two(2);
    const real_type eps = ats<real_type>::epsilon();
    data.TChem_reltol = sqrt(two * eps); // 1e-6;//
    data.TChem_abstol = data.TChem_reltol;

    data.vki = kmd.vski_.template view<DT>();
    data.vsurfki = kmd.vsurfki_.template view<DT>();

    data.coverageFactor = kmd.coverageFactor_.template view<DT>();

    return data;
  }
  
  template<typename DT>
  static inline
  Kokkos::View<KineticModelSurfaceConstData<DT>*,DT>
  createSurfaceKineticModelConstData(Kokkos::View<KineticModelData*,Kokkos::HostSpace> kmds) {
    Kokkos::View<KineticModelSurfaceConstData<DT>*,DT>
      r_val(do_not_init_tag("KMCD::surface phase const data objects"),
	    kmds.extent(0));
    auto r_val_host = Kokkos::create_mirror_view(r_val);
    Kokkos::parallel_for
      (Kokkos::RangePolicy<host_exec_space>(0, kmds.extent(0)),
       KOKKOS_LAMBDA(const int i) {
	r_val_host(i) = createSurfaceKineticModelConstData<DT>(kmds(i));
      });
    Kokkos::deep_copy(r_val, r_val_host);
    return r_val;
  }

  /// KK: once code is working, it will be deprecated
  template<typename DT>
  using KineticSurfModelConstData = KineticModelSurfaceConstData<DT>;

  
} // namespace TChem
#endif
