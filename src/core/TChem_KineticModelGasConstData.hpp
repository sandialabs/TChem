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
#ifndef __TCHEM_KINETIC_MODEL_GAS_CONST_DATA_HPP__
#define __TCHEM_KINETIC_MODEL_GAS_CONST_DATA_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"

namespace TChem {

  template<typename DeviceType>
  struct KineticModelGasConstData {
  public:

    using device_type = DeviceType;
    using exec_space_type = typename device_type::execution_space;

    /// non const
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
    using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;

    using ordinal_type_1d_view = Tines::value_type_1d_view<ordinal_type,device_type>;
    using ordinal_type_2d_view = Tines::value_type_2d_view<ordinal_type,device_type>;

    using string_type_1d_view_type = Tines::value_type_1d_view<char [LENGTHOFSPECNAME + 1],device_type>;

    //// const views
    using kmcd_ordinal_type_1d_view = ConstUnmanaged<ordinal_type_1d_view>;
    using kmcd_ordinal_type_2d_view = ConstUnmanaged<ordinal_type_2d_view>;

    using kmcd_real_type_1d_view = ConstUnmanaged<real_type_1d_view_type>;
    using kmcd_real_type_2d_view = ConstUnmanaged<real_type_2d_view_type>;
    using kmcd_real_type_3d_view = ConstUnmanaged<real_type_3d_view_type>;

    using kmcd_string_type_1d_view =ConstUnmanaged<string_type_1d_view_type>;

    real_type rho;
    kmcd_string_type_1d_view speciesNames;

    // ordinal_type nNASAinter;
    // ordinal_type nCpCoef;
    // ordinal_type nArhPar;
    // ordinal_type nLtPar;
    // ordinal_type nJanPar;
    // ordinal_type nFit1Par;
    // ordinal_type nIonSpec;
    // ordinal_type electrIndx;
    // ordinal_type nIonEspec;
    ordinal_type nElem; // this include all elements from chem.inp
    ordinal_type
    NumberofElementsGas; // this only include elements present in gas species
    ordinal_type nSpec;
    ordinal_type nReac;
    // ordinal_type nNASA9coef;
    ordinal_type nThbReac;
    // ordinal_type maxTbInReac;
    ordinal_type nFallReac;
    ordinal_type nFallPar;
    // ordinal_type maxSpecInReac;
    ordinal_type maxOrdPar;
    ordinal_type nRealNuReac;
    ordinal_type nRevReac;
    ordinal_type nOrdReac;
    ordinal_type nPlogReac;

    ordinal_type jacDim;
    bool enableRealReacScoef;

    real_type Runiv;
    real_type Rcal;
    real_type Rcgs;
    real_type TthrmMin;
    real_type TthrmMax;

    // kmcd_string_type_1d_view<LENGTHOFSPECNAME+1> sNames ;

    // kmcd_ordinal_type_1d_view spec9t;
    // kmcd_ordinal_type_1d_view spec9nrng;
    // kmcd_ordinal_type_1d_view sigNu;
    kmcd_ordinal_type_1d_view reacTbdy;
    kmcd_ordinal_type_1d_view reacTbno;
    // kmcd_ordinal_type_1d_view reac_to_Tbdy_index;
    kmcd_ordinal_type_1d_view reacPfal;
    kmcd_ordinal_type_1d_view reacPtype;
    kmcd_ordinal_type_1d_view reacPlohi;
    kmcd_ordinal_type_1d_view reacPspec;
    kmcd_ordinal_type_1d_view isRev;
    // kmcd_ordinal_type_1d_view isDup;
    kmcd_ordinal_type_1d_view reacAOrd;
    // kmcd_ordinal_type_1d_view reacNrp;
    kmcd_ordinal_type_1d_view reacNreac;
    kmcd_ordinal_type_1d_view reacNprod;
    kmcd_ordinal_type_1d_view reacScoef;
    kmcd_ordinal_type_1d_view reacRnu;
    kmcd_ordinal_type_1d_view reacRev;
    kmcd_ordinal_type_1d_view reacHvIdx;
    kmcd_ordinal_type_1d_view reacPlogIdx;
    kmcd_ordinal_type_1d_view reacPlogPno;
    // kmcd_ordinal_type_1d_view sNion;
    // kmcd_ordinal_type_1d_view sCharge;
    // kmcd_ordinal_type_1d_view sTfit;
    // kmcd_ordinal_type_1d_view sPhase;

    kmcd_real_type_2d_view NuIJ;
    kmcd_ordinal_type_2d_view specTbdIdx;
    kmcd_real_type_2d_view reacNuki;
    kmcd_ordinal_type_2d_view reacSidx;
    kmcd_ordinal_type_2d_view specAOidx;
    // kmcd_ordinal_type_2d_view elemCount;

    kmcd_real_type_1d_view sMass;
    // kmcd_real_type_1d_view Tlo;
    kmcd_real_type_1d_view Tmi;
    // kmcd_real_type_1d_view Thi;
    // kmcd_real_type_1d_view sigRealNu;
    // kmcd_real_type_1d_view reacHvPar;
    kmcd_real_type_1d_view kc_coeff;
    // kmcd_real_type_1d_view eMass;

    kmcd_real_type_2d_view RealNuIJ;
    kmcd_real_type_2d_view reacArhenFor;
    kmcd_real_type_2d_view reacArhenRev;
    kmcd_real_type_2d_view specTbdEff;
    kmcd_real_type_2d_view reacPpar;
    kmcd_real_type_2d_view reacRealNuki;
    kmcd_real_type_2d_view specAOval;
    kmcd_real_type_2d_view reacPlogPars;

    kmcd_real_type_3d_view cppol;
    kmcd_real_type_2d_view stoiCoefMatrix;

    // kmcd_real_type_3d_view spec9trng;
    // kmcd_real_type_3d_view spec9coefs;
  };

  template<typename DT>
  KineticModelGasConstData<DT> createGasKineticModelConstData(const KineticModelData & kmd) {
    KineticModelGasConstData<DT> data;
    // using DT = typename DeviceType::execution_space;

    /// given from non-const kinetic model data
    data.rho = real_type(-1); /// there is no minus density if this is minus, rhoset = 0
    data.speciesNames = kmd.sNames_.template view<DT>();
    // data.nNASAinter = kmd.nNASAinter_;
    // data.nCpCoef = kmd.nCpCoef_;
    // data.nArhPar = kmd.nArhPar_;
    // data.nLtPar = kmd.nLtPar_;
    // data.nJanPar = kmd.nJanPar_;
    // data.nFit1Par = kmd.nFit1Par_;
    // data.nIonSpec = kmd.nIonSpec_;
    // data.electrIndx = kmd.electrIndx_;
    // data.nIonEspec = kmd.nIonEspec_;
    data.nElem = kmd.nElem_; // includes all elements from chem.inp
    data.NumberofElementsGas = kmd.NumberofElementsGas_; // only includes elments present in gas phase
    data.nSpec = kmd.nSpec_;
    data.nReac = kmd.nReac_;
    // data.nNASA9coef = kmd.nNASA9coef_;
    data.nThbReac = kmd.nThbReac_;
    // data.maxTbInReac = kmd.maxTbInReac_;
    data.nFallReac = kmd.nFallReac_;
    data.nFallPar = kmd.nFallPar_;
    // data.maxSpecInReac = kmd.maxSpecInReac_;
    data.maxOrdPar = kmd.maxOrdPar_;
    data.nRealNuReac = kmd.nRealNuReac_;
    data.nRevReac = kmd.nRevReac_;
    data.nOrdReac = kmd.nOrdReac_;
    data.nPlogReac = kmd.nPlogReac_;

    data.jacDim = kmd.nSpec_ + 3; /// rho, temperature and pressure

    data.enableRealReacScoef = true;
    {
      const auto tmp = kmd.reacScoef_.template view<host_exec_space>();
      for (ordinal_type i = 0; i < kmd.nReac_; ++i) {
	const bool flag = (tmp(i) != -1);
	data.enableRealReacScoef &= flag;
      }
    }

    data.Runiv = kmd.Runiv_;
    data.Rcal = kmd.Rcal_;
    data.Rcgs = kmd.Rcgs_;
    data.TthrmMin = kmd.TthrmMin_;
    data.TthrmMax = kmd.TthrmMax_;

    // data.spec9t = kmd.spec9t_.template view<DT>();
    // data.spec9nrng = kmd.spec9nrng_.template view<DT>();
    // data.sigNu = kmd.sigNu_.template view<DT>();
    data.reacTbdy = kmd.reacTbdy_.template view<DT>();
    data.reacTbno = kmd.reacTbno_.template view<DT>();
    // data.reac_to_Tbdy_index = kmd.reac_to_Tbdy_index_.template view<DT>();
    data.reacPfal = kmd.reacPfal_.template view<DT>();
    data.reacPtype = kmd.reacPtype_.template view<DT>();
    data.reacPlohi = kmd.reacPlohi_.template view<DT>();
    data.reacPspec = kmd.reacPspec_.template view<DT>();
    data.isRev = kmd.isRev_.template view<DT>();
    // data.isDup = kmd.isDup_.template view<DT>();
    data.reacAOrd = kmd.reacAOrd_.template view<DT>();
    // data.reacNrp = kmd.reacNrp_.template view<DT>();
    data.reacNreac = kmd.reacNreac_.template view<DT>();
    data.reacNprod = kmd.reacNprod_.template view<DT>();
    data.reacScoef = kmd.reacScoef_.template view<DT>();
    data.reacRnu = kmd.reacRnu_.template view<DT>();
    data.reacRev = kmd.reacRev_.template view<DT>();
    data.reacHvIdx = kmd.reacHvIdx_.template view<DT>();
    data.reacPlogIdx = kmd.reacPlogIdx_.template view<DT>();
    data.reacPlogPno = kmd.reacPlogPno_.template view<DT>();
    // data.sNion = kmd.sNion_.template view<DT>();
    // data.sCharge = kmd.sCharge_.template view<DT>();
    // data.sTfit = kmd.sTfit_.template view<DT>();
    // data.sPhase = kmd.sPhase_.template view<DT>();

    data.NuIJ = kmd.NuIJ_.template view<DT>();
    data.specTbdIdx = kmd.specTbdIdx_.template view<DT>();
    data.reacNuki = kmd.reacNuki_.template view<DT>();
    data.reacSidx = kmd.reacSidx_.template view<DT>();
    data.specAOidx = kmd.specAOidx_.template view<DT>();
    // data.elemCount = kmd.elemCount_.template view<DT>(); // (nSpec_, nElem_)

    data.sMass = kmd.sMass_.template view<DT>();
    // data.Tlo = kmd.Tlo_.template view<DT>();
    data.Tmi = kmd.Tmi_.template view<DT>();
    // data.Thi = kmd.Thi_.template view<DT>();
    // data.sigRealNu = kmd.sigRealNu_.template view<DT>();
    // data.reacHvPar = kmd.reacHvPar_.template view<DT>();
    data.kc_coeff = kmd.kc_coeff_.template view<DT>();
    // data.eMass = kmd.eMass_.template view<DT>();

    data.RealNuIJ = kmd.RealNuIJ_.template view<DT>();
    data.reacArhenFor = kmd.reacArhenFor_.template view<DT>();
    data.reacArhenRev = kmd.reacArhenRev_.template view<DT>();
    data.specTbdEff = kmd.specTbdEff_.template view<DT>();
    data.reacPpar = kmd.reacPpar_.template view<DT>();
    data.reacRealNuki = kmd.reacRealNuki_.template view<DT>();
    data.specAOval = kmd.specAOval_.template view<DT>();
    data.reacPlogPars = kmd.reacPlogPars_.template view<DT>();

    data.cppol = kmd.cppol_.template view<DT>();
    // data.spec9trng = kmd.spec9trng_.template view<DT>();
    // data.spec9coefs = kmd.spec9coefs_.template view<DT>();
    // data.sNames = kmd.sNames_.template view<DT>();
    data.stoiCoefMatrix = kmd.stoiCoefMatrix_.template view<DT>();

    return data;
  }

  template<typename DT>
  static inline
  Kokkos::View<KineticModelGasConstData<DT>*,DT>
  createGasKineticModelConstData(const kmd_type_1d_view_host kmds) {
    Kokkos::View<KineticModelGasConstData<DT>*,DT>
      r_val(do_not_init_tag("KMCD::gas phase const data objects"),
	    kmds.extent(0));
    auto r_val_host = Kokkos::create_mirror_view(r_val);
#if !defined (__CUDA_ARCH__)
    Kokkos::parallel_for
      (Kokkos::RangePolicy<host_exec_space>(0, kmds.extent(0)),
       KOKKOS_LAMBDA(const int i) {
	r_val_host(i) = createGasKineticModelConstData<DT>(kmds(i));
      });
    Kokkos::deep_copy(r_val, r_val_host);
#endif
    return r_val;
  }

  /// KK: once code is working, it will be deprecated
  template<typename DT>
  using KineticModelConstData = KineticModelGasConstData<DT>;

} // namespace TChem

#endif
