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
#ifndef __TCHEM_TEST_SURF_THERMO_HPP__
#define __TCHEM_TEST_SURF_THERMO_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_GkSurfGas.hpp"


TEST( Thermo, hostBatch ) {
  using host_device_type = typename Tines::UseThisDevice<TChem::host_exec_space>::type;
  /// const data from kmd
  const auto kmcd = TChem::createGasKineticModelConstData<host_device_type>(kmd);
  const auto kmcdSurf =
    TChem::createSurfaceKineticModelConstData<host_device_type>(kmd);

  /// input: state vectors: temperature, pressure and concentration
  const ordinal_type nBatch(1);
  real_type_2d_view_host state("StateVector", nBatch,
   TChem::Impl::getStateVectorSize(kmcd.nSpec));


  real_type_2d_view_host gk("gibbsGas", nBatch, kmcd.nSpec);
  real_type_2d_view_host gkSurf("gibbsSurf", nBatch, kmcdSurf.nSpec);

  real_type_2d_view_host hks("enthalpyGas", nBatch, kmcd.nSpec);
  real_type_2d_view_host hksSurf("enthalpySurf", nBatch, kmcdSurf.nSpec);

 /// input from a file; this is not necessary as the input is created
 /// by other applications.
 {
   auto state_at_0 = Kokkos::subview(state, 0, Kokkos::ALL());
   TChem::Test::readStateVector(_prefixPath + "input-gas.dat",
   kmcd.nSpec, state_at_0);
   TChem::Test::cloneView(state);
 }

 TChem::GkSurfGas::runHostBatch(nBatch,
                                  state,
                                  gk, gkSurf,
                                  hks, hksSurf,
                                  kmcd,
                                  kmcdSurf);


 {

   auto gk_at_0 = Kokkos::subview(gk, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("HostBatch-Gk.txt",
                                   kmcd.nSpec,
                                   gk_at_0);

   auto gkSurf_at_0 = Kokkos::subview(gkSurf, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("HostBatch-GkSurf.txt",
                                   kmcdSurf.nSpec,
                                   gkSurf_at_0);

   auto hks_at_0 = Kokkos::subview(hks, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("HostBatch-Enthalpy.txt",
                                   kmcd.nSpec,
                                   hks_at_0);

   auto hksSurf_at_0 = Kokkos::subview(hksSurf, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("HostBatch-EnthalpySurf.txt",
                                   kmcdSurf.nSpec,
                                   hksSurf_at_0);

 }
 /// compare with ref
 EXPECT_TRUE(TChem::Test::compareFiles("HostBatch-Gk.txt",
 _prefixPath + "Gk.ref.txt"));

 EXPECT_TRUE(TChem::Test::compareFiles("HostBatch-GkSurf.txt",
 _prefixPath + "GkSurf.ref.txt"));

 EXPECT_TRUE(TChem::Test::compareFiles("HostBatch-Enthalpy.txt",
 _prefixPath + "Enthalpy.ref.txt"));

 EXPECT_TRUE(TChem::Test::compareFiles("HostBatch-EnthalpySurf.txt",
 _prefixPath + "EnthalpySurf.ref.txt"));

}

TEST( Thermo, deviceBatch ) {
  using device_type = typename Tines::UseThisDevice<TChem::exec_space>::type;
  /// const data from kmd
  const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);
  const auto kmcdSurf =
    TChem::createSurfaceKineticModelConstData<device_type>(kmd);

  /// input: state vectors: temperature, pressure and concentration
  const ordinal_type nBatch(1);
  real_type_2d_view state("StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));


  real_type_2d_view gk("gibbsGas", nBatch, kmcd.nSpec);
  real_type_2d_view gkSurf("gibbsSurf", nBatch, kmcdSurf.nSpec);

  real_type_2d_view hks("enthalpyGas", nBatch, kmcd.nSpec);
  real_type_2d_view hksSurf("enthalpySurf", nBatch, kmcdSurf.nSpec);

 /// create a mirror view to store input from a file
 auto state_host = Kokkos::create_mirror_view(state);

 /// input from a file; this is not necessary as the input is created
 /// by other applications.
 {
   auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
   TChem::Test::readStateVector(_prefixPath + "input-gas.dat",
   kmcd.nSpec, state_host_at_0);
   TChem::Test::cloneView(state_host);
 }

 Kokkos::deep_copy(state, state_host);

 TChem::GkSurfGas::runDeviceBatch(nBatch,
                                  state,
                                  gk, gkSurf,
                                  hks, hksSurf,
                                  kmcd,
                                  kmcdSurf);


 {

   auto gk_host = Kokkos::create_mirror_view(gk);
   Kokkos::deep_copy(gk_host, gk);

   auto gkSurf_host = Kokkos::create_mirror_view(gkSurf);
   Kokkos::deep_copy(gkSurf_host, gkSurf);

   auto hks_host = Kokkos::create_mirror_view(hks);
   Kokkos::deep_copy(hks_host, hks);

   auto hksSurf_host = Kokkos::create_mirror_view(hksSurf);
   Kokkos::deep_copy(hksSurf_host, hksSurf);


   auto gk_host_at_0 = Kokkos::subview(gk_host, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("DeviceBatch-Gk.txt",
                                   kmcd.nSpec,
                                   gk_host_at_0);

   auto gkSurf_host_at_0 = Kokkos::subview(gkSurf_host, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("DeviceBatch-GkSurf.txt",
                                   kmcdSurf.nSpec,
                                   gkSurf_host_at_0);

   auto hks_host_at_0 = Kokkos::subview(hks_host, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("DeviceBatch-Enthalpy.txt",
                                   kmcd.nSpec,
                                   hks_host_at_0);

   auto hksSurf_host_at_0 = Kokkos::subview(hksSurf_host, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("DeviceBatch-EnthalpySurf.txt",
                                   kmcdSurf.nSpec,
                                   hksSurf_host_at_0);

 }
 /// compare with ref
 EXPECT_TRUE(TChem::Test::compareFiles("DeviceBatch-Gk.txt",
 _prefixPath + "Gk.ref.txt"));

 EXPECT_TRUE(TChem::Test::compareFiles("DeviceBatch-GkSurf.txt",
 _prefixPath + "GkSurf.ref.txt"));

 EXPECT_TRUE(TChem::Test::compareFiles("DeviceBatch-Enthalpy.txt",
 _prefixPath + "Enthalpy.ref.txt"));

 EXPECT_TRUE(TChem::Test::compareFiles("DeviceBatch-EnthalpySurf.txt",
 _prefixPath + "EnthalpySurf.ref.txt"));

}



#endif
