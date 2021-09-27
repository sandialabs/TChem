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
#ifndef __TCHEM_TEST_SURF_REACTIONRATES_HPP__
#define __TCHEM_TEST_SURF_REACTIONRATES_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_NetProductionRatePerMass.hpp"
#include "TChem_NetProductionRateSurfacePerMass.hpp"

TEST( NetProductionRatePerMass, hostBatch ) {
  /// const data from kmd
  using host_device_type = typename Tines::UseThisDevice<TChem::host_exec_space>::type;

  const auto kmcd = TChem::createGasKineticModelConstData<host_device_type>(kmd);
  const auto kmcdSurf =
    TChem::createSurfaceKineticModelConstData<host_device_type>(kmd);


  /// input: state vectors: density, temperature, pressure and mass concentration
  const ordinal_type nBatch(1);
  real_type_2d_view_host state("StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));

  /// output: omega, reaction rates
  real_type_2d_view_host omega("NetProductionRatePerMass", nBatch, kmcd.nSpec);

  // input :: surface fraction vector, zk
  real_type_2d_view_host zSurf("StateVector", nBatch, kmcdSurf.nSpec);

  /// output: omega, reaction rates from surface (gas species)
  real_type_2d_view_host omegaGasSurf("NetProductionRateSurfacePerMassGasSurf", nBatch, kmcd.nSpec);

  /// output: omega, reaction rates from surface (surface species)
  real_type_2d_view_host omegaSurf("NetProductionRateSurfacePerMassSurf", nBatch, kmcdSurf.nSpec);

  /// input from a file; this is not necessary as the input is created
  /// by other applications.
  {
    auto state_at_0 = Kokkos::subview(state, 0, Kokkos::ALL());
    TChem::Test::readStateVector(_prefixPath + "input-gas.dat",
     kmcd.nSpec, state_at_0);
    TChem::Test::cloneView(state);

    auto zSurf_at_0 = Kokkos::subview(zSurf, 0, Kokkos::ALL());
    TChem::Test::readSiteFraction(_prefixPath + "input-surface.dat",
     kmcdSurf.nSpec, zSurf_at_0);
    TChem::Test::cloneView(zSurf);
  }

  TChem::NetProductionRateSurfacePerMass::runHostBatch(
                                              state,
                                              zSurf,
                                              omegaGasSurf,
                                              omegaSurf,
                                              kmcd,
                                              kmcdSurf);

  /// all values are same
  {
    auto omegaGasSurf_at_0 = Kokkos::subview(omegaGasSurf, 0, Kokkos::ALL());
    TChem::Test::writeReactionRates("HostBatch-reactionrates-gas-surface.txt",
     kmcd.nSpec, omegaGasSurf_at_0);

    auto omegaSurf_at_0 = Kokkos::subview(omegaSurf, 0, Kokkos::ALL());
    TChem::Test::writeReactionRates("HostBatch-reactionrates-surface.txt",
                                    kmcdSurf.nSpec,
                                    omegaSurf_at_0);

  }

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFiles("HostBatch-reactionrates-gas-surface.txt",
   _prefixPath + "reactionrates-gas-surface.ref.txt"));

  /// compare with ref
  EXPECT_TRUE(TChem::Test::compareFiles("HostBatch-reactionrates-surface.txt",
   _prefixPath + "reactionrates-surface.ref.txt"));

}
//
TEST( NetProductionRatePerMass, deviceBatch ) {
  using device_type = typename Tines::UseThisDevice<TChem::exec_space>::type;
  /// const data from kmd
  const auto kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);
  const auto kmcdSurf =
    TChem::createSurfaceKineticModelConstData<device_type>(kmd);


  /// input: state vectors: temperature, pressure and concentration
  const ordinal_type nBatch(1);
  real_type_2d_view state("StateVector", nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));

 //surface phase
 // input :: surface fraction vector, zk
 real_type_2d_view zSurf("StateVector", nBatch, kmcdSurf.nSpec);

 /// output: omega, reaction rates from surface (gas species)
 real_type_2d_view omegaGasSurf("ReactionRatesGasSurf", nBatch, kmcd.nSpec);

 /// output: omega, reaction rates from surface (surface species)
 real_type_2d_view omegaSurf("NetProductionRateSurfacePerMassSurf", nBatch, kmcdSurf.nSpec);

 /// create a mirror view to store input from a file
 auto state_host = Kokkos::create_mirror_view(state);

 /// create a mirror view to store input from a file
 auto zSurf_host = Kokkos::create_mirror_view(zSurf);

 /// input from a file; this is not necessary as the input is created
 /// by other applications.
 {
   auto zSurf_host_host_at_0 = Kokkos::subview(zSurf_host, 0, Kokkos::ALL());
   TChem::Test::readSiteFraction(_prefixPath + "input-surface.dat",
   kmcdSurf.nSpec, zSurf_host_host_at_0);
   TChem::Test::cloneView(zSurf_host);


   auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
   TChem::Test::readStateVector(_prefixPath + "input-gas.dat",
   kmcd.nSpec, state_host_at_0);
   TChem::Test::cloneView(state_host);
 }

 Kokkos::deep_copy(state, state_host);
 Kokkos::deep_copy(zSurf, zSurf_host);

 TChem::NetProductionRateSurfacePerMass::runDeviceBatch(
                                             state,
                                             zSurf,
                                             omegaGasSurf,
                                             omegaSurf,
                                             kmcd,
                                             kmcdSurf);

 auto omegaGasSurf_host = Kokkos::create_mirror_view(omegaGasSurf);
 Kokkos::deep_copy(omegaGasSurf_host, omegaGasSurf);

 auto omegaSurf_host = Kokkos::create_mirror_view(omegaSurf);
 Kokkos::deep_copy(omegaSurf_host, omegaSurf);

 {

   auto omegaGasSurf_host_at_0 = Kokkos::subview(omegaGasSurf_host, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("DeviceBatch-reactionrates-gas-surface.txt",
                                   kmcd.nSpec,
                                   omegaGasSurf_host_at_0);

   auto omegaSurf_host_at_0 = Kokkos::subview(omegaSurf_host, 0, Kokkos::ALL());
   TChem::Test::writeReactionRates("DeviceBatch-reactionrates-surface.txt",
                                   kmcdSurf.nSpec,
                                   omegaSurf_host_at_0);


 }
 /// compare with ref
 EXPECT_TRUE(TChem::Test::compareFiles("DeviceBatch-reactionrates-gas-surface.txt",
 _prefixPath + "reactionrates-gas-surface.ref.txt"));
 /// compare with ref
 EXPECT_TRUE(TChem::Test::compareFiles("DeviceBatch-reactionrates-surface.txt",
 _prefixPath + "reactionrates-surface.ref.txt"));




}



#endif
