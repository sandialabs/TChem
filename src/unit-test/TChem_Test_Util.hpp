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


#ifndef __TCHEM_TEST_UTIL_HPP__
#define __TCHEM_TEST_UTIL_HPP__

#include "TChem_Util.hpp"

TEST(Util, std_vs_kokkos)
{
  /// problem size
  const ordinal_type n0(1000), n1(100);

  /// user fill this vectors
  std::vector<real_type> v1(n0);
  std::vector<std::vector<real_type>> v2(n0, std::vector<real_type>(n1));

  Kokkos::parallel_for(Kokkos::RangePolicy<TChem::host_exec_space>(0, n0),
                       [&](const ordinal_type& i) {
                         v1[i] = i;
                         for (ordinal_type j = 0; j < n1; ++j)
                           v2[i][j] = i * n1 + j;
                       });

  Kokkos::View<real_type*, Kokkos::LayoutRight, TChem::exec_space> v1_kokkos;
  Kokkos::View<real_type**, Kokkos::LayoutRight, TChem::exec_space> v2_kokkos;

  TChem::convertToKokkos(v1_kokkos, v1);
  TChem::convertToKokkos(v2_kokkos, v2);

  EXPECT_EQ(ordinal_type(v1_kokkos.rank), 1);
  EXPECT_EQ(ordinal_type(v1_kokkos.extent(0)), n0);

  EXPECT_EQ(ordinal_type(v2_kokkos.rank), 2);
  EXPECT_EQ(ordinal_type(v2_kokkos.extent(0)), n0);
  EXPECT_EQ(ordinal_type(v2_kokkos.extent(1)), n1);

  /// if all okay with dimensions, test the contents
  { /// we cannot test this on device as device function to be public to capture
    /// necessary things
    /// gtest class has some private values (not sure why it tries to capture
    /// the gtest values though)
    auto v1_kokkos_host = Kokkos::create_mirror_view_and_copy(
      typename TChem::host_exec_space::memory_space(), v1_kokkos);
    auto v2_kokkos_host = Kokkos::create_mirror_view_and_copy(
      typename TChem::host_exec_space::memory_space(), v2_kokkos);
    TChem::Atomic<Kokkos::View<ordinal_type, TChem::host_exec_space>> diff(
      "Test::Util::diff");
    Kokkos::parallel_for(Kokkos::RangePolicy<TChem::host_exec_space>(0, n0),
                         [&](const ordinal_type& i) {
                           diff() += (v1_kokkos_host(i) != i);
                           for (ordinal_type j = 0; j < n1; ++j)
                             diff() += (v2_kokkos_host(i, j) != (i * n1 + j));
                         });
    EXPECT_EQ(diff(), 0);
  }

  /// let's see if the view contents can be converted to std vectors
  v1.assign(n0, real_type(0));
  v2.assign(n0, std::vector<real_type>(n1, 0));

  TChem::convertToStdVector(v1, v1_kokkos);
  TChem::convertToStdVector(v2, v2_kokkos);

  {
    TChem::Atomic<Kokkos::View<ordinal_type, TChem::host_exec_space>> diff(
      "Test::Util::diff");
    Kokkos::parallel_for(Kokkos::RangePolicy<TChem::host_exec_space>(0, n0),
                         [&](const ordinal_type& i) {
                           diff() += v1[i] != i;
                           for (ordinal_type j = 0; j < n1; ++j)
                             diff() += v2[i][j] != (i * n1 + j);
                         });

    EXPECT_EQ(diff(), 0);
  }
}

#endif
