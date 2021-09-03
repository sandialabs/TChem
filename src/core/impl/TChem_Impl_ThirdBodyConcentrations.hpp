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
#ifndef __TCHEM_IMPL_THIRDBODYCONCENTRATIONS_HPP__
#define __TCHEM_IMPL_THIRDBODYCONCENTRATIONS_HPP__

#include "TChem_Util.hpp"

namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
  struct ThirdBodyConcentrations
  {

    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;

    using ordinary_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

    /// sacado is value type
    using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
    using kinetic_model_type= KineticModelConstData<device_type>;

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION static void team_invoke(
      const MemberType& member,
      /// output
      const value_type_1d_view_type& concX,
      const value_type_1d_view_type& concM,
      /// const input from kinetic model
      const kinetic_model_type& kmcd)
    {
      const real_type one(1);
      Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac),
                           [&](const ordinal_type& k) { concM(k) = one; });
      if (kmcd.nThbReac > 0) {
        using reducer_type = Tines::SumReducer<value_type>;
        typename reducer_type::value_type concSum(0);

        Kokkos::parallel_reduce(
          Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
          [&](const ordinal_type& k, typename reducer_type::value_type& update) { update += concX(k); },
          reducer_type(concSum));

        Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nThbReac),
                             [&](const ordinal_type& i) {
                               const ordinal_type ireac = kmcd.reacTbdy(i);
                               concM(ireac) = concSum;
                             });
        member.team_barrier();
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, kmcd.nThbReac),
          [&](const ordinal_type& i) {
            const ordinal_type ireac = kmcd.reacTbdy(i);
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(member, kmcd.reacTbno(i)),
              [&](const ordinal_type& j) {
                int kspec = kmcd.specTbdIdx(i, j);
                Kokkos::atomic_add(
                  &concM(ireac), (kmcd.specTbdEff(i, j) - one) * concX(kspec));
              });
          });
      }
  #if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
      if (member.league_rank() == 0) {
        FILE* fs = fopen("ThirdBodyConcentrations.team_invoke.test.out", "a+");
        fprintf(fs, ":: ThirdBodyConcentrations::team_invoke\n");
        fprintf(fs, ":::: input\n");
        fprintf(fs,
                "     nSpec %3d, nReac %d, nThbReac %d\n",
                kmcd.nSpec,
                kmcd.nReac,
                kmcd.nThbReac);
        fprintf(fs, ":::: output\n");
        for (int i = 0; i < int(concX.extent(0)); ++i)
          fprintf(fs, "     i %3d, concX %e\n", i, concX(i));
        for (int i = 0; i < int(concM.extent(0)); ++i)
          fprintf(fs, "     i %3d, concM %e\n", i, concM(i));
      }
  #endif
    }
  };

} // namespace Impl
} // namespace TChem

#endif
