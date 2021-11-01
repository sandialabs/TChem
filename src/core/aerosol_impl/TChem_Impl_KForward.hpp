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
#ifndef __TCHEM_IMPL_KFORWARD_AEROSOL_HPP__
#define __TCHEM_IMPL_KFORWARD_AEROSOL_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
// #define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct KForward
{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using kinetic_model_type= KineticModelNCAR_ConstData<device_type>;

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input temperature
    const value_type& t,
    const value_type& p,
    /// output
    const value_type_1d_view_type& kfor,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    const value_type t_1 = real_type(1) / t;
    const value_type tln = ats<value_type>::log(t);
    const value_type one(1.0);
    const value_type conv = kmcd.CONV_PPM * p * t_1;

    // aux factor
    const ordinal_type n_arrhenius_reac = kmcd.ArrheniusCoef.extent(0);

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, n_arrhenius_reac), [&](const ordinal_type& i) {
        const auto param = kmcd.ArrheniusCoef(i);
        kfor(param._reaction_index) = param._A * ats<value_type>::exp(param._C * t_1) *
                                      ( param._B == 0.0 ? one : ats<value_type>::pow (t/param._D, param._B)) *
                                      ( param._E == 0.0 ? one : (one + param._E*p) ) *
                                      ats<value_type>::pow (conv,
                                      kmcd.reacNreac(param._reaction_index) - ordinal_type(1) ) ;
        // printf("Kforward A_ %e B_ %e C_ %e D_ %e E_ %e \n",param._A, param._B,  param._C,  param._D, param._E  );
        // printf("CONV_ %e PRESSURE_PA_ %e TEMPERATURE_K_ %e NUM_REACT_ %d \n",CONV_PPM, p,  t,  kmcd.reacNreac(param._reaction_index) );
        // printf("RATE_CONSTANT_ %e  ireac %d \n", kfor(param._reaction_index), param._reaction_index);
        //
        // const double term1 = param._A * ats<value_type><value_type>::exp(param._C/t) ;
        // const double term2 = (param._B == 0.0 ? one : ats<value_type><real_type>::pow (t/param._D, param._B));
        // const double term3 = (param._E == 0.0 ? one : (one + param._E*p) );
        //
        // printf("term1 %e term2 %e term3 %e \n",term1,term2, term3   );


    });

    // troe reactions
    const ordinal_type n_troe_reactions = kmcd.reacPfal.extent(0);
    //
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, n_troe_reactions), [&](const ordinal_type& i) {
        const auto rp = Kokkos::subview(kmcd.reacPpar, i, Kokkos::ALL());
        const ordinal_type idx_reac = kmcd.reacPfal(i);
        const real_type k0_A(rp(0));
        const real_type k0_B(rp(1));
        const real_type k0_C(rp(2));

        //// kinf_A  kinf_B  kinf_C Fc N
        const real_type kinf_A(rp(3));
        const real_type kinf_B(rp(4));
        const real_type kinf_C(rp(5));
        const real_type Fc(rp(6));
        const real_type N(rp(7));

        const value_type k0 = k0_A * ( k0_C == 0.0 ? one : ats<value_type>::exp(k0_C * t_1) ) *
                             ( k0_B == 0.0 ? one : ats<value_type>::pow (t/real_type(300.0), k0_B) )*
                             conv;

        const value_type k0_kinf = k0 / ( kinf_A * ( kinf_C == 0.0 ? one :ats<value_type>::exp(kinf_C * t_1) )*
                                  ( kinf_B == 0.0 ? one : ats<value_type>::pow (t/real_type(300.0), kinf_B) ) );

        //
        // printf(" K0_A_ %e K0_B_ %e K0_C_ %e k0 %e \n", k0_A, k0_B, k0_C , k0 );
        //
        // printf(" KINF_A_ %e KINF_B_ %e KINF_C_%e kinf_k0 %e \n", kinf_A, kinf_B, kinf_C , k0_kinf );


        kfor(idx_reac) =  k0 / ( real_type(1.0) + k0_kinf ) *
                          ats<value_type>::pow( Fc, real_type(1.0) / ( real_type(1.0) +
                          ats<value_type>::pow( ats<value_type>::log10(k0_kinf) / N , real_type(2.0) ) ) )   *
                          ats<value_type>::pow (conv, kmcd.reacNreac(idx_reac) - ordinal_type(1) ) ;

    });

    const ordinal_type count_cmaq_h2o2_reac = kmcd.CMAQ_H2O2Coef.extent(0);

    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, count_cmaq_h2o2_reac), [&](const ordinal_type& i) {
        const auto param = kmcd.CMAQ_H2O2Coef(i);
        kfor(param._reaction_index) = (param._A1 * ats<value_type>::exp(param._C1 * t_1) *
                                      ( param._B1 == 0.0 ? one : ats<value_type>::pow (t/real_type(300.0), param._B1)) +
                                      param._A2 * ats<value_type>::exp(param._C2 * t_1) *
                                      ( param._B2 == 0.0 ? one : ats<value_type>::pow (t/real_type(300.0), param._B2))*conv) *
                                      ats<value_type>::pow (conv,
                                      kmcd.reacNreac(param._reaction_index) - ordinal_type(1) ) ;


    });


    member.team_barrier();

      // });   /* done computing kforward*/
#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("KForward.team_invoke.test.out", "a+");
    }
#endif
  }

};


} // namespace Impl
} // namespace TChem

#endif
