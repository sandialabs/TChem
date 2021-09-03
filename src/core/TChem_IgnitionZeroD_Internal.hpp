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
#include "TChem_Util.hpp"
#include "TChem_IgnitionZeroD.hpp"

/// tadv - an input structure for time marching
/// state (nSpec+3) - initial condition of the state vector
/// qidx (lt nSpec+1) - QoI indices to store in qoi output
/// work - work space sized by getWorkSpaceSize
/// tcnt - time counter
/// qoi (time + qidx.extent(0)) - QoI output
/// kmcd - const data for kinetic model

namespace TChem {

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  IgnitionZeroD_TemplateRunModelVariation( /// required template arguments
                                          const std::string& profile_name,
                                          const ValueType& dummyValueType,
                                          /// team size setting
                                          const PolicyType& policy,

                                          /// input
                                          const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& tol_time,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& fac,
                                          const Tines::value_type_1d_view<time_advance_type, DeviceType>& tadv,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& state,
                                          /// output
                                          const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
                                          const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
                                          const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
                                          /// const data from kinetic model
                                          const Kokkos::View<KineticModelConstData<DeviceType >*,DeviceType>& kmcds)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;

    using RealType0DViewType = Tines::value_type_0d_view<real_type, DeviceType>;
    using RealType1DViewType = Tines::value_type_1d_view<real_type, DeviceType>;

    auto kmcd_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         Kokkos::subview(kmcds, 0));

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = IgnitionZeroD::getWorkSpaceSize(kmcd_host());

    Kokkos::parallel_for(
                         profile_name,
                         policy,
                         KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
                           const ordinal_type i = member.league_rank();
                           const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
                           const auto tadv_at_i = tadv(i);
                           const real_type t_end = tadv_at_i._tend;
                           const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
                           if (t_out_at_i() < t_end) {
                             const RealType1DViewType state_at_i =
                               Kokkos::subview(state, i, Kokkos::ALL());
                             const RealType1DViewType state_out_at_i =
                               Kokkos::subview(state_out, i, Kokkos::ALL());
                             //
                             const RealType1DViewType fac_at_i =
                               Kokkos::subview(fac, i, Kokkos::ALL());

                             const RealType0DViewType dt_out_at_i = Kokkos::subview(dt_out, i);
                             Scratch<RealType1DViewType> work(member.team_scratch(level),
                                                              per_team_extent);

                             Impl::StateVector<RealType1DViewType> sv_at_i(kmcd_at_i.nSpec, state_at_i);
                             Impl::StateVector<RealType1DViewType> sv_out_at_i(kmcd_at_i.nSpec,
                                                                               state_out_at_i);
                             TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                                               "Error: input state vector is not valid");
                             TCHEM_CHECK_ERROR(!sv_out_at_i.isValid(),
                                               "Error: input state vector is not valid");
                             {
                               const ordinal_type jacobian_interval =
                                 tadv_at_i._jacobian_interval;
                               const ordinal_type max_num_newton_iterations =
                                 tadv_at_i._max_num_newton_iterations;
                               const ordinal_type max_num_time_iterations =
                                 tadv_at_i._num_time_iterations_per_interval;

                               const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin,
                                 dt_max = tadv_at_i._dtmax;
                               const real_type t_beg = tadv_at_i._tbeg;

                               const auto temperature = sv_at_i.Temperature();
                               const auto pressure = sv_at_i.Pressure();
                               const auto Ys = sv_at_i.MassFractions();

                               const RealType0DViewType temperature_out(sv_out_at_i.TemperaturePtr());
                               const RealType0DViewType pressure_out(sv_out_at_i.PressurePtr());
                               const RealType1DViewType Ys_out = sv_out_at_i.MassFractions();
                               const RealType0DViewType density_out(sv_at_i.DensityPtr());

                               const ordinal_type m = kmcd_at_i.nSpec + 1;
                               auto wptr = work.data();
                               const RealType1DViewType vals(wptr, m);
                               wptr += m;
                               const RealType1DViewType ww(wptr,
                                                           work.extent(0) - (wptr - work.data()));

                               /// we can only guarantee vals is contiguous array. we basically assume
                               /// that a state vector can be arbitrary ordered.

                               /// m is nSpec + 1
                               Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                                                    [&](const ordinal_type& i) {
                                                      vals(i) = i == 0 ? temperature : Ys(i - 1);
                                                    });
                               member.team_barrier();

                               using ignition_zeroD_sacado_type =
                                 Impl::IgnitionZeroD<ValueType,DeviceType>;

                               ignition_zeroD_sacado_type::team_invoke(member,
                                                                       jacobian_interval,
                                                                       max_num_newton_iterations,
                                                                       max_num_time_iterations,
                                                                       tol_newton,
                                                                       tol_time,
                                                                       fac_at_i,
                                                                       dt_in,
                                                                       dt_min,
                                                                       dt_max,
                                                                       t_beg,
                                                                       t_end,
                                                                       pressure,
                                                                       vals,
                                                                       t_out_at_i,
                                                                       dt_out_at_i,
                                                                       pressure_out,
                                                                       vals,
                                                                       ww,
                                                                       kmcd_at_i);

                               member.team_barrier();
                               Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                                                    [&](const ordinal_type& i) {
                                                      if (i == 0) {
                                                        temperature_out() = vals(0);
                                                      } else {
                                                        Ys_out(i - 1) = vals(i);
                                                      }
                                                    });
                               //
                               member.team_barrier();
                               density_out() = Impl::RhoMixMs<real_type,DeviceType>::team_invoke(member, temperature_out(),
                                                                                                 pressure_out(), Ys_out, kmcd_at_i);
                               member.team_barrier();
                             }
                           }
                         });
    Kokkos::Profiling::popRegion();
  }

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  IgnitionZeroD_TemplateRun( /// required template arguments
                            const std::string& profile_name,
                            const ValueType& dummyValueType,
                            /// team size setting
                            const PolicyType& policy,

                            /// input
                            const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton,
                            const Tines::value_type_2d_view<real_type, DeviceType>& tol_time,
                            const Tines::value_type_2d_view<real_type, DeviceType>& fac,
                            const Tines::value_type_1d_view<time_advance_type, DeviceType>& tadv,
                            const Tines::value_type_2d_view<real_type, DeviceType>& state,
                            /// output
                            const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
                            const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
                            const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
                            /// const data from kinetic model
                            const KineticModelConstData<DeviceType>& kmcd)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    Kokkos::View<KineticModelConstData<DeviceType>*,DeviceType>
      kmcds(do_not_init_tag("IgnitionaZeroD::kmcds"), 1);
    Kokkos::deep_copy(kmcds, kmcd);

    IgnitionZeroD_TemplateRunModelVariation
      (profile_name,
       dummyValueType, policy,
       tol_newton, tol_time,
       fac,
       tadv, state,
       t_out, dt_out, state_out, kmcds);

    Kokkos::Profiling::popRegion();
  }

} // namespace TChem
