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
#include "TChem_IgnitionZeroDNumJacobian.hpp"

namespace TChem {

  template<typename PolicyType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename RealType3DViewType,
           typename KineticModelConstType>
  void
  IgnitionZeroDNumJacobian_TemplateRun( /// required template arguments
    const std::string& profile_name,
    const RealType1DViewType& dummy_1d,
    /// team size setting
    const PolicyType& policy,

    // inputs
    const RealType2DViewType& state,
    /// output
    /// const data from kinetic model
    const RealType3DViewType& jac,
    const RealType2DViewType& fac,
    const KineticModelConstType& kmcd)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;

    const ordinal_type m = Impl::IgnitionZeroD_Problem<
      KineticModelConstType>::getNumberOfEquations(kmcd);

    const ordinal_type level = 1;

    const ordinal_type per_team_extent =
     TChem::IgnitionZeroDNumJacobian
          ::getWorkSpaceSize(kmcd); ///

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const RealType1DViewType state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());

        const RealType2DViewType jac_at_i =
        Kokkos::subview(jac, i, Kokkos::ALL(), Kokkos::ALL());

        const RealType1DViewType fac_at_i =
          Kokkos::subview(fac, i, Kokkos::ALL());

        Scratch<RealType1DViewType> work(member.team_scratch(level),
                                         per_team_extent);

        Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec, state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {

          const real_type temperature = sv_at_i.Temperature();
          const real_type pressure = sv_at_i.Pressure();
          const real_type density = sv_at_i.Density();
          const RealType1DViewType Ys = sv_at_i.MassFractions();


          auto wptr = work.data();
          const RealType1DViewType vals(wptr, m);
          wptr += m;
          const RealType1DViewType ww(wptr,
                                      work.extent(0) - (wptr - work.data()));


          //
          /// m is nSpec + 1
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                               [&](const ordinal_type& i) {
                                 vals(i) = i == 0 ? temperature : Ys(i - 1);
                               });
          member.team_barrier();

          Impl::IgnitionZeroDNumJacobian::team_invoke(member,
                                                  vals,
                                                  jac_at_i,
                                                  fac_at_i, // output
                                                  pressure,
                                                  ww,
                                                  kmcd);



        }
      });
    Kokkos::Profiling::popRegion();
  }

  void
  IgnitionZeroDNumJacobian::runDeviceBatch( /// thread block size
    const ordinal_type nBatch,
    const real_type_2d_view& state,
    /// output
    const real_type_3d_view& jac,
    const real_type_2d_view& fac,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd)
  {
    using policy_type = Kokkos::TeamPolicy<exec_space>;
    const ordinal_type per_team_extent =
     TChem::IgnitionZeroDNumJacobian
          ::getWorkSpaceSize(kmcd); ///

    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view>::shmem_size(per_team_extent);

    const ordinal_type level = 1;
    policy_type policy(nBatch, Kokkos::AUTO()); // fine
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    IgnitionZeroDNumJacobian_TemplateRun( /// template arguments deduction
      "TChem::IgnitionZeroDNumJacobian::runDeviceBatch",
      real_type_1d_view(),
      policy,
      state,
      jac,
      fac,
      /// const data of kinetic model
      kmcd);
  }

  void
  IgnitionZeroDNumJacobian::runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_2d_view& state,
    /// output
    const real_type_3d_view& jac,
    const real_type_2d_view& fac,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd)
  {

    IgnitionZeroDNumJacobian_TemplateRun( /// template arguments deduction
      "TChem::IgnitionZeroDNumJacobian::runDeviceBatch",
      real_type_1d_view(),
      policy,
      state,
      jac,
      fac,
      /// const data of kinetic model
      kmcd);
  }

  void
  IgnitionZeroDNumJacobian::runHostBatch( /// thread block size
    const ordinal_type nBatch,
    const real_type_2d_view_host& state,
    /// output
    const real_type_3d_view_host& jac,
    const real_type_2d_view_host& fac,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd)
  {
    using policy_type = Kokkos::TeamPolicy<host_exec_space>;
    const ordinal_type per_team_extent =
     TChem::IgnitionZeroDNumJacobian
          ::getWorkSpaceSize(kmcd); ///

    const ordinal_type per_team_scratch =
        Scratch<real_type_1d_view_host>::shmem_size(per_team_extent);

    const ordinal_type level = 1;
    policy_type policy(nBatch, Kokkos::AUTO()); // fine
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    IgnitionZeroDNumJacobian_TemplateRun( /// template arguments deduction
      "TChem::IgnitionZeroDNumJacobian::runHostBatch",
      real_type_1d_view_host(),
      policy,
      state,
      jac,
      fac,
      /// const data of kinetic model
      kmcd);
  }


  void
  IgnitionZeroDNumJacobian::runHostBatch( /// thread block size
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    const real_type_2d_view_host& state,
    /// output
    const real_type_3d_view_host& jac,
    const real_type_2d_view_host& fac,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd)
    {

    IgnitionZeroDNumJacobian_TemplateRun( /// template arguments deduction
        "TChem::IgnitionZeroDNumJacobian::runHostBatch",
        real_type_1d_view_host(),
        policy,
        state,
        jac,
        fac,
        kmcd);
    }


} // namespace TChem
