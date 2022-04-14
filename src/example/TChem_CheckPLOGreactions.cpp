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
#include "TChem_CommandLineParser.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Impl_Gk.hpp"
#include "TChem_Impl_KForwardReverse.hpp"
#include "TChem_Util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;

using real_type_0d_view = TChem::real_type_0d_view;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

using ordinal_type_1d_view = TChem::ordinal_type_1d_view;

// #define TCHEM_EXAMPLE_IGNITIONZEROD_QOI_PRINT

int
main(int argc, char* argv[])
{
  /// default inputs
  std::string prefixPath("data/propane/");
  int nBatch(1), team_size(-1), vector_size(-1);
  bool verbose(true);

  int npointsT(10);
  int npointsP(10);
  double temp_s(300); // K
  double temp_e(400);// K
  double press_s(100); // Pa 0.001 bar
  double press_e(4e6); // Pa 0.04

  double factor_pre_exp(1);
  double factor_activation_energy(1);
  double factor_temp_coeff(1);

  std::string chemFile(prefixPath+ "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");

  /// parse command line arguments
  TChem::CommandLineParser opts(
    "Check PLOG reactions in a range of pressures and temperatures.");
  opts.set_option<std::string>(
    "inputsPath", "path to input files e.g., data/inputs", &prefixPath);
  //
  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",&chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<bool>(
    "verbose", "If true, prints aditional information", &verbose);
  // //
  // opts.set_option<int>("team-size", "User defined team size", &team_size);
  // opts.set_option<int>("vector-size", "User defined vector size", &vector_size);
  opts.set_option<int>("npoints-temperature", "number of points temperarure", &npointsT);
  opts.set_option<int>("npoints-pressure", "number of points pressures", &npointsP);
  opts.set_option<double>("initial-temperature", "initial temperarure", &temp_s);
  opts.set_option<double>("final-temperature", "final temperarure", &temp_e);
  opts.set_option<double>("initial-pressure", "initial pressure", &press_s);
  opts.set_option<double>("final-pressure", "final pressure", &press_e);
  opts.set_option<double>("factor-pre-exponential", "factor pre-exponential", &factor_pre_exp);
  opts.set_option<double>("factor-activation-energy", "factor activation energy", &factor_activation_energy);
  opts.set_option<double>("factor-temperature-coeff", "factor temperaturecoeff", &factor_temp_coeff);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse)
    return 0; // print help return


  Kokkos::initialize(argc, argv);
  {

    const bool detail = false;

    TChem::exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);
    using device_type      = typename Tines::UseThisDevice<exec_space>::type;
    using host_device_type      = typename Tines::UseThisDevice<host_exec_space>::type;

    /// construct kmd and use the view for testing
    TChem::KineticModelData kmd(chemFile, thermFile);
    const TChem::KineticModelConstData<device_type> kmcd =
      TChem::createGasKineticModelConstData<device_type>(kmd);


    const ordinal_type stateVecDim =
      TChem::Impl::getStateVectorSize(kmcd.nSpec);

    printf("Number of Species %d \n", kmcd.nSpec);
    printf("Number of Reactions %d \n", kmcd.nReac);

    const real_type dtemp = (temp_e - temp_s) / (npointsT-1);
    const real_type dpress = (press_e - press_s) / (npointsP-1);

    nBatch = npointsT * npointsP;

    real_type_1d_view temperature(do_not_init_tag("temperature"), nBatch);
    real_type_1d_view pressure(do_not_init_tag("pressure"), nBatch);

    Kokkos::parallel_for
      (Kokkos::RangePolicy<exec_space>(0, npointsT), KOKKOS_LAMBDA(const ordinal_type& i) {
        for (ordinal_type j = 0; j < npointsP; j++) {
          temperature(i * npointsP + j ) = temp_s + dtemp * i;
          pressure(i * npointsP + j)    = press_s + dpress * j;
        }
    });
    real_type_1d_view mass_fractions(do_not_init_tag("mass_fractions"), kmcd.nSpec);
    // kfwd does not depend on mass fraction
    Kokkos::deep_copy(mass_fractions, real_type(1.)/kmcd.nSpec);

    real_type_2d_view kFor("kfor gas ", nBatch, kmcd.nReac);
    real_type_2d_view kRev("krev gas" , nBatch, kmcd.nReac);

    if (verbose) {
      printf("initial temperature: %e final temperature: %e number of points: %d \n", temp_s, temp_e, npointsT );
      printf("initial pressure: %e final pressure: %e number of points: %d \n", press_s, press_e, npointsP );
      printf("factor-pre-exponential: %e \n",  factor_pre_exp);
      printf("factor-activation-energy: %e\n", factor_activation_energy );
      printf("factor-temperature-coeff: %e\n", factor_temp_coeff);
    }



    /// different sample would have a different kinetic model
    auto kmds = kmd.clone(nBatch);
    constexpr ordinal_type gas(0);
    real_type_3d_view_host factors(do_not_init_tag("Plog factors"), nBatch, kmcd.nPlogReac, 3);
    // pre-exponential
    auto fastors_A = Kokkos::subview(factors, Kokkos::ALL(),Kokkos::ALL(), 0);
    // activation energy
    auto fastors_E = Kokkos::subview(factors, Kokkos::ALL(),Kokkos::ALL(), 1);
    // temperature coeff
    auto fastors_B = Kokkos::subview(factors, Kokkos::ALL(),Kokkos::ALL(), 2);

    Kokkos::deep_copy(fastors_A, factor_pre_exp);
    Kokkos::deep_copy(fastors_E, factor_activation_energy);
    Kokkos::deep_copy(fastors_B, factor_temp_coeff);

    // check for plog reactions
    ordinal_type_1d_view_host PlogIdx;
    PlogIdx = kmd.reacPlogIdx_.view_host();

    TChem::modifyArrheniusForwardParameter(kmds, gas, PlogIdx, factors );
    auto kmcds = TChem::createGasKineticModelConstData<device_type>(kmds);

    const auto exec_space_instance = TChem::exec_space();

    using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    /// team policy
    policy_type policy(exec_space_instance, nBatch, Kokkos::AUTO());
    const ordinal_type level = 1;

    if (team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space_instance, nBatch, team_size, vector_size);
    }

    const ordinal_type work_kfor_rev_size =
    TChem::Impl::KForwardReverse<real_type,device_type>::getWorkSpaceSize(kmcd);
    const ordinal_type per_team_extent = 3*kmcd.nSpec + 2*kmcd.nReac + work_kfor_rev_size;

    const ordinal_type per_team_scratch =
      TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    Kokkos::parallel_for(
      "TChem::KForwardReverse::example model variation",
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {


        const ordinal_type i = member.league_rank();
        const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
        const real_type_1d_view kfor_at_i = Kokkos::subview(kFor, i, Kokkos::ALL());
        const real_type_1d_view krev_at_i = Kokkos::subview(kRev, i, Kokkos::ALL());

        Scratch<real_type_1d_view> work(member.team_scratch(level),
                                        per_team_extent);

        {
          const real_type t = temperature(i);
          const real_type p = pressure(i);
          const real_type_1d_view Ys = mass_fractions;
          auto w = (real_type*)work.data();

          auto gk = real_type_1d_view(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto hks = real_type_1d_view(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto cpks = real_type_1d_view(w, kmcd_at_i.nSpec);
          w += kmcd_at_i.nSpec;
          auto work_kfor_rev = real_type_1d_view(w, work_kfor_rev_size);
          w += work_kfor_rev_size;
          auto iter_real_type = real_type_1d_view(w, kmcd_at_i.nReac * 2);
          w += kmcd_at_i.nReac * 2;
          auto ww =  (ordinal_type*)iter_real_type.data();
          auto iter = ordinal_type_1d_view(ww,iter_real_type.span());

          Impl::GkFcn<real_type, device_type>::team_invoke(member,
                           t, /// input
                           gk,
                           hks,  /// output
                           cpks, /// workspace
                           kmcd_at_i);
          member.team_barrier();


          Impl::KForwardReverse<real_type, device_type> ::team_invoke(
            member, t, p, gk, kfor_at_i, krev_at_i, iter, work_kfor_rev,  kmcd_at_i);
        }
      });
    Kokkos::Profiling::popRegion();

    printf("\n \n ");
    printf("Any samples produces negative pre-exponential values in plog reactions ... \n Done with verification \n");



  }
  Kokkos::finalize();

  return 0;

}
