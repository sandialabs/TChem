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
			    const Tines::value_type_1d_view<KineticModelConstData<DeviceType>,DeviceType>& kmcds) {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;

    using RealType0DViewType = Tines::value_type_0d_view<real_type, DeviceType>;
    using RealType1DViewType = Tines::value_type_1d_view<real_type, DeviceType>;

    auto kmcd_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(kmcds, 0));

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = IgnitionZeroD::getWorkSpaceSize(kmcd_host());

    Kokkos::parallel_for
      (profile_name,
       policy,
       KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
	const ordinal_type i = member.league_rank();
	const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
	const auto tadv_at_i = tadv(i);
	const real_type t_end = tadv_at_i._tend;
	const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
	if (t_out_at_i() < t_end) {
	  const RealType1DViewType state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
	  const RealType1DViewType state_out_at_i = Kokkos::subview(state_out, i, Kokkos::ALL());
	  const RealType1DViewType fac_at_i = Kokkos::subview(fac, i, Kokkos::ALL());
	  
	  const RealType0DViewType dt_out_at_i = Kokkos::subview(dt_out, i);
	  Scratch<RealType1DViewType> work(member.team_scratch(level),
					   per_team_extent);
	  
	  Impl::StateVector<RealType1DViewType> sv_at_i(kmcd_at_i.nSpec, state_at_i);
	  Impl::StateVector<RealType1DViewType> sv_out_at_i(kmcd_at_i.nSpec, state_out_at_i);
	  TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
	  TCHEM_CHECK_ERROR(!sv_out_at_i.isValid(), "Error: input state vector is not valid");
	  {
	    const ordinal_type jacobian_interval = tadv_at_i._jacobian_interval;
	    const ordinal_type max_num_newton_iterations = tadv_at_i._max_num_newton_iterations;
	    const ordinal_type max_num_time_iterations = tadv_at_i._num_time_iterations_per_interval;
	    
	    const real_type
	      dt_in = tadv_at_i._dt,
	      dt_min = tadv_at_i._dtmin,
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
	    Kokkos::parallel_for
	      (Kokkos::TeamVectorRange(member, m),
	       [&](const ordinal_type& i) {
		 vals(i) = i == 0 ? temperature : Ys(i - 1);
	       });
	    member.team_barrier();
	    
	    using ignition_zeroD = Impl::IgnitionZeroD<ValueType,DeviceType>;
	    
	    ignition_zeroD
	      ::team_invoke(member,
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
	    Kokkos::parallel_for
	      (Kokkos::TeamVectorRange(member, m),
	       [&](const ordinal_type& i) {
		 if (i == 0) {
		   temperature_out() = vals(0);
		 } else {
		   Ys_out(i - 1) = vals(i);
		 }
	       });
	    
	    member.team_barrier();
	    density_out() = Impl::RhoMixMs<real_type,DeviceType>
	      ::team_invoke(member, temperature_out(),
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
  IgnitionZeroD_CVODE_TemplateRun( /// required template arguments
				  const std::string& profile_name,
				  const ValueType& dummyValueType,
				  /// team size setting
				  const PolicyType& policy,
				  /// input
				  const Tines::value_type_2d_view<real_type, DeviceType>& tol,
				  const Tines::value_type_2d_view<real_type, DeviceType>& fac,
				  const Tines::value_type_1d_view<time_advance_type, DeviceType>& tadv,
				  const Tines::value_type_2d_view<real_type, DeviceType>& state,
				  /// output
				  const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
				  const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
				  const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
				  /// const data from kinetic model
				  const Tines::value_type_1d_view<KineticModelConstData<DeviceType>,DeviceType>& kmcds,
				  const Tines::value_type_1d_view<Tines::TimeIntegratorCVODE<real_type,DeviceType>,DeviceType>& cvodes) {
    Kokkos::Profiling::pushRegion(profile_name);
#if defined(TINES_ENABLE_TPL_SUNDIALS)
    using policy_type = PolicyType;

    using RealType0DViewType = Tines::value_type_0d_view<real_type, DeviceType>;
    using RealType1DViewType = Tines::value_type_1d_view<real_type, DeviceType>;

    using problem_type = TChem::Impl::IgnitionZeroD_Problem<ValueType,DeviceType>;	        
    
    auto kmcd_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(kmcds, 0));
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd_host());
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = IgnitionZeroD::getWorkSpaceSize(kmcd_host()) + m*m;

    /// this assumes host space
    Kokkos::parallel_for
      (profile_name,
       policy,
       KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
	const ordinal_type i = member.league_rank();
	const auto kmcd = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
        auto cvode = cvodes(i);

	const auto tadv_at_i = tadv(i);
	const real_type t_end = tadv_at_i._tend;
	const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
	if (t_out_at_i() < t_end) {
	  const RealType1DViewType state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
	  const RealType1DViewType state_out_at_i = Kokkos::subview(state_out, i, Kokkos::ALL());
	  const RealType1DViewType fac_at_i = Kokkos::subview(fac, i, Kokkos::ALL());
	  
	  const RealType0DViewType dt_out_at_i = Kokkos::subview(dt_out, i);
	  Scratch<RealType1DViewType> work(member.team_scratch(level), per_team_extent);
	  
	  Impl::StateVector<RealType1DViewType> sv_at_i(kmcd.nSpec, state_at_i);
	  Impl::StateVector<RealType1DViewType> sv_out_at_i(kmcd.nSpec, state_out_at_i);
	  TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
	  TCHEM_CHECK_ERROR(!sv_out_at_i.isValid(), "Error: input state vector is not valid");
	  {
	    const ordinal_type max_num_time_iterations = tadv_at_i._num_time_iterations_per_interval;
	    const real_type
	      dt_in = tadv_at_i._dt,
	      dt_min = tadv_at_i._dtmin,
	      dt_max = tadv_at_i._dtmax;
	    const real_type t_beg = tadv_at_i._tbeg;
	    
	    const auto temperature = sv_at_i.Temperature();
	    const auto pressure = sv_at_i.Pressure();
	    const auto Ys = sv_at_i.MassFractions();
	    
	    const RealType0DViewType temperature_out(sv_out_at_i.TemperaturePtr());
	    const RealType0DViewType pressure_out(sv_out_at_i.PressurePtr());
	    const RealType1DViewType Ys_out = sv_out_at_i.MassFractions();
	    const RealType0DViewType density_out(sv_at_i.DensityPtr());
	    
	    const ordinal_type m = kmcd.nSpec + 1;
	    auto vals = cvode.getStateVector();

	    /// problem setup
	    const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);

	    problem_type problem;
	    problem._kmcd = kmcd;
	    
	    /// problem workspace
	    auto wptr = work.data();
	    auto pw = RealType1DViewType(wptr, problem_workspace_size);
	    wptr += problem_workspace_size;
	    
	    /// error check
	    const ordinal_type workspace_used(wptr - work.data()), workspace_extent(work.extent(0));
	    if (workspace_used > workspace_extent) {
	      Kokkos::abort("Error Ignition ZeroD Sacado : workspace used is larger than it is provided\n");
	    }

	    /// time integrator workspace
	    auto tw = RealType1DViewType(wptr, workspace_extent - workspace_used);
	    
	    /// initialize problem
	    problem._p = pressure; // pressure
	    problem._work = pw;    // problem workspace array
	    problem._kmcd = kmcd;  // kinetic model
	    problem._fac = fac_at_i;
	    problem._work_cvode = tw; // time integrator workspace
	    
	    /// m is nSpec + 1
	    vals(0) = temperature;
	    for (ordinal_type i=1;i<m;++i) 
	      vals(i) = Ys(i - 1);
	    
	    real_type t = t_out_at_i(), dt = 0;
	    cvode.initialize(t,
			     dt_in, dt_min, dt_max,
			     tol(0,0), tol(0,1),
			     TChem::Impl::ProblemIgnitionZeroD_ComputeFunctionCVODE,
			     TChem::Impl::ProblemIgnitionZeroD_ComputeJacobianCVODE);
	    
	    cvode.setProblem(problem);
	    for (ordinal_type iter=0;iter<max_num_time_iterations && t<=t_end;++iter) {
	      const real_type t_prev = t;
	      cvode.advance(t_end, t, 1);
	      dt = t - t_prev;
	    }
	    t_out_at_i() = t;
	    dt_out_at_i() = dt;
	    
	    temperature_out() = vals(0);	    
	    for (ordinal_type i=1;i<m;++i) 
	      Ys_out(i - 1) = vals(i);

	    density_out() = Impl::RhoMixMs<real_type,DeviceType>
	      ::team_invoke(member, temperature_out(),
			    pressure_out(), Ys_out, kmcd);
	  }
	}
      });
#endif
    Kokkos::Profiling::popRegion();
  }
  
  
} // namespace TChem
