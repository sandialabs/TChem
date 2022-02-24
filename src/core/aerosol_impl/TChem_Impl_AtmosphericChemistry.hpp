#ifndef __TCHEM_IMPL_ATMOSPHERIC_CHEMISTRY_HPP__
#define __TCHEM_IMPL_ATMOSPHERIC_CHEMISTRY_HPP__

#include "TChem_Util.hpp"
#include "TChem_Impl_AtmosphericChemistry_Problem.hpp"

#include "Sacado.hpp"
#include "Tines.hpp"


namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct AtmosphericChemistry
{

  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
  using kinetic_model_type= KineticModelNCAR_ConstData<device_type>;

  using time_integrator_type =
  Tines::TimeIntegratorTrBDF2<value_type, device_type>;

  using problem_type = TChem::Impl::AtmosphericChemistry_Problem<value_type, device_type>;

  static inline ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd)
  {

    problem_type problem;
    problem._kmcd = kmcd;
    ordinal_type work_size_problem = problem.getWorkSpaceSize();
    ordinal_type m = problem.getNumberOfEquations();
    ordinal_type wlen(0);
    time_integrator_type::workspace(m, wlen);

    // if(kmcd.nConstSpec > 0) {
    //   work_size_problem += kmcd.nSpec;
    // }

    return wlen + work_size_problem;
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input iteration and qoi index to store
    const ordinal_type & jacobian_interval,
    const ordinal_type& max_num_newton_iterations,
    const ordinal_type& max_num_time_iterations,
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    const real_type_1d_view_type& fac,
    /// input time step and time range
    const real_type& dt_in,
    const real_type& dt_min,
    const real_type& dt_max,
    const real_type& t_beg,
    const real_type& t_end,
    /// input (initial condition)
    const real_type& temperature,
    const real_type& pressure,
    const real_type_1d_view_type& const_vals,
    const real_type_1d_view_type& vals, /// mass fraction (kmcd.nSpec)
    /// output (final output conditions)
    const real_type_0d_view_type& t_out,
    const real_type_0d_view_type& dt_out,
    const real_type_1d_view_type& vals_out,
    /// workspace
    const real_type_1d_view_type& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    using problem_type = TChem::Impl::AtmosphericChemistry_Problem<value_type, device_type>;

    const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);
    problem_type problem;

    /// problem workspace
    auto wptr = work.data();
    auto pw = real_type_1d_view_type(wptr, problem_workspace_size);
    wptr += problem_workspace_size;
    //only needs omega for species mark as constant tracers
    // real_type_1d_view_type omega;
    // if(kmcd.nConstSpec > 0) {
    //  omega = real_type_1d_view_type(wptr, kmcd.nSpec);
    //  wptr += kmcd.nSpec;
    // }
    /// error check
    const ordinal_type workspace_used(wptr - work.data()),
      workspace_extent(work.extent(0));
    if (workspace_used > workspace_extent) {
      Kokkos::abort("Error Atmospheric Chemistry : workspace used is larger than it is provided\n");
    }

    /// time integrator workspace
    auto tw = real_type_1d_view_type(wptr, workspace_extent - workspace_used);

    /// initialize problem
    problem._work = pw;    // problem workspace array
    problem._kmcd = kmcd;  // kinetic model
    problem._fac = fac;
    // problem._omega = omega;
    problem._temperature= temperature;
    problem._pressure= pressure;
    problem._const_concentration= const_vals;

    const ordinal_type r_val =
      time_integrator_type::invoke(member,
                                         problem,
                                         jacobian_interval,
                                         max_num_newton_iterations,
                                         max_num_time_iterations,
                                         tol_newton,
                                         tol_time,
                                         dt_in,
                                         dt_min,
                                         dt_max,
                                         t_beg,
                                         t_end,
                                         vals,
                                         t_out,
                                         dt_out,
                                         vals_out,
                                         tw);

  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input iteration and qoi index to store
    const ordinal_type & jacobian_interval,
    const ordinal_type& max_num_newton_iterations,
    const ordinal_type& max_num_time_iterations,
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    const real_type_1d_view_type& fac,
    /// input time step and time range
    const real_type& dt_in,
    const real_type& dt_min,
    const real_type& dt_max,
    const real_type& t_beg,
    const real_type& t_end,
    /// input (initial condition)
    const real_type& temperature,
    const real_type& pressure,
    const real_type_1d_view_type& const_vals,
    const real_type_1d_view_type& vals, /// temperature, pressure, mass fractions
    /// output (final output conditions)
    const real_type_0d_view_type& t_out,
    const real_type_0d_view_type& dt_out,
    const real_type_1d_view_type& vals_out,
    /// workspace
    const real_type_1d_view_type& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {

    team_invoke_detail(member,
                       jacobian_interval,
                       max_num_newton_iterations,
                       max_num_time_iterations,
                       tol_newton,
                       tol_time,
                       fac,
                       dt_in,
                       dt_min,
                       dt_max,
                       t_beg,
                       t_end,
                       temperature,
                       pressure,
                       const_vals,
                       vals,
                       t_out,
                       dt_out,
                       vals_out,
                       work,
                       kmcd);
    member.team_barrier();
    /// input is valid and output is not valid then, send warning message
#if !defined(__CUDA_ARCH__)
    const real_type zero(0);
    if (dt_in > zero && dt_out() < zero) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        printf("Warning: AtmosphericChemistry sample id(%d) failed\n",
               int(member.league_rank()));
      });
    }
#endif

  }
};

} // namespace Impl
} // namespace TChem

#endif
