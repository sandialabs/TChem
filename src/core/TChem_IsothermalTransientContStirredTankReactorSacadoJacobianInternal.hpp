#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_IsothermalTransientContStirredTankReactorSacadoJacobian.hpp"
#include "TChem_Impl_IsothermalTransientContStirredTankReactorRHS.hpp"
#include "TChem_Impl_RhoMixMs.hpp"

namespace TChem {

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  IsothermalTransientContStirredTankReactorSacadoJacobian_TemplateRun( /// input
                                                            const std::string& profile_name,
                                                            const ValueType& dummyValueType,
                                                            /// team size setting
                                                            const PolicyType& policy,

                                                            const Tines::value_type_2d_view<real_type,DeviceType>& state,
                                                            const Tines::value_type_2d_view<real_type,DeviceType> & site_fraction,
                                                            const Tines::value_type_3d_view<real_type,DeviceType>& Jacobian,
                                                            const Tines::value_type_2d_view<real_type, DeviceType>& rhs,
                                                            const KineticModelConstData<DeviceType>& kmcd,
                                                            const KineticSurfModelConstData<DeviceType>& kmcdSurf,
                                                            const TransientContStirredTankReactorData<DeviceType>& cstr)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;
    using value_type  = ValueType;

    using value_type_1d_view_type = Tines::value_type_1d_view<value_type, device_type>;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    using problem_type = Impl::IsothermalTransientContStirredTankReactor_Problem<value_type, device_type>;
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = IsothermalTransientContStirredTankReactorSacadoJacobian::getWorkSpaceSize(kmcd,kmcdSurf);
    const ordinal_type number_of_equations = problem_type::getNumberOfEquations(kmcd,kmcdSurf);

    Kokkos::parallel_for
      (profile_name,
       policy,
       KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();

        const real_type_1d_view_type state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        const real_type_1d_view_type site_fraction_at_i =
          Kokkos::subview(site_fraction, i, Kokkos::ALL());
        const real_type_2d_view_type Jacobian_at_i =
          Kokkos::subview(Jacobian, i, Kokkos::ALL(), Kokkos::ALL());
        const real_type_1d_view_type rhs_at_i = Kokkos::subview(rhs, i, Kokkos::ALL());

        const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec,
                                                                state_at_i);

        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {
          const real_type_1d_view_type Ys = sv_at_i.MassFractions();
          Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                               per_team_extent + 1);

          const ordinal_type len = ats<value_type>::sacadoStorageCapacity();
          const ordinal_type m = number_of_equations;

          real_type* wptr = work.data();
          value_type_1d_view_type x(wptr, m, m+1); wptr += m*len;
          value_type_1d_view_type f(wptr, m, m+1); wptr += m*len;
          const real_type_1d_view_type mass_flow_out_work(wptr,1);
          wptr += 1;
          const real_type_0d_view_type mass_flow_out = Kokkos::subview(mass_flow_out_work, 0);;

          const real_type_1d_view_type ww(wptr,
                                      work.extent(0) - (wptr - work.data()));

          Kokkos::parallel_for
  	      (Kokkos::TeamVectorRange(member, kmcd.nSpec),
  	        [=](const ordinal_type &i) {
  	        x(i) = value_type(m, i, Ys(i));
  	      });

          Kokkos::parallel_for
  	      (Kokkos::TeamVectorRange(member, kmcdSurf.nSpec),
  	        [=](const ordinal_type &i) {
  	        x(i + kmcd.nSpec) = value_type(m, i+ kmcd.nSpec, site_fraction_at_i(i));
  	      });
          member.team_barrier();
          const real_type temperature_at_i = sv_at_i.Temperature();
          using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
          const value_type_1d_view_type Y_sacado_at_i = Kokkos::subview(x,range_type(0, kmcd.nSpec  ));
          member.team_barrier();

          const value_type density = Impl::RhoMixMs<value_type,device_type>
          ::team_invoke(member,  sv_at_i.Temperature(),
                                 sv_at_i.Pressure(),
                                 Y_sacado_at_i, kmcd);

          const value_type_1d_view_type Z_sacado_at_i =
          Kokkos::subview(x,range_type(kmcd.nSpec, kmcd.nSpec +kmcdSurf.nSpec  ));
          member.team_barrier();
          Impl::IsothermalTransientContStirredTankReactorRHS<value_type, device_type>::team_invoke_sacado(member,
                                                 temperature_at_i, // constant temperature
                                                 Y_sacado_at_i,
                                                 Z_sacado_at_i,
                                                 density,
                                                 cstr.pressure, // constant pressure
                                                 f,
                                                 mass_flow_out,
                                                 ww,
                                                 kmcd,
                                                 kmcdSurf,
                                                 cstr);
          member.team_barrier();
          Kokkos::parallel_for
  	      (Kokkos::TeamThreadRange(member, m),
  	        [=](const ordinal_type &i) {
  	        Kokkos::parallel_for
  	        (Kokkos::ThreadVectorRange(member, m),
  	         [=](const ordinal_type &j) {
  	          Jacobian_at_i(i,j) = f(i).fastAccessDx(j);
  	       });
  	       });

           Kokkos::parallel_for
   	      (Kokkos::TeamVectorRange(member, m),
   	        [&](const ordinal_type &i) {
   	        rhs_at_i(i) = f(i).val();
   	      });

          member.team_barrier();

        }
      });
    Kokkos::Profiling::popRegion();
  }

} // namespace TChem
