#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_TransientContStirredTankReactorSacadoJacobian.hpp"

namespace TChem {

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  TransientContStirredTankReactorSacadoJacobian_TemplateRun( /// input
                                                            const std::string& profile_name,
                                                            const ValueType& dummyValueType,
                                                            /// team size setting
                                                            const PolicyType& policy,
                                                            
                                                            const Tines::value_type_2d_view<real_type,DeviceType>& state,
                                                            const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction,
                                                            const Tines::value_type_3d_view<real_type,DeviceType>& Jacobian,
                                                            const KineticModelConstData<DeviceType>& kmcd,
                                                            const KineticSurfModelConstData<DeviceType>& kmcdSurf,
                                                            const TransientContStirredTankReactorData<DeviceType>& cstr)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;
    using value_type  = ValueType;
    
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;
    
    using problem_type = TChem::Impl::TransientContStirredTankReactor_Problem<value_type, device_type>;
    const ordinal_type level = 1;
    const ordinal_type per_team_extent = TransientContStirredTankReactorSacadoJacobian::getWorkSpaceSize(kmcd,kmcdSurf);
    const ordinal_type work_size_problem = problem_type::getWorkSpaceSize(kmcd,kmcdSurf);
    const ordinal_type number_of_equations = problem_type::getNumberOfEquations(kmcd,kmcdSurf);
    
    Kokkos::parallel_for
      (profile_name,
       policy,
       KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        problem_type problem_at_i;
        problem_at_i._kmcd = kmcd;
        problem_at_i._kmcdSurf = kmcdSurf;
        problem_at_i._cstr = cstr;
        Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                             per_team_extent);
        //
        auto wptr = work.data();
        
        const real_type_1d_view_type x(wptr, number_of_equations);
        wptr += number_of_equations;
        const real_type_1d_view_type w(wptr, work_size_problem);
        wptr += work_size_problem;
        problem_at_i._work = w;
        
        const real_type_1d_view_type state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        //
        const real_type_1d_view_type site_fraction_at_i =
          Kokkos::subview(site_fraction, i, Kokkos::ALL());
        const real_type_2d_view_type Jacobian_at_i =
          Kokkos::subview(Jacobian, i, Kokkos::ALL(), Kokkos::ALL());
        
        
        const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec,
                                                                state_at_i);
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                          "Error: input state vector is not valid");
        {
          x(0) = sv_at_i.Temperature();
          const real_type_1d_view_type Ys = sv_at_i.MassFractions();
          for (ordinal_type k = 0; k < kmcd.nSpec; k++) {
            x(k+1) = Ys(k);
          }
          
          for (ordinal_type k = 0; k < kmcdSurf.nSpec; k++) {
            x(k + kmcd.nSpec + 1) = site_fraction_at_i(k); // site fraction
          }
          
          problem_at_i.computeSacadoJacobian(member, x, Jacobian_at_i);
          
        }
      });
    Kokkos::Profiling::popRegion();
  }
  
} // namespace TChem
