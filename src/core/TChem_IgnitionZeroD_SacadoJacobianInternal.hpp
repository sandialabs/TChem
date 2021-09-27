#include "TChem_IgnitionZeroD_SacadoJacobian.hpp"

namespace TChem {
  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  IgnitionZeroD_SacadoJacobian_TemplateRun( /// input
					   const std::string& profile_name,
					   const ValueType& dummyValueType,
					   /// team size setting
					   const PolicyType& policy,
					   const Tines::value_type_2d_view<real_type,DeviceType>& state,
					   const Tines::value_type_3d_view<real_type,DeviceType>& jacobian,
					   const Tines::value_type_2d_view<real_type,DeviceType>& workspace,
					   const KineticModelConstData<DeviceType>& kmcd)
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;
    using value_type  = ValueType;
    using range_type = Kokkos::pair<ordinal_type, ordinal_type>;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;
    using value_type_1d_view_type = Tines::value_type_1d_view<value_type, device_type>;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = IgnitionZeroD_SacadoJacobian::getWorkSpaceSize(kmcd);
    const ordinal_type n = state.extent(0);

    if (workspace.span()) {
      TCHEM_CHECK_ERROR(workspace.extent(0) < policy.league_size(), "Workspace is allocated smaller than the league size");
      TCHEM_CHECK_ERROR(workspace.extent(1) < per_team_extent, "Workspace is allocated smaller than the required");
    }

    Kokkos::parallel_for
      (profile_name,
       policy,
       KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        /// either of these two is used
        real_type_1d_view_type work;
        Scratch<real_type_1d_view_type> swork; 
        if (workspace.span()) {
          work = Kokkos::subview(workspace, member.league_rank(), Kokkos::ALL());
        } else {
          /// assume that the workspace is given from scratch space
          swork = Scratch<real_type_1d_view_type>(member.team_scratch(level), per_team_extent);
          work = real_type_1d_view_type(swork.data(), swork.span());
        }

        auto wptr = work.data();
        const ordinal_type m = kmcd.nSpec + 1;;        
        const ordinal_type len = ats<value_type>::sacadoStorageCapacity();
        value_type_1d_view_type x_fad(wptr, m, m+1); wptr += m*len; 
        value_type_1d_view_type f_fad(wptr, m, m+1); wptr += m*len;
        
        const ordinal_type ws = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);
        real_type_1d_view_type w(wptr, ws); wptr += w.span();
        
        ordinal_type ibeg(0), iend(0), iinc(0);
        Impl::getLeagueRange(member, n, ibeg, iend, iinc);        
        for (ordinal_type i=ibeg;i<iend;i+=iinc) {
          const real_type_1d_view_type s = Kokkos::subview(state, i, Kokkos::ALL());
          const Impl::StateVector<real_type_1d_view_type> sv(kmcd.nSpec, s);
          TCHEM_CHECK_ERROR(!sv.isValid(), "Error: input state vector is not valid");
          const real_type t = sv.Temperature();
          const real_type p = sv.Pressure();
          const real_type_1d_view_type Ys = sv.MassFractions();

          Kokkos::parallel_for
            (Kokkos::TeamVectorRange(member, m),
             [=](const ordinal_type k) {
              if (k == 0) {
                x_fad(0) = value_type(m, k, t);
              } else {
                x_fad(k) = value_type(m, k, Ys(k-1));
              }
            });
          member.team_barrier();

          value_type t_fad = x_fad(0);
          value_type_1d_view_type Ys_fad = Kokkos::subview(x_fad, range_type(1, m));

          Impl::SourceTerm<value_type, device_type>::team_invoke_sacado(member, t_fad, p, Ys_fad, f_fad, w, kmcd);
          member.team_barrier();
          // {
          //   Kokkos::single(Kokkos::PerTeam(member), [=]() {
          //       if (i ==0) {
          //         printf("SourceTerm\n");
          //         for (int k=0;k<4;++k)
          //           printf("%e\n", f_fad(k).val());
          //       }
          //     });
          // }
          Kokkos::parallel_for
            (Kokkos::TeamVectorRange(member, m*m),
             [=](const ordinal_type k) {
              const ordinal_type k0 = k/m, k1 = k%m;
              jacobian(i,k0,k1) = f_fad(k0).fastAccessDx(k1);
            });
          member.team_barrier();
        }
      });
#if defined(KOKKOS_ENABLE_CUDA)
    {
      auto err = cudaGetLastError();
      if (err)
        printf("error %s \n", cudaGetErrorString(err));
    }
#endif
    Kokkos::Profiling::popRegion();
  }
  
} // namespace TChem
