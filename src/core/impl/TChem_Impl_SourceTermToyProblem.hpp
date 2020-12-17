#ifndef __TCHEM_IMPL_REACTIONRATESTOYPROBLEM_HPP__
#define __TCHEM_IMPL_REACTIONRATESTOYPROBLEM_HPP__

#include "TChem_Impl_RateOfProgress.hpp"
#include "TChem_Util.hpp"
// #define TCHEM_ENABLE_SERIAL_TEST_OUTPUT
namespace TChem {
namespace Impl {

struct SourceTermToyProblem
{

  ///
  template<typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    return 6 * kmcd.nReac;
  }

  template<typename MemberType,
           typename RealType1DViewType,
           typename OrdinalType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const real_type& theta,
    const real_type& lambda,
    const RealType1DViewType& concX,
    /// output
    const RealType1DViewType& omega, /// (kmcd.nSpec)
    const RealType1DViewType& kfor,
    const RealType1DViewType& krev,
    const RealType1DViewType& ropFor,
    const RealType1DViewType& ropRev,
    const OrdinalType1DViewType& iter,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {

    /// 1. compute forward and reverse rate constants
    {
      const real_type thetac(20.0); //! degrees
      const real_type lambdac(300.0);// ! degrees
      const real_type k1 = ats<real_type>::sin(theta*PI/180) * ats<real_type>::sin(thetac*PI/180) *
                           ats<real_type>::cos(theta*PI/180) * ats<real_type>::cos(thetac*PI/180) *
                           ats<real_type>::cos(lambda*PI/180 - lambdac*PI/180);
      kfor(0) = k1 > 0 ? k1 : 0;
      kfor(1) = 1;
      krev(0) = 0;
      krev(0) = 0;

      // printf("sin(90)%e\n",ats<real_type>::sin(90*PI/180) );
      // printf("cos(90)%e\n",ats<real_type>::cos(90*PI/180) );
    }

    member.team_barrier();


    /// 2. compute rate-of-progress
    RateOfProgress::team_invoke(member,
                                kfor,
                                krev,
                                concX, /// input
                                ropFor,
                                ropRev, /// output
                                iter,   /// workspace for iterators
                                kmcd);

    member.team_barrier();

    /// 3. assemble reaction rates
    auto rop = ropFor;
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        rop(i) -= ropRev(i);
        const real_type rop_at_i = rop(i);
        for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          // omega(kspec) += kmcd.reacNuki(i,j)*rop_at_i;
          const real_type val = kmcd.reacNuki(i, j) * rop_at_i;
          Kokkos::atomic_fetch_add(&omega(kspec), val);
        }
        const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
        for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          // omega(kspec) += kmcd.reacNuki(i,j+joff)*rop_at_i;
          const real_type val = kmcd.reacNuki(i, j + joff) * rop_at_i;
          Kokkos::atomic_fetch_add(&omega(kspec), val);
        }
      });


    member.team_barrier();

#if defined(TCHEM_ENABLE_SERIAL_TEST_OUTPUT) && !defined(__CUDA_ARCH__)
    if (member.league_rank() == 0) {
      FILE* fs = fopen("SourceTermToyProblem.team_invoke.test.out", "a+");
      fprintf(fs, ":: SourceTermToyProblem::team_invoke\n");
      fprintf(fs, ":::: input\n");
      fprintf(fs,
              "     nSpec %3d, nReac %3d, t %e, p %e\n",
              kmcd.nSpec,
              kmcd.nReac,
              t,
              p);
      //
      fprintf(fs, ":: concX\n");
      for (int i = 0; i < int(concX.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, kfor %e\n",
                i,
                concX(i));        
      fprintf(fs, ":: kfor\n");
      for (int i = 0; i < int(kfor.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, kfor %e\n",
                i,
                kfor(i));
      fprintf(fs, ":: krev\n");
      for (int i = 0; i < int(krev.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, krev %e\n",
                i,
                krev(i));
      //
      fprintf(fs, ":: ropFor\n");
      for (int i = 0; i < int(ropFor.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, kfor %e\n",
                i,
                ropFor(i));
      fprintf(fs, ":: ropRev\n");
      for (int i = 0; i < int(ropRev.extent(0)); ++i)
        fprintf(fs,
                "     i %3d, krev %e\n",
                i,
                ropRev(i));
      fprintf(fs, "\n");
      fprintf(fs, ":::: output\n");
      for (int i = 0; i < int(omega.extent(0)); ++i)
        fprintf(fs, "     i %3d, omega %e\n", i, omega(i));
      fprintf(fs, "\n");
    }
#endif

    // if (member.league_rank() == 0) {
    //   FILE *fs = fopen("SourceTermToyProblem.team_invoke.test.out", "a+");
    //   for (int i=0;i<int(Crnd.extent(0));++i)
    //     fprintf(fs, " %d %e\n", i , real_type(1e-3)*Crnd(i)*kfor(i));
    // }
  }


  template<typename MemberType,
           typename WorkViewType,
           typename RealType1DViewType,
           typename KineticModelConstDataType>
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& theta,
    const real_type& lambda,
    const RealType1DViewType& X, /// (kmcd.nSpec)
    /// output
    const RealType1DViewType& omega, /// (kmcd.nSpec)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const KineticModelConstDataType& kmcd)
  {

    ///
    auto w = (real_type*)work.data();
    auto kfor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto krev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropFor = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropRev = RealType1DViewType(w, kmcd.nReac);
    w += kmcd.nReac;
    auto iter = Kokkos::View<ordinal_type*,
                             Kokkos::LayoutRight,
                             typename WorkViewType::memory_space>(
      (ordinal_type*)w, kmcd.nReac * 2);
    w += kmcd.nReac * 2;

    team_invoke_detail(member,
                       theta,
                       lambda,
                       X,
                       omega,
                       kfor,
                       krev,
                       ropFor,
                       ropRev,
                       iter,
                       kmcd);
  }

};

} // namespace Impl
} // namespace TChem

#endif
