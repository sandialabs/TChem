#ifndef __TCHEM_IMPL_CONST_VOLUME_IGNITION_REACTOR_S_MATRIX_SENSITIVITY_ANALYSIS_HPP__
#define __TCHEM_IMPL_CONST_VOLUME_IGNITION_REACTOR_S_MATRIX_SENSITIVITY_ANALYSIS_HPP__

#include "TChem_Impl_EnthalpySpecMs.hpp"
#include "TChem_Impl_RateOfProgressInd.hpp"
#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_Util.hpp"
#include "TChem_Impl_ConstantVolumeIgnitionReactor_Problem.hpp"

namespace TChem {
namespace Impl {

template<typename ValueType, typename DeviceType>
struct ConstantVolumeIgnitionReactorSmatrixSensitivityAnalysis
{

  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;

  using ordinary_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using value_type_2d_view_type = Tines::value_type_2d_view<value_type,device_type>;
  using kinetic_model_type= KineticModelConstData<device_type>;



  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd)
  {
    const ordinal_type len_work_for_rate_of_progress =
    Impl::RateOfProgressInd<real_type, device_type>::getWorkSpaceSize(kmcd);

    const ordinal_type workspace_size =  len_work_for_rate_of_progress +   3 * kmcd.nSpec + 2 * kmcd.nReac;
    return workspace_size;
  }

  ///
  ///  \param t  : temperature [K]
  ///  \param Xc : array of \f$N_{spec}\f$ doubles \f$((XC_1,XC_2,...,XC_N)\f$:
  ///              molar concentrations XC \f$[kmol/m^3]\f$
  ///  \return omega : array of \f$N_{spec}\f$ molar reaction rates
  ///  \f$\dot{\omega}_i\f$ \f$\left[kmol/(m^3\cdot s)\right]\f$
  ///
  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    /// input
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    /// output
    const value_type_2d_view_type& Smat_rev, /// (1)
    const value_type_2d_view_type& Smat_fwd, /// (1)
    const value_type_1d_view_type& ak, /// (1)
    /// workspace
    const value_type_1d_view_type& work_for_rate_of_progress,
    const value_type_1d_view_type& hks,
    const value_type_1d_view_type& cpks,
    const value_type_1d_view_type& ropFor,
    const value_type_1d_view_type& ropRev,
    const kinetic_model_type& kmcd)
  {
    const real_type zero(0), one(1);
    using reducer_type = Tines::SumReducer<value_type>;

    using EnthalpySpecMs = EnthalpySpecMsFcn<value_type,device_type>;
    using CpMixMs = Impl::CpMixMs<value_type, device_type>;

    /// 1. compute  cpmix
    const value_type cpmix = CpMixMs::team_invoke(member, t, Ys, cpks, kmcd);

    /// 2. compute rate of progress
    RateOfProgressInd<value_type,device_type>::team_invoke(member,
                                       t,
                                       p,
                                       density,
                                       Ys,
                                       ropFor,
                                       ropRev,
                                       work_for_rate_of_progress,
                                       kmcd);

    const value_type Wmix = MolarWeights<value_type, device_type>
        ::team_invoke(member, Ys, kmcd);

    member.team_barrier();
    // 3. compute enthalpy
    EnthalpySpecMs::team_invoke(member, t, hks, cpks, kmcd);

    // 4. gamma
    const value_type cvmix = cpmix - kmcd.Runiv / Wmix;
    const value_type gamma = cpmix / cvmix;

    Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec),
                         [&](const ordinal_type& i) {
                         ak(i) = zero;
    });

    member.team_barrier();

    /// 5. ensamble S matrices
    Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nReac), [&](const ordinal_type& i) {
        const value_type rop_fwd_at_i = ropFor(i)/density;
        const value_type rop_rev_at_i = ropRev(i)/density;
        // species equations
        // reactants
        real_type sum_term1_fwd(0);
        real_type sum_term2_fwd(0);
        real_type sum_term1_rev(0);
        real_type sum_term2_rev(0);

        for (ordinal_type j = 0; j < kmcd.reacNreac(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j);
          Smat_fwd(kspec + 1 ,i) = kmcd.reacNuki(i, j) * rop_fwd_at_i * kmcd.sMass(kspec);
          Smat_rev(kspec + 1 ,i) = -kmcd.reacNuki(i, j) * rop_rev_at_i * kmcd.sMass(kspec);

          sum_term1_fwd += Smat_fwd(kspec + 1 ,i) * hks(kspec);
          sum_term2_fwd += Smat_fwd(kspec + 1 ,i) / kmcd.sMass(kspec);

          sum_term1_rev += Smat_rev(kspec + 1 ,i) * hks(kspec);
          sum_term2_rev += Smat_rev(kspec + 1 ,i ) / kmcd.sMass(kspec);

          const real_type ak_at_i = - ( Smat_fwd(kspec + 1 ,i) + Smat_rev(kspec + 1 ,i) ) *
                                       kmcd.Runiv * gamma / kmcd.sMass(kspec) / cpmix;

          Kokkos::atomic_add(&ak(kspec), ak_at_i);

        }
        // productos
        const ordinal_type joff = kmcd.reacSidx.extent(1) / 2;
        for (ordinal_type j = 0; j < kmcd.reacNprod(i); ++j) {
          const ordinal_type kspec = kmcd.reacSidx(i, j + joff);
          const value_type val_fwd = kmcd.reacNuki(i, j + joff) * rop_fwd_at_i * kmcd.sMass(kspec);
          const value_type val_rev = - kmcd.reacNuki(i, j + joff) * rop_rev_at_i * kmcd.sMass(kspec);

          Kokkos::atomic_add(&Smat_fwd(kspec +1 ,i), val_fwd);
          Kokkos::atomic_add(&Smat_rev(kspec +1 ,i), val_rev);

          sum_term1_fwd += Smat_fwd(kspec +1 ,i) * hks(kspec);
          sum_term2_fwd += Smat_fwd(kspec +1 ,i) / kmcd.sMass(kspec);

          sum_term1_rev += Smat_rev(kspec +1 ,i) * hks(kspec);
          sum_term2_rev += Smat_rev(kspec +1 ,i) / kmcd.sMass(kspec);

          const real_type ak_at_i = -(val_fwd + val_rev) *
                                     kmcd.Runiv * gamma / kmcd.sMass(kspec) / cpmix;;

          Kokkos::atomic_add(&ak(kspec), ak_at_i);

        }
        // temperature
        Smat_fwd(0,i) = - gamma * sum_term1_fwd / cpmix + (gamma - one) * sum_term2_fwd;
        Smat_rev(0, i ) = - gamma * sum_term1_rev / cpmix + (gamma - one) * sum_term2_rev;

      });

  }

  template<typename MemberType,typename WorkViewType >
  KOKKOS_FORCEINLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    /// input
    const real_type& t,
    const real_type& p,
    const real_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    /// output
    const value_type_2d_view_type& Smat_rev, /// (1)
    const value_type_2d_view_type& Smat_fwd, /// (1)
    const value_type_1d_view_type& ak, /// (1)
    /// workspace
    const WorkViewType& work,
    /// const input from kinetic model
    const kinetic_model_type& kmcd)
  {
    // const real_type zero(0);


    ///
    auto w = (real_type*)work.data();
    const ordinal_type len_work_for_rate_of_progress =
    RateOfProgressInd<real_type, device_type>::getWorkSpaceSize(kmcd);

    // auto Xc = RealType1DViewType(w, kmcd.nSpec);
    // w += kmcd.nSpec;
    auto work_for_rate_of_progress = real_type_1d_view_type(w, len_work_for_rate_of_progress);
    w += len_work_for_rate_of_progress;
    auto hks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto cpks = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;
    auto concX = real_type_1d_view_type(w, kmcd.nSpec);
    w += kmcd.nSpec;

    auto ropFor = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;
    auto ropRev = real_type_1d_view_type(w, kmcd.nReac);
    w += kmcd.nReac;

    team_invoke_detail(member,
                       // inputs
                       t,
                       p,
                       density,
                       Ys,
                       // outputs
                       Smat_rev,
                       Smat_fwd,
                       ak,
                       /// workspace
                       work_for_rate_of_progress,
                       hks,
                       cpks,
                       ropFor,
                       ropRev,
                       kmcd);
  }


};

template<typename ValueType, typename DeviceType>
struct ConstantVolumeIgnitionReactorTLASourceTerm

{
  using value_type = ValueType;
  using device_type = DeviceType;
  using scalar_type = typename ats<value_type>::scalar_type;

  using real_type = scalar_type;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using ordinary_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

  /// sacado is value type
  using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
  using value_type_2d_view_type = Tines::value_type_2d_view<value_type,device_type>;

  using kinetic_model_type= KineticModelConstData<device_type>;

  KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
    const kinetic_model_type& kmcd)
  {

    using problem_type = Impl::ConstantVolumeIgnitionReactor_Problem<value_type, device_type>;
    const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);

    const ordinal_type len_work_smatrix = Impl::ConstantVolumeIgnitionReactorSmatrixSensitivityAnalysis<value_type, device_type>
                                          ::getWorkSpaceSize(kmcd);
    const ordinal_type n_vars(kmcd.nSpec + 1);

    const ordinal_type workspace_size = len_work_smatrix +
                                        3 * n_vars * kmcd.nReac + n_vars + /* Smat_rev, Smat_fwd, ak, sn */
                                        kmcd.nReac + n_vars * n_vars +  n_vars + /* irevs + jacobian + x */
                                        problem_workspace_size; /* work problem*/
    return workspace_size;
  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke_detail(
    const MemberType& member,
    const value_type_1d_view_type& Zn, /// column-major order
    /// outputs
    const value_type_1d_view_type& source, // column-major order
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    const real_type& alpha,
    const real_type_1d_view_type& fac,

    /// workspace
    const real_type_1d_view_type& work_smatrix, /// (1)
    const value_type_2d_view_type& Smat_rev, /// (1)
    const value_type_2d_view_type& Smat_fwd, /// (1)
    const value_type_1d_view_type& ak, /// (1)
    const value_type_2d_view_type& sn,
    const ordinary_type_1d_view_type& irevs,
    const value_type_2d_view_type& Jacobian,
    const real_type_1d_view_type& work_problem,
    const value_type_1d_view_type& x,
    const kinetic_model_type& kmcd)
  {

    using Smatrix_type =
    Impl::ConstantVolumeIgnitionReactorSmatrixSensitivityAnalysis<value_type, device_type>;


    Smatrix_type::team_invoke(member,
                              /// input
                              t,
                              p,
                              density,
                              Ys, /// (kmcd.nSpec)
                              Smat_rev, /// (1)
                              Smat_fwd, /// (1)
                              ak, /// (1)
                              work_smatrix,
                              kmcd);


    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      /// compute iterators
      ordinal_type irev(0);
      for (ordinal_type i = 0; i < kmcd.nReac; ++i) {
        irevs(i) = irev;
        irev += (kmcd.isRev(i)) && (irev < kmcd.nRevReac) &&
                (kmcd.reacRev(irev) == i);
      }
    });



    member.team_barrier();

    Kokkos::parallel_for(
    Kokkos::TeamThreadRange(member, kmcd.nSpec + 1),
    [&](const int &i) {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, kmcd.nSpec),
        [&](const int &j) {
           double val(0);

           for (int k = 0; k < kmcd.nReac; k++) {
             if ( irevs(k) ) {
               val += 0 ; // reaction is irreversible
             } else {
               val +=  alpha * Smat_rev(i,k) * kmcd.stoiCoefMatrix(k,j) / t;
             }

           }

           sn(i,j) = val;
           if ( i == 0)
             Kokkos::atomic_add(&sn(i,j), alpha * ak(j));
       });
    });



    using problem_type = TChem::Impl::ConstantVolumeIgnitionReactor_Problem<value_type, device_type>;

    const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);
    problem_type problem;

    /// problem workspace
    auto wptr = work_problem.data();
    auto pw = real_type_1d_view_type(wptr, problem_workspace_size);
    wptr += problem_workspace_size;

    /// error check
    const ordinal_type workspace_used(wptr - work_problem.data()),
            workspace_extent(work_problem.extent(0));
    if (workspace_used > workspace_extent) {
         Kokkos::abort("Error Ignition ZeroD Sacado : workspace used is larger than it is provided\n");
    }

    /// time integrator workspace
    // auto tw = real_type_1d_view_type(wptr, workspace_extent - workspace_used);
    /// initialize problem
    problem._work = pw;    // problem workspace array
    problem._kmcd = kmcd;  // kinetic model
    problem._fac = fac;
    problem._density= density;
    //
    member.team_barrier();
    //
    // problem.computeJacobian(member, x, Jacobian);
    problem.computeNumericalJacobianRichardsonExtrapolation(member, x, Jacobian);

    /// compiler bug work around... weird code structure..   
    member.team_barrier();
    Kokkos::parallel_for(
    Kokkos::TeamThreadRange(member, kmcd.nSpec+1),
      [=](const int i) {
      const ordinal_type ns(kmcd.nSpec + 1);
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, ns),
          [=](const int j) {
          double val(0);
          for (int k = 0; k < ns; k++) {
                val += Jacobian(i,k)*Zn(ns*k + j);
          }
            source(ns*j + i) = val + sn(i,j);
      });
    });

    member.team_barrier();


  }

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION static void team_invoke(
    const MemberType& member,
    const value_type_1d_view_type& Zn, /// column-major order
    /// outputs
    const value_type_1d_view_type& source, // column-major order
    const value_type& t,
    const value_type& p,
    const value_type& density,
    const value_type_1d_view_type& Ys, /// (kmcd.nSpec)
    const real_type& alpha,
    const real_type_1d_view_type& fac,
    const real_type_1d_view_type& work,
    const kinetic_model_type& kmcd)
  {

    auto w = (real_type*)work.data();

    const ordinal_type len_work_smatrix = Impl::ConstantVolumeIgnitionReactorSmatrixSensitivityAnalysis<value_type, device_type>
                                          ::getWorkSpaceSize(kmcd);
    const ordinal_type n_vars(kmcd.nSpec + 1);

    using problem_type = Impl::ConstantVolumeIgnitionReactor_Problem<value_type, device_type>;
    const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);


    auto work_smatrix = real_type_1d_view_type(w, len_work_smatrix);
    w += len_work_smatrix;
    auto Smat_rev = real_type_2d_view_type(w, n_vars, kmcd.nReac);
    w += n_vars * kmcd.nReac;
    auto Smat_fwd = real_type_2d_view_type(w, n_vars, kmcd.nReac);
    w += n_vars * kmcd.nReac;
    auto sn = real_type_2d_view_type(w, n_vars, kmcd.nReac);
    w += n_vars * kmcd.nReac;
    auto ak = real_type_1d_view_type(w, n_vars);
    w += n_vars;
    auto irevs = ordinary_type_1d_view_type((ordinal_type*)w, kmcd.nReac);
    w += kmcd.nReac;
    auto Jacobian = real_type_2d_view_type(w, n_vars, n_vars);
    w += n_vars * n_vars;
    auto work_problem =real_type_1d_view_type(w, problem_workspace_size);
    w += problem_workspace_size;
    auto x = real_type_1d_view_type(w, n_vars);
    w += n_vars;

    x(0) = t;
    for (ordinal_type i = 1; i < n_vars; i++) {
      x(i) = Ys(i-1);
    }

    team_invoke_detail( member, Zn, /// column-major order
      /// outputs
      source, // column-major order
      //additional inputs
      t,
      p,
      density,
      Ys, /// (kmcd.nSpec)
      alpha,
      fac,
      /// workspace
      work_smatrix, ///
      Smat_rev, ///
      Smat_fwd, ///
      ak, ///
      sn,
      irevs,
      Jacobian,
      work_problem,
      x,
      kmcd);




  }


};



} // namespace Impl
} // namespace TChem

#endif
