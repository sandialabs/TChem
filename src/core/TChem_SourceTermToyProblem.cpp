#include "TChem_SourceTermToyProblem.hpp"

namespace TChem {

  template<typename PolicyType,
           typename RealType1DViewType,
           typename RealType2DViewType,
           typename KineticModelConstType>

  //
  void
  SourceTermToyProblem_TemplateRun( /// input
    const std::string& profile_name,
    /// team size setting
    const PolicyType& policy,
    const RealType1DViewType& theta,
    const RealType1DViewType& lambda,
    const RealType2DViewType& state,
    const RealType2DViewType& SourceTermToyProblem,
    const KineticModelConstType& kmcd
  )
  {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = SourceTermToyProblem::getWorkSpaceSize(kmcd);

    Kokkos::parallel_for(
      profile_name,
      policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const ordinal_type i = member.league_rank();
        const RealType1DViewType state_at_i =
          Kokkos::subview(state, i, Kokkos::ALL());
        const RealType1DViewType SourceTermToyProblem_at_i =
          Kokkos::subview(SourceTermToyProblem, i, Kokkos::ALL());

        Scratch<RealType1DViewType> work(member.team_scratch(level),
                                        per_team_extent);

        //
        const real_type theta_at_i = theta(i);
        const real_type lambda_at_i = lambda(i);

        Impl::SourceTermToyProblem ::team_invoke(member, theta_at_i, lambda_at_i,
          state_at_i, SourceTermToyProblem_at_i, work, kmcd);

      });
    Kokkos::Profiling::popRegion();
  }



void
SourceTermToyProblem::runDeviceBatch( /// input
  typename UseThisTeamPolicy<exec_space>::type& policy,
  const real_type_1d_view& theta,
  const real_type_1d_view& lambda,
  const real_type_2d_view& state,
  /// output
  const real_type_2d_view& SourceTermToyProblem,
  /// const data from kinetic model
  const KineticModelConstDataDevice& kmcd)
{

  SourceTermToyProblem_TemplateRun( /// input
    "TChem::SourceTermToyProblem::runDeviceBatch",
    /// team size setting
    policy,
    theta,
    lambda,
    state,
    SourceTermToyProblem,
    kmcd);

}

void
SourceTermToyProblem::runHostBatch( /// input
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_1d_view_host& theta,
  const real_type_1d_view_host& lambda,
  const real_type_2d_view_host& state,
  /// output
  const real_type_2d_view_host& SourceTermToyProblem,
  /// const data from kinetic model
  const KineticModelConstDataHost& kmcd)
{

  SourceTermToyProblem_TemplateRun( /// input
    "TChem::SourceTermToyProblem::runHostBatch",
    /// team size setting
    policy,
    theta,
    lambda,
    state,
    SourceTermToyProblem,
    kmcd);

}


} // namespace TChem
