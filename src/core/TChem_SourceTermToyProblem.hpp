#ifndef __TCHEM_SOURCETERMTOYPROBLEM_HPP__
#define __TCHEM_SOURCETERMTOYPROBLEM_HPP__

#include "TChem_Impl_SourceTermToyProblem.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace TChem {

struct SourceTermToyProblem
{

  template<typename KineticModelConstDataType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstDataType& kmcd)
  {
    return Impl::SourceTermToyProblem::getWorkSpaceSize(kmcd);
  }

  //
  static void runDeviceBatch( /// input
    typename UseThisTeamPolicy<exec_space>::type& policy,
    const real_type_1d_view& theta,
    const real_type_1d_view& lambda,
    const real_type_2d_view& state,
    /// output
    const real_type_2d_view& SourceTermToyProblem,
    /// const data from kinetic model
    const KineticModelConstDataDevice& kmcd);
  //
  static void runHostBatch( /// input
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    const real_type_1d_view_host& theta,
    const real_type_1d_view_host& lambda,
    const real_type_2d_view_host& state,
    /// output
    const real_type_2d_view_host& SourceTermToyProblem,
    /// const data from kinetic model
    const KineticModelConstDataHost& kmcd);

};

} // namespace TChem

#endif
