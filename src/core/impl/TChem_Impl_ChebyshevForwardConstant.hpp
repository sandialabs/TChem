#ifndef __TCHEM_IMPL_CHEBYSHEV_FORWARD_CONST_HPP__
#define __TCHEM_IMPL_CHEBYSHEV_FORWARD_CONST_HPP__

namespace TChem {
namespace Impl {
  template<typename ValueType, typename DeviceType>
  struct ChebyshevForwardConstant
  {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;

    using ordinary_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

    /// sacado is value type
    using value_type_1d_view_type = Tines::value_type_1d_view<value_type,device_type>;
    using kinetic_model_type= KineticModelConstData<device_type>;

    KOKKOS_INLINE_FUNCTION static ordinal_type getWorkSpaceSize(
      const kinetic_model_type& kmcd)
    {
      return kmcd.Chebyshev_max_nrows;
    }

    KOKKOS_INLINE_FUNCTION
    static void single_invoke(const ordinal_type& i,
                              const value_type& temperature_inv,
                              const value_type& pressure_log10,
                              const value_type_1d_view_type& kfor,
                              const value_type_1d_view_type& dotProd,
                              const kinetic_model_type& kmcd)
    {
      const real_type one(1.);
      const real_type two(2.);
      const real_type ten(10.);
      const auto ireac = kmcd.ChebyshevCoef(i);
      const auto idata = Kokkos::subview(kmcd.Chebyshev_data, i, Kokkos::ALL());

      // pressure
      {
        value_type Pr = (two * pressure_log10 + ireac._pressure_num) * ireac._pressure_den;
        value_type Cnm1 = Pr;
        value_type Cn = one;

        for (ordinal_type j = 0; j < ireac._nrows; j++) {
          dotProd(j) =idata(j);
          // printf("dotProd(%d):%e \n",j,dotProd(j)  );
        }

        for (ordinal_type j = 1; j < ireac._ncols; j++) {
          value_type Cnp1 = two * Pr * Cn - Cnm1;
          for (ordinal_type k = 0; k < ireac._nrows; k++) {
            dotProd(k) += Cnp1 * idata(j*ireac._nrows + k);
            // printf("dotProd(%d,%d):%e \n",k,j,idata(j*ireac._nrows + k) );
          }
          // printf("dotProd(%d):%e \n",j,dotProd(j)  );
          Cnm1 = Cn;
          Cn = Cnp1;
        }
      }
      // temperature
      {
        value_type Tr = (two  * temperature_inv + ireac._temperature_num) * ireac._temperature_den;
        value_type Cnm1 = Tr;
        value_type Cn = one;
        value_type logk = dotProd(0);
        for (ordinal_type j = 1; j < ireac._nrows ; j++) {
          value_type Cnp1 = two* Tr * Cn - Cnm1;
          logk += Cnp1 * dotProd(j);
          Cnm1 = Cn;
          Cn = Cnp1;
        }
        kfor(ireac._reaction_index) = ats<value_type>::pow(ten, logk);
      }

    }

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    static void team_invoke(const MemberType& member,
                                             /// input
                            const value_type& temperature,
                            const value_type& pressure,
                            const value_type_1d_view_type& kfor, // output
                            const value_type_1d_view_type& dotProd,
                            const kinetic_model_type& kmcd)
    {
      const real_type one(1.);
      const value_type pressure_log10 = ats<value_type>::log10(pressure);
      const value_type temperature_inv = one/temperature;

      // printf("Number of reactions %d \n",kmcd.nReac );
      // printf("pressure_log10 %e\n",pressure_log10 );
      // printf("temperature_inv %e\n", temperature_inv);

      const ordinal_type n_Chebyshev = kmcd.ChebyshevCoef.extent(0);
      // printf("Number of Chebyshev reaction type %d\n", n_Chebyshev );

      Kokkos::parallel_for(
      Tines::RangeFactory<value_type>::TeamVectorRange(member, n_Chebyshev),[&](const ordinal_type& i) {
        single_invoke(i, temperature_inv, pressure_log10, kfor, dotProd, kmcd );
      });

    }

  };

} // namespace Impl
} // namespace TChem

#endif
