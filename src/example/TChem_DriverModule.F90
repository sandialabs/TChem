!====================================================================================
! TChem version 2.0
! Copyright (2020) NTESS
! https://github.com/sandialabs/TChem

! Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
! Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
! certain rights in this software.

! This file is part of TChem. TChem is open source software: you can redistribute it
! and/or modify it under the terms of BSD 2-Clause License
! (https://opensource.org/licenses/BSD-2-Clause). A copy of the licese is also
! provided under the main directory

! Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
!            Kyungjoo Kim at <kyukim@sandia.gov>, or
!            Oscar Diaz-Ibarra at <odiazib@sandia.gov>

! Sandia National Laboratories, Livermore, CA, USA
!====================================================================================
module TChemDriver
  interface
     subroutine TChem_createKineticModel(arg_chemfile, arg_thermfile) bind(C, name="TChem_createKineticModel")
       use iso_c_binding, only: c_char
       character(kind=c_char) :: arg_chemfile(*)
       character(kind=c_char) :: arg_thermfile(*)
     end subroutine TChem_createKineticModel

     subroutine TChem_setNumberOfSamples(arg_nbatch) bind(C, name="TChem_setNumberOfSamples")
       use iso_c_binding, only: c_int
       integer(c_int), value :: arg_nbatch
     end subroutine TChem_setNumberOfSamples

     subroutine TChem_createStateVector() bind(C, name="TChem_createStateVector")
     end subroutine TChem_createStateVector

     subroutine TChem_setAllStateVectorHost(arg_state_vector) bind(C, name="TChem_setAllStateVectorHost")
       use iso_c_binding, only: c_ptr
       type(c_ptr), value :: arg_state_vector
     end subroutine TChem_setAllStateVectorHost

     subroutine TChem_getAllStateVectorHost(arg_state_vector) bind(C, name="TChem_getAllStateVectorHost")
       use iso_c_binding, only: c_ptr
       type(c_ptr), value :: arg_state_vector
     end subroutine TChem_getAllStateVectorHost
     
     subroutine TChem_computeNetProductionRatePerMassDevice() &
          bind(C, name="TChem_computeNetProductionRatePerMassDevice")
     end subroutine TChem_computeNetProductionRatePerMassDevice
     
     subroutine TChem_getAllNetProductionRatePerMassHost(arg_net_production_rates) &
          bind(C, name="TChem_getAllNetProductionRatePerMassHost")
       use iso_c_binding, only: c_ptr
       type(c_ptr), value :: arg_net_production_rates       
     end subroutine TChem_getAllNetProductionRatePerMassHost

     subroutine TChem_getTimeStepHost(arg_time_step) bind(C, name="TChem_getTimeStepHost")
       use iso_c_binding, only: c_ptr
       type(c_ptr), value :: arg_time_step
     end subroutine TChem_getTimeStepHost

     subroutine TChem_getTimeStepSizeHost(arg_time_step_size) bind(C, name="TChem_getTimeStepSizeHost")
       use iso_c_binding, only: c_ptr
       type(c_ptr), value :: arg_time_step_size
     end subroutine TChem_getTimeStepSizeHost
     
     subroutine TChem_setTimeAdvanceHomogeneousGasReactor( &
          arg_tbeg, arg_tend, arg_dtmin, arg_dtmax, &
          arg_max_num_newton_iterations, arg_num_time_iterations_per_interval, &
          arg_atol_newton, arg_rtol_newton, &
          arg_atol_time, arg_rtol_time) bind(C, name="TChem_setTimeAdvanceHomogeneousGasReactor")
       use iso_c_binding, only: c_int, c_double
       real(c_double), value :: arg_tbeg, arg_tend, arg_dtmin, arg_dtmax
       integer(c_int), value :: arg_max_num_newton_iterations, arg_num_time_iterations_per_interval
       real(c_double), value :: arg_atol_newton, arg_rtol_newton
       real(c_double), value :: arg_atol_time, arg_rtol_time
     end subroutine TChem_setTimeAdvanceHomogeneousGasReactor

     function TChem_computeTimeAdvanceHomogeneousGasReactorDevice() &
          bind(C, name="TChem_computeTimeAdvanceHomogeneousGasReactorDevice")
       use iso_c_binding, only: c_double
       real(c_double) :: TChem_computeTimeAdvanceHomogeneousGasReactorDevice
     end function TChem_computeTimeAdvanceHomogeneousGasReactorDevice
     
     subroutine TChem_showAllViews(arg_label) bind(C, name="TChem_showAllViews")
       use iso_c_binding, only: c_char
       character(kind=c_char) :: arg_label(*)
     end subroutine TChem_showAllViews

     subroutine TChem_freeKineticModel() bind(C, name="TChem_freeKineticModel")
     end subroutine TChem_freeKineticModel

     function TChem_getNumberOfSpecies() bind(C, name="TChem_getNumberOfSpecies")
       use iso_c_binding, only: c_int
       integer(c_int) :: TChem_getNumberOfSpecies
     end function TChem_getNumberOfSpecies

     function TChem_getLengthOfStateVector() bind(C, name="TChem_getLengthOfStateVector")
       use iso_c_binding, only: c_int
       integer(c_int) :: TChem_getLengthOfStateVector
     end function TChem_getLengthOfStateVector

  end interface
end module TChemDriver

!     TChem_setTimeAdvanceHomogeneousGasReactor(tbeg, tend, dtmin, dtmax,
! 					      max_num_newton_iterations, num_time_iterations_per_interval,
! 					      atol_newton, rtol_newton,
! 					      atol_time, rtol_time);
!       TChem_computeTimeAdvanceHomogeneousGasReactorDevice();
!       TChem_getTimeStepHost(t.data());
!       TChem_getTimeStepSizeHost(dt.data());
!       TChem_getSingleStateVectorHost(0, s.data());
