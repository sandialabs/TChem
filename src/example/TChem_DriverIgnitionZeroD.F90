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

program tchem_fortran_driver
  use iso_c_binding
  use tchemdriver
  implicit none
  integer,        parameter :: nbatch = 1
  real(c_double), parameter :: tbeg = 0, tend = 1
  real(c_double), parameter :: dtmin = 1e-11, dtmax = 1e-6
  integer(c_int), parameter :: max_num_newton_iterations = 20, num_time_iterations_per_interval = 10
  real(c_double), parameter :: atol_newton = 1e-12, rtol_newton = 1e-6
  real(c_double), parameter :: atol_time = 1e-12, rtol_time = 1e-6;  
  integer :: isample, i, iter
  real(c_double) :: tsum
  
  !     const real_type 
  integer(c_int) :: num_species
  integer(c_int) :: len_state_vector
  real(c_double), dimension(:,:), allocatable, target :: state_vector
  real(c_double), dimension(:), allocatable, target :: t, dt
  
  ! create kinetic model
  call TChem_createKineticModel( &
       c_char_"data/ignition-zero-d/chem.inp"//c_null_char, &
       c_char_"data/ignition-zero-d/therm.dat"//c_null_char)
  ! create internal data structure for state vector
  call TChem_setNumberOfSamples(nBatch)
  call TChem_createStateVector
  call TChem_showAllViews(c_char_"FYI 1. Created Internal Views"//c_null_char)

  ! create users data structure for state vector and net production rates
  ! C is layout right and Fortran is layout left, flip the indicies
  num_species = TChem_getNumberOfSpecies()
  len_state_vector = TChem_getLengthOfStateVector()
  allocate(state_vector(len_state_vector, nbatch))

  ! read state vector from a file
  open(1, file="data/ignition-zero-d/input.dat")
  do isample=1,nbatch
     do i=1,len_state_vector 
        read(1,*) state_vector(i,isample)
     end do
  end do

  ! compute ignition problem
  call TChem_setAllStateVectorHost(c_loc(state_vector))
  call TChem_setTimeAdvanceHomogeneousGasReactor( &
       tbeg, tend, dtmin, dtmax, &
       max_num_newton_iterations, num_time_iterations_per_interval, &
       atol_newton, rtol_newton, &
       atol_time, rtol_time)

  tsum = 0
  allocate(t(nbatch))
  allocate(dt(nbatch))
  do iter=1,1000
     if (tsum.ge.tend) then
        exit
     end if
     tsum = TChem_computeTimeAdvanceHomogeneousGasReactorDevice()

     call TChem_getAllStateVectorHost(c_loc(state_vector))
     call TChem_getTimeStepHost(c_loc(t))
     call TChem_getTimeStepSizeHost(c_loc(dt))
     print *, t(1), dt(1), state_vector(1:12, 1)
  end do

  ! print net production rates to a file
  call TChem_showAllViews(c_char_"FYI: 2. Created Internal Views"//c_null_char)

  call TChem_freeKineticModel
end program tchem_fortran_driver

