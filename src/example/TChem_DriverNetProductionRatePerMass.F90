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
  integer, parameter :: nbatch = 1
  integer :: isample, i
  
  integer(c_int) :: num_species
  integer(c_int) :: len_state_vector
  real(c_double), dimension(:,:), allocatable, target :: state_vector
  real(c_double), dimension(:,:), allocatable, target :: net_production_rates

  ! create kinetic model
  call TChem_createKineticModel( &
       c_char_"data/reaction-rates/chem.inp"//c_null_char, &
       c_char_"data/reaction-rates/therm.dat"//c_null_char)
  ! create internal data structure for state vector
  call TChem_setNumberOfSamples(nBatch)
  call TChem_createStateVector
  call TChem_showAllViews(c_char_"FYI 1. Created Internal Views"//c_null_char)

  ! create users data structure for state vector and net production rates
  ! C is layout right and Fortran is layout left, flip the indicies
  num_species = TChem_getNumberOfSpecies()
  len_state_vector = TChem_getLengthOfStateVector()
  allocate(state_vector(len_state_vector, nbatch))
  allocate(net_production_rates(num_species, nbatch))

  ! read state vector from a file
  open(1, file="data/reaction-rates/input.dat")
  do isample=1,nbatch
     do i=1,len_state_vector 
        read(1,*) state_vector(i,isample)
     end do
  end do

  ! compute net production rates
  call TChem_setAllStateVectorHost(c_loc(state_vector))
  call TChem_computeNetProductionRatePerMassDevice
  call TChem_getAllNetProductionRatePerMassHost(c_loc(net_production_rates))

  open(2, file="data/reaction-rates/omega-fortran.dat")
  do isample=1,nbatch
     do i=1,len_state_vector 
        write(2,*) net_production_rates(i,isample)
     end do
  end do

  ! print net production rates to a file
  call TChem_showAllViews(c_char_"FYI: 2. Created Internal Views"//c_null_char)

  call TChem_freeKineticModel
end program tchem_fortran_driver

