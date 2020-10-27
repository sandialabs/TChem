#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH -N 1 
#SBATCH -J isoOctane
##SBATCH --output=test-tchem.out

exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDSA.x
inputs=inputs/
save=-1
dtmin=1e-12
dtmax=1e-3
tend=2
max_time_iterations=500
T_threshold=1500
max_newton_iterations=500
atol_newton=1e-12
rtol_newton=1e-6
tol_time=1e-5
OnlyComputeIgnDelayTime=true

module purge
module load devpack/20190329/openmpi/4.0.1/intel/19.3.199
module load ninja
module list

export OMP_NUM_THREADS=48

$exec --OnlyComputeIgnDelayTimeOnly=$ComputeIgnDelayTime --inputsPath=$inputs --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --max-newton-iterations=$max_newton_iterations --T_threshold=$T_threshold --output_frequency=$save --dtmin=$dtmin --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations 

