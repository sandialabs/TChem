#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH -N 1
#SBATCH -J isoOctane
##SBATCH --output=test-tchem.out

exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD.x

module purge
module load devpack/20190329/openmpi/4.0.1/intel/19.3.199
module load ninja
module list

export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread

this="$exec --chemfile=chem.inp \
            --thermfile=therm.dat \
            --samplefile=sample10.dat \
            --atol-newton=1e-18 \
            --rtol-newton=1e-8\
            --max-newton-iterations=20 \
            --tol-time=1e-6 \
            --dtmax=1e-3 \
            --dtmin=1e-20 \
            --tend=2 \
            --time-iterations-per-interval=10 \
            --max-time-iterations=500 \
            --only-compute-ignition-delay-time=true \
            --ignition-delay-time-file=IgnitionDelayTime.dat \
            --ignition-delay-time-w-threshold-temperature-file=IgnitionDelayTimeTthreshold.dat \
            --threshold-temperature=1500"

echo $this
eval $this
