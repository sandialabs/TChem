exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDSA.x
mech=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0

export OMP_PROC_BIND=spread 
this="$exec --chemfile=$mech/chem.inp \
            --thermfile=$mech/therm.dat \
            --samplefile=inputs/sample.dat \
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
