exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD.x
mech=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0

export OMP_PROC_BIND=spread 
this="$exec --chemfile=$mech/chem.inp \
            --thermfile=$mech/therm.dat \
            --samplefile=inputs/sample.dat \
            --atol-newton=1e-18 \
            --rtol-newton=1e-8\
            --max-newton-iterations=20 \
            --tol-time=1e-6 \
            --dtmax=1e-1 \
            --dtmin=1e-8 \
            --tend=2 \
            --jacobian-interval=4 \
            --time-iterations-per-interval=1 \
            --max-time-iterations=5000 \
            --only-compute-ignition-delay-time=false \
            --ignition-delay-time-file=IgnitionDelayTime.dat \
            --ignition-delay-time-w-threshold-temperature-file=IgnitionDelayTimeTthreshold.dat \
            --threshold-temperature=1500"

echo $this
eval $this
