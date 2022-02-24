exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/isoOctane

this="$exec --chemfile=$inputs/chem.inp \
            --thermfile=$inputs/therm.dat \
            --samplefile=sample.dat \
            --outputfile=IgnSolution_cvode.dat \
            --use-cvode=true \
            --atol-newton=1e-18 \
            --rtol-newton=1e-8\
            --max-newton-iterations=20 \
            --tol-time=1e-8 \
            --atol-time=1e-18 \
            --dtmax=1e-3 \
            --dtmin=1e-20 \
            --tend=0.25 \
            --time-iterations-per-interval=50 \
            --max-time-iterations=2000 \
            --ignition-delay-time-file=IgnitionDelayTime_cvode.dat \
            --ignition-delay-time-w-threshold-temperature-file=IgnitionDelayTimeTthreshold_cvode.dat
            --threshold-temperature=1500"

echo $this
eval $this
