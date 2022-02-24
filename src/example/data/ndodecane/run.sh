exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD.x
inputs=$TCHEM_INSTALL_PATH/example/data/ndodecane
this="$exec --chemfile=$inputs/chem_ndodecane.inp \
            --thermfile=$inputs/therm_ndodecane.dat \
            --samplefile=$inputs/sample.dat \
            --outputfile=IgnSolution.dat \
            --use-cvode=false \
            --atol-newton=1e-18 \
            --rtol-newton=1e-6\
            --max-newton-iterations=20 \
            --tol-time=1e-6 \
            --rtol-time=1e-12 \
            --dtmax=1e-3 \
            --dtmin=1e-20 \
            --tend=0.2 \
            --time-iterations-per-interval=10 \
            --max-time-iterations=1000 \
            --ignition-delay-time-file=IgnitionDelayTime.dat \
            --ignition-delay-time-w-threshold-temperature-file=IgnitionDelayTimeTthreshold.dat
            --threshold-temperature=1500"

echo $this
eval $this
