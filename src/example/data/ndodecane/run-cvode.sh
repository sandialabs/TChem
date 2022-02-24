exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD.x
inputs=$TCHEM_INSTALL_PATH/example/data/ndodecane
this="$exec --chemfile=$inputs/chem_ndodecane.inp \
            --thermfile=$inputs/therm_ndodecane.dat \
            --samplefile=$inputs/sample.dat \
            --outputfile=IgnSolution_cvode.dat \
            --use-cvode=true \
            --atol-newton=1e-18 \
            --rtol-newton=1e-4\
            --max-newton-iterations=20 \
            --tol-time=1e-8 \
            --atol-time=1e-18 \
            --dtmax=1e-3 \
            --dtmin=1e-20 \
            --tend=0.2 \
            --time-iterations-per-interval=20 \
            --max-time-iterations=5000 \
            --ignition-delay-time-file=IgnitionDelayTime.dat \
            --ignition-delay-time-w-threshold-temperature-file=IgnitionDelayTimeTthreshold.dat
            --threshold-temperature=1500"

echo $this
eval $this
