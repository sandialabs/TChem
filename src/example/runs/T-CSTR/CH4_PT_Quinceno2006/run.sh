exec=$TCHEM_INSTALL_PATH/example/TChem_TransientContStirredTankReactor.x

run_this="$exec --chemfile=inputs/chemgri30.inp \
                --thermfile=inputs/thermgri30.dat \
                --samplefile=inputs/sample_phi1.dat \
                --surf-chemfile=inputs/chemSurf.inp \
                --surf-thermfile=inputs/thermSurf.dat \
                --surf-inputfile=inputs/inputSurf.dat \
                --outputfile=CSTRSolutionODE.dat \
                --catalytic-area=1.347e-2 \
                --reactor-volume=1.347e-1 \
                --inlet-mass-flow=1e-2 \
                --number-of-algebraic-constraints=0 \
                --transient-initial-condition=true \
                --max-newton-iterations=20 \
                --atol-newton=1e-18 \
                --rtol-newton=1e-8 \
                --tol-time=1e-8 \
                --atol-newton=1e-12 \
                --dtmin=1e-20 \
                --dtmax=1e-3 \
                --tend=3 \
                --time-iterations-per-interval=10\
                --max-z-iterations=400 "

echo $run_this
eval $run_this
