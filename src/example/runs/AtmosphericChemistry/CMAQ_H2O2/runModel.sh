exec=$TCHEM_INSTALL_PATH/example/TChem_AtmosphericChemistry.x

run_this="$exec --chemfile=config_cmaq_h2o2.yaml \
          --outputfile=CMAQ_H2O2.dat \
          --time-iterations-per-interval=10 \
          --tol-time=1e-6 \
          --dtmin=1e-20 \
          --dtmax=1e-1 \
          --tend=100 \
          --atol-newton=1e-18 \
          --rtol-newton=1e-8 \
          --max-newton-iterations=20 \
          --max-time-iterations=200"

echo $run_this
eval $run_this
