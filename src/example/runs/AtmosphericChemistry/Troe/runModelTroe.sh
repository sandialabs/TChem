exec=$TCHEM_INSTALL_PATH/example/TChem_AtmosphericChemistry.x

run_this="$exec --chemfile=config_troe/config_troe.yaml \
          --outputfile=Troe.dat \
          --time-iterations-per-interval=50 \
          --tol-time=1e-8 \
          --dtmin=1e-20 \
          --dtmax=1e-1 \
          --tend=100 \
          --atol-newton=1e-18 \
          --rtol-newton=1e-8 \
          --max-newton-iterations=20 \
          --max-time-iterations=200"

echo $run_this
eval $run_this
