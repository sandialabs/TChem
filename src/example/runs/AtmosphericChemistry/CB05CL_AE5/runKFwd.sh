TCHEM_INSTALL_PATH=${HOME}/Documents/CODE/Getz/install/tchem
exec=$TCHEM_INSTALL_PATH/example/TChem_NetProductionRatePerMassNCAR.x

run_this="$exec --chemfile=config_full_gas.yaml \
          --outputfile=kfwd.dat" 

echo $run_this
eval $run_this
