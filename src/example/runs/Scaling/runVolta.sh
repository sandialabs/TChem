export TCHEM_INSTALL_PATH=${PWD}/../../../
dirs=(H2 CO GRI3.0 ndodecane ThinkMech n-butanol)
exes=(runVoltaAnalytic.sh  runVoltaNumFwd.sh  runVoltaSacado.sh  runVoltaSource.sh)
for dir in ${dirs[@]}
do
    for exe in ${exes[@]}
    do
        run_this="cd $dir;./$exe;cd -"
        echo $run_this
        eval $run_this
    done
done
