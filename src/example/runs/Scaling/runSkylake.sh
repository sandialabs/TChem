export TCHEM_INSTALL_PATH=${PWD}/../../../
dirs=(H2 CO GRI3.0 ndodecane ThinkMech n-butanol)

for dir in ${dirs[@]}
do
    run_this="cd $dir;./runSkylake.sh;cd -"
    echo $run_this
    eval $run_this
done
