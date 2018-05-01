problem=$1
lw=$2
up=$3
n_restarts=$4
lr=$5
std=$6

for i in {$lw..$up}
do
    screen -d -m python -m multi_start.script_run_policies_from_beginning $i $n_restarts $std $lr approx_lipschitz 0 100 $problem
done
