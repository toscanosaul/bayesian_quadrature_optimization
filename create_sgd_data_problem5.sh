
problem=problem5
rs=$1
lr=$2
std=$3
n_epochs=$4

python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0 0.1  $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.1 0.2 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.2 0.3 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.3 0.4 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.4 0.5 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.5 0.6 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.7 0.8 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.8 0.9 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 1.0 1.1 $std $lr real_gradient $problem 0

