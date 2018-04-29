
problem=problem6
rs=$1
lr=$2
std=$3
n_epochs=$4

python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0 0.5  $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 0.5 1.0 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 1.0 1.5 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 1.5 2.0 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs 2.0 2.5 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs -0.5 0.0 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs -1.0 -0.5 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs -1.5 -1.0 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs -2.0 -1.5 $std $lr real_gradient $problem 0
python -m multi_start.problems.sgd_specific_function $rs 1 $n_epochs -2.5 -2.0 $std $lr real_gradient $problem 0

