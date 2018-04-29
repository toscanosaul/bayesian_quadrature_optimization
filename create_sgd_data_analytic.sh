
problem=$1
rs=$2
lr=$3
for i in {0..5}
do
    lower=$((10**$i))
    next=$(($i+1))
    upper=$((10**$next))
    echo $lower
    echo $upper
    python -m multi_start.problems.sgd_specific_function $rs 1 100 $lower $upper 1 $lr real_gradient $problem 1
    python -m multi_start.problems.sgd_specific_function $rs 1 100 $lower $upper 1 $lr grad_epoch $problem 1
done

python -m multi_start.problems.sgd_specific_function $rs 1 100 0.01 0.1 1 $lr real_gradient $problem 1
python -m multi_start.problems.sgd_specific_function $rs 1 100 0.01 0.1 1 $lr grad_epoch $problem 1
python -m multi_start.problems.sgd_specific_function $rs 1 100 0.0 1.0 1 $lr real_gradient $problem 1
python -m multi_start.problems.sgd_specific_function $rs 1 100 0.0 1.0 1 $lr grad_epoch $problem 1
python -m multi_start.problems.sgd_specific_function $rs 1 100 0.1 0.0 1 $lr real_gradient $problem 1
python -m multi_start.problems.sgd_specific_function $rs 1 100 0.1 0.0 1 $lr grad_epoch $problem 1
