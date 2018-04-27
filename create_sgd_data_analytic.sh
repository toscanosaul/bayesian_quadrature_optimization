
for i in {0..5}
do
    lower=$((10**$i))
    next=$(($i+1))
    upper=$((10**$next))
    echo $lower
    echo $upper
    python -m multi_start.problems.sgd_specific_function 1540 1 100 $lower $upper 1 10.0 real_gradient
    python -m multi_start.problems.sgd_specific_function 1540 1 100 $lower $upper 1 10.0 grad_epoch
done

python -m multi_start.problems.sgd_specific_function 1540 1 100 0.1 1.0 1 10.0 real_gradient
python -m multi_start.problems.sgd_specific_function 1540 1 100 0.01 0.1 1 10.0 grad_epoch
python -m multi_start.problems.sgd_specific_function 1540 1 100 0.1 1.0 1 10.0 real_gradient
python -m multi_start.problems.sgd_specific_function 1540 1 100 0.1 1.0 1 10.0 grad_epoch
python -m multi_start.problems.sgd_specific_function 1540 1 100 0.001 0.01 1 10.0 real_gradient
python -m multi_start.problems.sgd_specific_function 1540 1 100 0.001 0.01 1 10.0 grad_epoch
