export PYTHONPATH="/home/st684/bayesian_quadrature_optimization"
export PATH="/home/st684/anaconda2/bin:$PATH"
export PYTHONPATH="/home/st684/OpenAeroStruct"
SEED=$1
python -m scripts.run_multiple_spec aircraft_sde.json $SEED

