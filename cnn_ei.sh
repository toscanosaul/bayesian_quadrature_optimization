export PYTHONPATH="/fs/home/st684/Documents/stratified_bayesian_optimization/"
export PATH="/fs/home/st684/anaconda2/bin:$PATH"
python -m scripts.run_multiple_spec cnn_ei.json $SEED
