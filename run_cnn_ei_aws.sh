END=$2
BEGIN=$1

for i in $(seq $BEGIN $END); do qsub -cwd -v SEED=$i cnn_ei.sh; done
