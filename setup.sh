module purge

module load cuda/10.1
module load python/anaconda3.6-5.2.0

source activate cbio3.6

unset PYTHONPATH
unset PYTHONHOME
