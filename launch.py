
import argparse
import sys
import os
import subprocess
import tempfile


RUN_DIR = '/zfs1/jdurrant/durrantlab/hag63/leadopt_pytorch'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t','--time',default='10:00:00')
    parser.add_argument('-p','--partition',default='gtx1080')
    parser.add_argument('-m','--mem',default='16g')
    parser.add_argument('path')
    parser.add_argument('script')
    
    args = parser.parse_args()

    run_path = os.path.join(RUN_DIR, args.path)
    if os.path.exists(run_path):
        print('[!] Run exists at %s' % run_path)
        overwrite = input('- Overwrite? [Y/n]: ')
        if overwrite.lower() == 'n':
            print('Exiting...')
            exit(0)
    else:
        print('[*] Creating run directory %s' % run_path)
        os.mkdir(run_path)
    
    script = '''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={run_path}/slurm_out.txt
#SBATCH --error={run_path}/slurm_err.txt
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cluster=gpu
#SBATCH --partition={partition}
#SBATCH --mail-user=hag63@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}

cd /ihome/jdurrant/hag63/cbio/leadopt_pytorch/
./setup.sh

export PYTHONPATH=/ihome/jdurrant/hag63/cbio/leadopt_pytorch/
# export WANDB_DIR=/ihome/jdurrant/hag63/wandb_abs
export WANDB_DISABLE_CODE=true

cd {run_path}
python /ihome/jdurrant/hag63/cbio/leadopt_pytorch/{script}
    '''.format(
        name='leadopt_%s' % args.path,
        run_path=run_path,
        time=args.time,
        partition=args.partition,
        mem=args.mem,
        script=args.script
    )

    print('[*] Running script...')
    
    with tempfile.NamedTemporaryFile('w') as f:
        f.write(script)
        f.flush()
        
        r = subprocess.run('sbatch %s' % f.name, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        print(r)



if __name__=='__main__':
    main()
