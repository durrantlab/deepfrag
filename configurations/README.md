This folder contains benchmark model configurations referenced in the paper.

You can retrain from these configurations using the `train.py` script:

```sh
python train.py \
    --save_path=/path/to/model \
    --wandb_project=my_project \
    --configuration=./configurations/model.json
```

Note: these configuration files assume the working directory is the `leadopt` base directory and that the data directory is accessible at `./data`.
