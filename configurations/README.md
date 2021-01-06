This folder contains benchmark model configurations referenced in the paper.

Overview:
- `layer_type_sweep/*`: experimenting with different parent/receptor typing schemes
- `voxelation_sweep/*`: experimenting with different voxelation types and atomic influence radii
- `final.json`: final production model

You can train new models using these configurations with the `train.py` script:

```sh
python train.py \
    --save_path=/path/to/model \
    --wandb_project=my_project \
    --configuration=./configurations/model.json
```

Note: these configuration files assume the working directory is the `leadopt` base directory and that the data directory is accessible at `./data`.
