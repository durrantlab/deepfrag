
# Data processing scripts

## `process_pdbbind.py`

Used to generate compressed atomic coordinate arrays from pdbbind .pdb and .sdf files.

Usage:

```
usage: process_pdbbind.py [-h] -d DATASETS [DATASETS ...] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASETS [DATASETS ...], --datasets DATASETS [DATASETS ...]
                        List of dataset folders to include
  -o OUTPUT, --output OUTPUT
                        Output file path (.h5)
```

Example:
```
$ python scripts/process_pdbbind.py -d /path/to/v2018-other-PL /path/to/refined-set -o /path/to/fragments.h5
```

## `make_fingerprints.py`

Utility script to generate fingerprints for a set of smiles strings. By precomputing the fingerprints for all the fragments in our dataset, we can speed up training.

To use, pass in the `fragments.h5` file created by the `process_pdbbind.py` script and specify the fingerprint type and output path.

Supported fingerprints:
- `rdk`: RDKFingerprint (2048 bits)
- `morgan`: Mogan fingerprint (r=2) (2048 bits)
- `gobbi2d`: Gobbi 2d pharmophocore fingerprint (folded to 2048 bits)

Usage:

```
usage: make_fingerprints.py [-h] -f FRAGMENTS -fp {rdk,morgan,gobbi2d}
                            [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -f FRAGMENTS, --fragments FRAGMENTS
                        Path to fragemnts.h5 containing "frag_smiles" array
  -fp {rdk,morgan,gobbi2d}, --fingerprint {rdk,morgan,gobbi2d}
                        Which fingerprint type to generate
  -o OUTPUT, --output OUTPUT
                        Output file path (.h5)
```

