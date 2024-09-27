# STAFFR

This tool is designed to analyze STIX index evidence distributions for structural variant (SV) queries.
It leverages a Hardy-Weinberg (HW) modified Gaussian Mixuture Model and an accompanying non-HW null model
to allow the user to determine whether a given variant in a queried population is in HW equilibrium or not.
If a query is predicted in HW equilibrium, then one can use the model's predicted q-value as an estimated
allele frequency for the variant in the queried population.

## Installation

Before running the script, ensure you have the required packages installed. 

- Python 3.12
- Pandas
- NumPy
- Matplotlib 
- Scipy 
- tqdm

You can use `conda`/`mamba` to install them as follows:
```sh
conda install python=3.12 pandas numpy matplotlib scipy tqdm
```

## Usage

```
usage: staffr.py [-h] [--iteration-limit ITERATION_LIMIT] [-c CONVERGENCE] [-p] [--threads THREADS]
                 --data-column-start DATA_COLUMN_START [--data-column-format DATA_COLUMN_FORMAT]
                 [--acceptance-threshold ACCEPTANCE_THRESHOLD] [--plot-distributions PLOT_DISTRIBUTIONS]
                 [input]

Retrieve probability of alternate allele and associated p-value from structural variant. Input/output data is tab-separated. See args for more details on input format. Output is printed to stdout and follows the format of the input
data.

positional arguments:
  input                 data input

options:
  -h, --help            show this help message and exit
  --iteration-limit ITERATION_LIMIT
                        iteration limit for EM
  -c CONVERGENCE, --convergence CONVERGENCE
                        convergence tolerance for EM
  -p, --p-value         p-value calculation
  --threads THREADS     number of threads to use
  --data-column-start DATA_COLUMN_START
                        data column start (0-based)
  --data-column-format DATA_COLUMN_FORMAT
                        Data column format. Expected to be a string with colon separated fields. This program will
                        expect one of the fields to be "count" and will use that field as the data.
  --acceptance-threshold ACCEPTANCE_THRESHOLD
                        minimum depth of stix evidence to consider as non-zero
  --plot-distributions PLOT_DISTRIBUTIONS
                        plots distributions in path specified as argument
```

### Examples
- TODO


6. If you want to visualize the null model with the data: 
- TODO add arg for plotting null model

## Output
- TODO
