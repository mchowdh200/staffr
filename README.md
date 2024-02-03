# STAFFR

This script retrieves the probability of a recessive allele and associated p-value from structural variant data.

## Installation

Before running the script, ensure you have the required packages installed. You can use pip to install them as follows:
```sh
pip install argparse pandas numpy json os
```

## Usage
Run the following command after downloading the necessary files

- `cd run/tool/staffr`

Consider the following options for running the script:

- `python staffr.py inputData [-viz] [-p] [-c] [-v] [-m] [-o]`

Where:

- `inputData`: The path to the file containing the input data
- `-viz`: Shows plots, based on data provided
- `-c [float]`: Defines convergence tolerance for EM
- `-v`: Calculates p-value if used
- `-m [.json file]`: Provides the theta file if analysis already ran
- `-o`: Determines if there are outliers, given p-value

For instance, if you want to run the script on a file named `example.csv` and visualize the resulting plot, you can run:

```sh
python staffr.py example.csv -viz
```

## Output

The output file will contain parameters named `pi`, `mu`, `sigma`, `q`, with the corresponding values for each parameter. The output file will be in JSON format, with the same name as the input file, followed by '_output'. The output parameters are calculated by the function `model_hw()` from the `model_hw` module.

If '-v' is used during script execution, the output file will also include a `p_val` field containing the calculated p-value.

## Plotting

If the '-viz' flag is used, a plot is generated based on the `model_hw` function. If '-m' flag is used along with '-viz', the plots are generated based on values from the theta file provided.

## Convergence Tolerance

The `-c` flag followed by a float value helps to define the convergence tolerance for the Expectation-Maximization (EM) algorithm used in `model_hw()` function.

## Outliers

If '-o' flag is used, outlier detection is also performed in the data analysis.