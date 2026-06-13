## Environment Setup

# SCM_MLDP Research Guide

This guide describes the complete workflow for setting up the environment, generating training data, training the GNN model, running inference, generating ML-guided rules, and conducting comparative experiments for the **SCM_MLDP** project.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Clone the Project](#clone-the-project)
3. [Enter the Project Directory](#enter-the-project-directory)
4. [Generate Training Instances](#generate-training-instances)
5. [Sort DP Recipes](#sort-dp-recipes)
6. [Convert DP Recipes to JSON](#convert-dp-recipes-to-json)
7. [Split Training and Test Data](#split-training-and-test-data)
8. [Train the GNN Model](#train-the-gnn-model)
9. [Generate Test Data](#generate-test-data)
10. [Run Inference](#run-inference)
11. [Generate Good Rules](#generate-good-rules)
12. [Run Comparative Experiments](#run-comparative-experiments)
13. [Expected Output Directories](#expected-output-directories)

---

## Environment Setup

### Prerequisites

Please make sure that either [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed.

### Installation

Create the conda environment from the provided configuration file:

```bash
conda env create -f environment.yml
conda activate <env_name>
```

Replace `<env_name>` with the environment name defined in `environment.yml`.

---

## Clone the Project

Clone the project repository:

```bash
git clone https://github.com/Chufeng-Jiang/SCM_MLDP.git
```

---

## Enter the Project Directory

```bash
cd SCM_MLDP
```

---

## Generate Training Instances

Before generating decomposition recipes, prepare the training constants.

The training constants used in our experiments are provided in:

```bash
./data/training_constants
```

This directory contains **12,767 constants**.

You can download the appropriate Picat version for your operating system from the official Picat website:

```text
https://picat-lang.org/download.html
```

If necessary, replace the provided Picat executable with the version suitable for your operating system.

Run the following commands to generate DP recipes:

```bash
cd Picat
./picat dpmink
```

After execution, **12,767 generated recipes** will be saved in:

```bash
./data/data
```

---

## Sort DP Recipes

```bash
cd ../data
python sort_dp_sol.py
```

This script sorts the generated DP recipes by their target constants in ascending order.

The sorted results are stored in:

```bash
./data/data_sorted
```

---

## Convert DP Recipes to JSON

```bash
python convert_to_json.py
```

This converts the sorted DP recipes into a JSON dataset named:

```bash
dpmink.json
```

This JSON file is used as the main training dataset for the experiments.

---

## Split Training and Test Data

```bash
cd ..
python data_split.py
```

This step creates the training and test splits used by the GNN model.

The split files are saved under:

```bash
./data/split
```

Typical output files include:

```bash
./data/split/train_targets.json
./data/split/test_targets.json
./data/split/train_data.json
./data/split/test_data.json
./data/split/split_info.json
```

---

## Train the GNN Model

Run the training script:

```bash
python train_gnn_simple.py
```

This trains the GNN model for operation prediction.

The trained model checkpoint will be saved under:

```bash
./model_results
```

The main checkpoint file is:

```bash
./model_results/best_model_simple.pth
```

Training logs are saved under:

```bash
./training_history
```

---

## Generate Test Data

First, generate the test constants:

```bash
cd test_numbers
python generate_test_numbers.py
```

Then run the Picat script to generate all candidate decompositions:

```bash
cd ../Picat
./picat dpmink_allsplit.pi
```

Next, process the generated test data:

```bash
cd ../test_numbers
python remove_dup.py
python generate_json.py
```

The processed inference input files will be stored under:

```bash
./test_numbers/inference_input
```

---

## Run Inference

The following command provides an example of how to run inference with the trained GNN model.

You may need to adjust the input range using `--start-c` and `--end-c` until inference has been performed for all target constants.

```bash
python op_inference_simple.py \
  --model ./model_results/best_model_simple.pth \
  --input ./test_numbers/inference_input/17input.json \
  --output ./test_numbers/inference_output/500confidence.csv \
  --start-c 1 \
  --end-c 1730635
```

After generating multiple confidence CSV files, merge them:

```bash
cd ./test_numbers/inference_output
python merge_csv.py
```

The merged confidence results are used to generate ML-guided Picat rules.

---

## Generate Good Rules

Run:

```bash
python generate_goodrules.py
```

This script generates a Picat rule file:

```bash
goodRules.pi
```

Move the generated rule file into the Picat directory:

```bash
mv goodRules.pi ../Picat/goodRules.pi
```

---

## Run Comparative Experiments

Run the following Picat programs to compare the ML-guided method, the baseline method, and the original DP method:

```bash
cd ../Picat

./picat dpmink_ML
./picat baseline
./picat dpmink_DP
```

The three experiment modes are:


| Script      | Description                                                           |
| ----------- | --------------------------------------------------------------------- |
| `dpmink_ML` | Runs the ML-guided decomposition strategy using generated good rules. |
| `baseline`  | Runs the baseline strategy.                                           |
| `dpmink_DP` | Runs the original dynamic programming strategy.                       |

---

## Expected Output Directories

The main output results are stored in:

```bash
./test_numbers/picat_output
```

Other important generated directories include:

```bash
./data/data
./data/data_sorted
./data/split
./model_results
./training_history
./test_numbers/inference_input
./test_numbers/inference_output
```

---

## Overall Workflow Summary

The full experimental pipeline is:

```text
Training constants
        ↓
Picat DP recipe generation
        ↓
Recipe sorting
        ↓
JSON conversion
        ↓
Train/test split
        ↓
GNN model training
        ↓
Test number generation
        ↓
GNN inference
        ↓
Confidence CSV merging
        ↓
Good rule generation
        ↓
Picat comparative experiments
```

This pipeline allows the project to evaluate whether a learned GNN model can guide symbolic SCM decomposition more effectively than baseline or pure dynamic programming strategies.

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Install

```bash
conda env create -f environment.yml
conda activate 
```

## Clone Project

```
https://github.com/Chufeng-Jiang/SCM_MLDP.git
```

## Enter Project Folder

```
cd SCM_MLDP
```

## Generate Training Instance

You should prepare your own training constants. The training constants used in our experiments are provided in ./data/training_constants, including 12767 constants.

```
# You can download the appropriate Picat version for your operating system from https://picat-lang.org/download.html and replace the provided executable.

cd Picat
./picat dpmink
```

After running the above command, 12,767 generated recipes will be saved in the directory ./data/data.

## Sorting DP Recipes

```
cd ../data
python sort_dp_sol.py
```

After execution, the content of the 12,767 generated recipes are sorted based on the target constants in ascending order and stored in ./data/data_sorted.

## Converting DP Recipes to JSON

```
python convert_to_json.py
```

The training data is converted into a JSON file named dpmink.json, which is used for the experiments.

## Splitting Training and Test (Validation) Data

```
cd ..
python data_split.py
```

## Training the Model

```cd
python train_gnn_simple.py 
```

## Generating Test Data

```
cd test_number
python generate_test_numbers.py

cd ../Picat
./picat dpmink_allsplit.pi 

cd ../test_number
python remove_dup.py
python generate_json.py
```

## Running Inference

The following command provides an example of how to run inference. You should adjust the input range as needed until inference has been performed for all target constants.

```
python op_inference_simple.py \
  --model ./model_results/best_model_simple.pth \
  --input ./test_numbers/inference_input/17input.json \
  --output ./test_numbers/inference_output/500confidence.csv \
  --start-c 1 \
  --end-c 1730635

cd ../inference_output
python merge_csv.py
```

## Generating Good Rules

```python
python generate_goodrules.py

mv goodRules.pi ../Picat/goodRules.pi
```

## Running Comparative Experiments

```
./picat dpmink_ML
./picat baseline
./picat dpmink_DP
```

The output results are stored in the directory:

```./test_numbers/picat_output
./test_numbers/picat_output
```
