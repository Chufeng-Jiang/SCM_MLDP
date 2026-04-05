## Environment Setup

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

```cd ..
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

```python generate_goodrules.py
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

