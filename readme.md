## Environment Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Install
```bash
conda env create -f environment.yml
conda activate 
```

cd project root

cd Picat
./picat dpmink

# preprocessing the DP solution
cd ../data1
python sort_dp_sol.py

# generate the training set
python convert_to_json.py

cd ..
# split the train data and test(validate) data
python data_split.py

#Traing the model
cd ..
python train_gnn_simple.py 


# inference

# 生成测试数据
cd test_number
python genetate_test_numbers.py

cd ../Picat
./picat dpmink_allsplit.pi 

cd ../test_number
python remove_dup.py
python generate_json.py 

# 开始推理, 这里是一个示例，需要自己调整所有的input范围直到所有常数都进行了推理

python op_inference_simple.py --model ./model_results/best_model_simple.pth --input ./test_numbers/inference_input/17input.json --output ./test_numbers/inference_output/500confidence.csv --start-c 1 --end-c 1730635


cd ../inference_output
python merge_csv.py

# generate good rules
python generate_goodrules.py

# 移动goodrules文件
mv goodRules.pi ../Picat/goodRules.pi

./picat dpmink_ML
./picat baseline
./picat dpmink_DP


