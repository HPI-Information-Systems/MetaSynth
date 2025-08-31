# MetaSynth
Metadata-based Synthesis of Realistic Tabular Data using Large Language Models

## How to run MetaSynth

### 1. Prepare the data

Every dataset in the data/ folder has either a ```download.py```or a ```process.py```. In case of downloading, you just need to run ```python download.py``` in the dataset directory. 
In the other case, please visit the URL from the ```source.txt```, download the dataset from there and put it as ```original.csv```in the dataset folder and then run ```python process.py```.

### 2. Run MetaSynth

Run MetaSynth either with ```bash run_metasynth.sh``` or directly by:
````python run_evaluation_fast.py --config_files $(ls configs/MetaSynth/*.json````

## Finetuning MetaSynth-X

### 1. Generate the finetuning conversations
```bash generate_finetuning_data.sh```

### 2. Finetune the LLM
```bash finetune_metasynth50.sh```

Based on the finetuned LLMs, you now need to create new config files for your finetuned versions. For this, update the ```"llm_name"``` in the config files with the path to the folder of the finetuned model.