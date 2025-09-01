# MetaSynth
Metadata-based Synthesis of Realistic Tabular Data using Large Language Models

## How to run MetaSynth

### 0. Installation

First, install pytorch by following the instructions [here](https://pytorch.org/get-started/locally/)
Afterwards you can install the requirements from the requirements.txt file.

### 1. Prepare the data

Every dataset in the data/ folder has either a ```download.py```or a ```process.py```. In case of downloading, you just need to run ```python download.py``` in the dataset directory. 
In the other case, please visit the URL from the ```source.txt```, download the dataset from there and put it as ```original.csv```in the dataset folder and then run ```python process.py```.

### 2. Run MetaSynth

Run MetaSynth either with ```bash run_metasynth.sh``` or directly by:
````python run_evaluation_fast.py --config_files $(ls configs/MetaSynth/*.json````

## Add your own dataset
### Add your data
Place your dataset as a CSV file in a folder. If you follow our data structure, you would need to create a new folder in the ```data/``` folder and place your CSV inside. 
In the same folder, create a ```descriptions.json``` file and a ```types.json``` file. Both files are expected to contain a dictionary with the column names as found in the CSV as keys.
In the descripitions file, the values should be short textual descriptions of the individual columns to use as metadata.
In the types file, the keys sould be either "int" for integer colums, "float" for coninuous numerical columns or "str" for categorical columns.

## Finetuning MetaSynth-X

### 1. Generate the finetuning conversations
```bash generate_finetuning_data.sh```

### 2. Finetune the LLM
```bash finetune_metasynth50.sh```

Based on the finetuned LLMs, you now need to create new config files for your finetuned versions. For this, update the ```"llm_name"``` in the config files with the path to the folder of the finetuned model.