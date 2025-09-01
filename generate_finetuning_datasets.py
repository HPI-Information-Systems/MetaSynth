from metasynth.generators.metasynth_columnwise_json import MetaSynth
import json
import pandas as pd
import numpy as np
from metasynth.processing.utils import preprocess_df
from datasets import load_dataset, concatenate_datasets
import copy
import random
import os
from datasets import Dataset, DatasetDict
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description="Generate finetuning datasets for MetaSynth.")
parser.add_argument('--llm_name', type=str, default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", help='Path to the LLM model.')
parser.add_argument('--base_path', type=str, default="data/", help='Base path to the datasets.')
parser.add_argument('--save_path', type=str, default="finetuning_data/", help='Path to save the finetuning data.')
parser.add_argument('--datasets', type=str, nargs='+', default=["abalone", "cardio", "crowdfunding", "flight-price", "gaming", "heart-failure", "housing", "insurance", "student-performance", "weather"], help='List of datasets to process.')
parser.add_argument('--num_examples', type=str, default="50", help='Number of examples to use (int or "all").')

args = parser.parse_args()

llm_name = args.llm_name
base_path = args.base_path
save_path = args.save_path
datasets = args.datasets
num_examples = args.num_examples


# Convert num_examples to int if not "all"
if num_examples != "all":
    num_examples = int(num_examples)

num_conversations = 250 if (num_examples != "all" and num_examples < 75) else 500 if num_examples != "all" else 750
num_examples = num_examples if num_examples != "all" else 1000000

def get_random_generation_groups(synth):
    previous_colums = []
    groups = []
    group = []
    columns = [col for col in synth.metadata.keys() if '_int' not in col]
    random.shuffle(columns)
    while len(previous_colums) < len(columns):
        for col in columns:
            if col in previous_colums or len(group) >= 8:
                continue
            if not any([c in synth.metadata[col]["correlations"] for c in group]):
                group.append(col)
                previous_colums.append(col)
                
        groups.append(group)
        group = []
    return groups

for dataset in datasets:
    eval_data = {
        "messages": [],
    }

    data = {
        "messages": [],
    }
    
    print(dataset)
    with open(f"{base_path}/{dataset}/descriptions.json", 'r') as f:
        DESCRIPTIONS = json.load(f)
        target = [key for key in DESCRIPTIONS.keys() if "Role: Target" in DESCRIPTIONS[key]][0]

    with open(f"{base_path}/{dataset}/types.json", 'r') as f:
        TYPES = json.load(f)
        
    df = pd.read_csv(f"{base_path}/{dataset}/{dataset}.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    df.dropna(inplace=True)
    if TYPES[target] == "str":
        df[target] = df[target].astype(str)
        
    processed_df, metadata = preprocess_df(copy.deepcopy(df), DESCRIPTIONS, TYPES)

    synth = MetaSynth(
        metadata=metadata,
        types=TYPES,
        llm=None,
        llm_name=llm_name,
        tokenizer_name=llm_name
    )
    synth.fit(processed_df)

    for response_num in tqdm(range(num_conversations)):
        #randomly dropping columns
        num_columns_to_drop = random.randint(0, int(len(df.columns)/2))
        columns_to_drop = random.sample([col for col in df.columns if col != target], num_columns_to_drop)
        #columns_to_drop = []
        
        df_dropped = df.drop(columns=columns_to_drop)
        current_description = {key: value for key, value in DESCRIPTIONS.items() if key not in columns_to_drop}
        current_types = {key: value for key, value in TYPES.items() if key not in columns_to_drop}
        synth.types = current_types
        processed_df, metadata = preprocess_df(df_dropped.sample(min(500, int(len(df_dropped) * 0.75))).reset_index(drop=True), current_description, current_types)
        synth.metadata = metadata
        synth.df = processed_df
        
        #Randomly choose generation groups
        groups = get_random_generation_groups(synth)
        
        columns = [col for group in groups for col in group]
        previous_colums = []

        # TRY changing back to 20 - 25
        good_df = copy.deepcopy(df_dropped[:num_examples].sample(random.randint(20, 25)).reset_index(drop=True))
        
        # Add small random noise
        for col in TYPES.keys():
            if TYPES[col] == "float" and col in good_df.columns:
                try:
                    num_digits = len(str(good_df[col][0]).split(".")[1])
                except:
                    num_digits = 0
                good_df[col] = round(good_df[col] + np.random.normal(0, metadata[col]["std"] * 0.05, len(good_df)), num_digits)
        
        pos_chat = copy.deepcopy(synth.chat)
        
        for idx, group in enumerate(groups):
            previous_colums.extend(group)
            
            if idx == 0:
                call = synth.CALL_FIRST if len(group) > 1 else synth.CALL_CAT_FIRST if "unique" in synth.metadata[group[0]] else synth.CALL_CONT_FIRST
            else:
                call = synth.CALL if len(group) > 1 else synth.CALL_CAT if "unique" in synth.metadata[group[0]] else synth.CALL_CONT
                
            introduction = synth.COLUMNS_INTRODUCTION if len(group) > 1 else synth.COLUMN_INTRODUCTION
            example = str({col: "<placeholder>" for col in previous_colums})
                
            if len(pos_chat) == 5:
                pos_chat = [pos_chat[0], pos_chat[3], pos_chat[4]]
                
            pos_chat = pos_chat + [
                    {
                        "role": "user", 
                        "content": introduction.format(column_names=", ".join(group))+"\n".join([synth.get_metadata(col, previous_cols=previous_colums) for col in group])+"\n"+call.format(column_names=", ".join(group))+example
                    }
                ]
                
            pos_response = "\n".join(json.dumps(dictionary, indent=4) for dictionary in good_df[previous_colums].to_dict(orient='records'))
            pos_chat = pos_chat + [{"role": "assistant", "content": pos_response}]


            if response_num % 5 == 0:
                eval_data["messages"].append(pos_chat)

            else:
                data["messages"].append(pos_chat)
                
    # Step 1: Create a Dataset from the dictionary
    ds = Dataset.from_dict(data)

    # Create the directory if it doesn't exist
    os.makedirs(f"{save_path}/{num_examples}/{dataset}", exist_ok=True)
    # Step 2: Save the dataset to disk
    ds.save_to_disk(f'{save_path}/{num_examples}/{dataset}/train')

    # Step 1: Create a Dataset from the dictionary
    ds = Dataset.from_dict(eval_data)

    # Step 2: Save the dataset to disk
    ds.save_to_disk(f'{save_path}/{num_examples}/{dataset}/eval')
