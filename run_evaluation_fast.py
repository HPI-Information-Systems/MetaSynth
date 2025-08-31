import argparse
import copy
import json
import os
import warnings
import time
warnings.filterwarnings('ignore')

import pandas as pd
from metasynth.evaluation.mle_evaluator import MLEEvaluator
from metasynth.evaluation.stat_evaluator import CorrelationPerformanceEvaluator, MixedCorrelationPerformanceEvaluator
from metasynth.evaluation.privacy_evaluator import PrivacyEvaluator, DCREvaluator, DublicateEvaluator
from metasynth.evaluation.utils import benchmark
from metasynth.processing.utils import preprocess_df
from metasynth.processing.mapper import Mapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (BayesianRidge, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from metasynth.generators.dummy_sampler import MarginalSampler, UniformSampler


def main(config, llm, client):
    if config["generator"] == "relationships":
        from metasynth.generators.metasynth_basic_better_relationships import MetaSynth
    elif config["generator"] == "columnwise_relationships_json":
        from metasynth.generators.metasynth_columnwise_json import MetaSynth
    elif config["generator"] == "columnwise_relationships_json_nodist":
        from metasynth.generators.metasynth_columnwise_json_nodist import MetaSynth
    elif config["generator"] == "json_relationships":
        from metasynth.generators.metasynth_basic_json_relationships import MetaSynth
    elif config["generator"] == "basic_bettercat":
        from metasynth.generators.metasynth_basic_better_categorical import MetaSynth
    else:
        from metasynth.generators.metasynth_basic import MetaSynth
    
    # Step 1: Load the ground truth data
    save_dir = os.path.join(config["results_root"], config["display_name"], config["dataset_name"])
    os.makedirs(save_dir, exist_ok=True)

    with open(config["descriptions_file"], 'r') as f:
        DESCRIPTIONS = json.load(f)
        
    with open(config["descriptions_file"].replace("descriptions", "types"), 'r') as f:
        TYPES = json.load(f)

    df = pd.read_csv(config["original_file"]).sample(frac=1, random_state=42)
    df.dropna(inplace=True)
    _, metadata = preprocess_df(copy.deepcopy(df), DESCRIPTIONS, TYPES)
    cat_columns = [key for key in metadata.keys() if "_int" not in key and "unique" in metadata[key]]
    cat_columns = [value if value != config["target"] else "target" for value in cat_columns]
    continuous_columns = [key for key in metadata.keys() if "_int" not in key and "unique" not in metadata[key]]
    continuous_columns = [value if value != config["target"] else "target" for value in continuous_columns]
    mapper = Mapper(metadata)
    processed_df = mapper.process(df)
    
    config["num_generations"] = min(config["num_generations"], int(len(processed_df)*0.75))
    config["num_test"] = min(config["num_test"], int(len(processed_df)*0.25))
    
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    gt_train, gt_test = processed_df[:config["num_generations"]] , processed_df[config["num_generations"]:config["num_generations"]+config["num_test"]]

    # Step 2: Load and fit Metasynth
    metadata_generator = MetaSynth(
            metadata=metadata, 
            types=TYPES, 
            llm=llm, 
            llm_name=config["llm_name"], 
            tokenizer_name=config.get("tokenizer_name", config["llm_name"]), 
            stats=config["stats"], nodesc=config["no_descriptions"], 
            nostat=config["no_stats"], 
            temperature=config.get("temperature", None), 
            top_p=config.get("top_p", None), 
            top_k=config.get("top_k", None), 
            client=client
        )

    print(gt_train.head())
    metadata_generator.fit(gt_train)

    # Step 3: Generate synthetic data (It is faster to generate all repetitions at once. The generations are still independent of each other)
    generation_start_time = time.time()
    generated_metadata_all = metadata_generator.generate(count=config["num_generations"] * config["num_repetitions"]).sample(frac=1, random_state=42).reset_index(drop=True)
    generation_end_time = time.time()
    print(f"Generation took {generation_end_time - generation_start_time} seconds")
    
    # Step 4: Prepare the data for standardized evaluation 
    gt_train, gt_test = gt_train.rename(columns={config["target"]: 'target'}), gt_test.rename(columns={config["target"]: 'target'})
    scaler = StandardScaler()
    scaler.fit(gt_train)
    results = {}
    
    # Step 5: Evaluation
    for repetition in tqdm(range(config["num_repetitions"])):

        # Step 5.1: Save & Postprocess our synthetic data
        generated_metadata = generated_metadata_all[repetition*config["num_generations"]:(repetition+1)*config["num_generations"]].reset_index(drop=True)
        generated_metadata.to_csv(os.path.join(save_dir, f"generated_{repetition}.csv"), index=False)
        generated_metadata = generated_metadata.rename(columns={config["target"]: 'target'})[gt_train.columns.tolist()]
        nan_counts = generated_metadata.isnull().sum()
        if nan_counts.sum() > 0:
            print(nan_counts)
        generated_metadata.dropna(inplace=True)

        # Step 5.2: Add all MLE metrics
        benchmarks = []
        if config["task"] == "classification":
            benchmarks = benchmarks + [
                ("LinearSVC-F1", MLEEvaluator(model=LinearSVC(random_state=42), metric=f1_score)),
                ("DecisionTreeClassifier-F1", MLEEvaluator(model=DecisionTreeClassifier(max_depth=5, random_state=42), metric=f1_score)),
                ("RandomForestClassifier-F1", MLEEvaluator(model=RandomForestClassifier(max_depth=5, random_state=42), metric=f1_score)),
                ("LogisticRegression-F1", MLEEvaluator(model=LogisticRegression(random_state=42), metric=f1_score)),
                ("MLPClassifier-F1", MLEEvaluator(model=MLPClassifier(random_state=42, max_iter=300), metric=f1_score)),
                ("LinearSVC-Accuracy", MLEEvaluator(model=LinearSVC(random_state=42), metric=accuracy_score)),
                ("DecisionTreeClassifier-Accuracy", MLEEvaluator(model=DecisionTreeClassifier(max_depth=5, random_state=42), metric=accuracy_score)),
                ("RandomForestClassifier-Accuracy", MLEEvaluator(model=RandomForestClassifier(max_depth=5, random_state=42), metric=accuracy_score)),
                ("LogisticRegression-Accuracy", MLEEvaluator(model=LogisticRegression(random_state=42), metric=accuracy_score)),
                ("MLPClassifier-Accuracy", MLEEvaluator(model=MLPClassifier(random_state=42, max_iter=300), metric=accuracy_score))
            ]
        else:
            benchmarks = benchmarks + [
                ("LinearRegression-MSE", MLEEvaluator(model=LinearRegression(), metric=mean_squared_error, scaler=scaler)),
                ("Ridge-MSE", MLEEvaluator(model=Ridge(), metric=mean_squared_error, scaler=scaler)),
                ("Lasso-MSE", MLEEvaluator(model=Lasso(), metric=mean_squared_error, scaler=scaler)),
                ("BayesianRidge-MSE", MLEEvaluator(model=BayesianRidge(), metric=mean_squared_error, scaler=scaler))
            ]

        # Step 5.2: Add all other metrics
        benchmarks = benchmarks + [
            ("CorrelationMatricDifference-Corr", CorrelationPerformanceEvaluator()), 
            ("PrivacyEvaluator-DCR", PrivacyEvaluator()),
            ("PrivacyEvaluator-DCR0", DCREvaluator()),
            ("PrivacyEvaluator-Dub", DublicateEvaluator()),
        ]

        # Step 5.3: Run all metrics
        inputs = {
                config["display_name"]: generated_metadata[gt_test.columns],
                "GT": gt_train[gt_test.columns]
            }
        result = benchmark(
            benchmarks=benchmarks,
            inputs=inputs,
            gt=(gt_train[gt_test.columns], gt_test)
        )
        
        for model, groups in result.items():
            if model not in results:
                results[model] = {group: [] for group in groups.keys()}
            for group, scores in groups.items():
                results[model][group].extend(scores)
                
    # Step 6: Save the results as JSON
    results_file = os.path.join(save_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run synthetic data generation and evaluation.")
    parser.add_argument('--config_files', type=str, nargs='+', required=True, help="Path to the JSON configuration file.")
    
    args = parser.parse_args()
    
    llm_name = None
    llm = None
    client = None
    
    for file in args.config_files:
        # Read the configuration file
        print(file)
        with open(file, 'r') as f:
            config = json.load(f)
        
        if "base_url" in config:
            if config["llm_name"] != llm_name:
                from openai import AsyncOpenAI
                llm_name = config["llm_name"]
                client = AsyncOpenAI(
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                )       
        else:
            if config["llm_name"] != llm_name:
                from vllm import LLM
                import torch
                llm_name = config["llm_name"]
                else:
                    llm = LLM(model=llm_name)

        # Create absolute paths for the input files
        config["descriptions_file"] = os.path.join(config["data_root"], config["dataset_name"], config["descriptions_file"])
        config["original_file"] = os.path.join(config["data_root"], config["dataset_name"], config["original_file"])
        
        main(config, llm, client)
