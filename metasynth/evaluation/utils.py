import pandas as pd
from sklearn.utils import resample

def benchmark(benchmarks, inputs, gt):
    groups = list({name.split("-")[-1] for name, _ in benchmarks})
    scores = {model: {name: [] for name in groups} for model in inputs.keys()}
    for model, df in inputs.items():
        for name, benchmark in benchmarks:
            print(model, name)
            group = name.split("-")[-1]
            try:
                score = benchmark.evaluate_default(gt, df)
                scores[model][group].append(score)
            except ValueError as e:
                print(f"Error: {e}")
                with open("errors.txt", "a+") as file:
                    file.write("Model: "+model + "\n")
                    file.write("Metric: "+name + "\n")
                    file.write(str(e) + "\n\n")
                scores[model][group].append(-1)

    return scores

def balance_dataframe(gen_df, orig_df, target_column):
    # Calculate the class distribution in the original dataframe
    class_distribution = orig_df[target_column].value_counts(normalize=True)
    
    # Initialize an empty dataframe for the balanced dataset
    balanced_df = pd.DataFrame(columns=gen_df.columns)
    
    # Resample each class in the generated dataframe to match the original distribution
    for cls, proportion in class_distribution.items():
        # Number of samples for the current class in the balanced dataset
        num_samples = int(proportion * len(gen_df))
        
        # Get all samples of the current class from the generated dataframe
        class_samples = gen_df[gen_df[target_column] == cls]
        
        # If there are not enough samples in the generated dataframe, resample with replacement
        if len(class_samples) < num_samples:
            class_samples = resample(class_samples, replace=True, n_samples=num_samples, random_state=42)
        # Otherwise, sample without replacement
        else:
            class_samples = class_samples.sample(n=num_samples, random_state=42)
        
        # Append the resampled class samples to the balanced dataframe
        balanced_df = pd.concat([balanced_df, class_samples])
    
    # Shuffle the balanced dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

def balance_dataframe_equal_classes(gen_df, target_column):
    # Calculate the number of samples in each class
    class_counts = gen_df[target_column].value_counts()
    
    # Identify the minimum class count
    min_class_count = class_counts.min()
    
    # Initialize an empty dataframe for the balanced dataset
    balanced_df = pd.DataFrame(columns=gen_df.columns)
    
    # Resample each class to have the same number of samples as the minimum class count
    for cls in class_counts.index:
        # Get all samples of the current class from the generated dataframe
        class_samples = gen_df[gen_df[target_column] == cls]
        
        # Sample without replacement to get the minimum class count samples
        class_samples = class_samples.sample(n=min_class_count, random_state=42)
        
        # Append the resampled class samples to the balanced dataframe
        balanced_df = pd.concat([balanced_df, class_samples])
    
    # Shuffle the balanced dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df