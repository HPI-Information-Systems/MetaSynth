import glob
import json


# List to store the contents of the JSON files
configs = []
names = []

# Iterate over the matching files
for file_path in glob.glob('cardio*'):
    names.append(file_path.split("cardio")[1].split(".json")[0])
    with open(file_path) as file:
        json_data = json.load(file)
        configs.append(json_data)


datasets = ["student-performance", "weather", "heart-failure", "crowdfunding", "gaming", "abalone", "flight-price", "insurance", "housing"]
targets = ["GradeClass", "Weather_Type", "HeartDisease", "IsSuccessful", "EngagementLevel", "Class_number_of_rings", "price", "charges",  "ocean_proximity"]
tasks = ["classification"] * 5 + ["regression"] * 5

for dataset, target, task in zip(datasets, targets, tasks):
    for name, config in zip(names, configs):
        config["dataset_name"] = dataset
        config["target"] = target
        config["task"] = task
        with open(f"{dataset}{name}.json", "w") as file:
            json.dump(config, file, indent=4)