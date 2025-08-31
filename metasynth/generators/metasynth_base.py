from typing import Dict, List

import pandas as pd
from transformers import AutoTokenizer
import math


class MetaSynthBase():
    def __init__(self, metadata: Dict, types: Dict, llm=None, client=None, llm_name: str = "", tokenizer_name: str = "", stats: List[str] = ["min", "max", "mean", "std", "25%", "50%", "75%", "skew", "kurtiosis"], nostat: bool = False, nodesc: bool = False, temperature=None, top_p=None, top_k=None) -> None:
        self.metadata = metadata
        self.llm = llm
        self.client = client
        self.temperature = temperature
        if llm is not None:
            from vllm import SamplingParams
            if temperature == None:
                self.sampling_params = SamplingParams(max_tokens=2048, min_tokens=256) #SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048, min_tokens=1024, logprobs=1)
            elif top_p is not None and top_k is not None:
                self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=2048, min_tokens=1024)
            elif top_p is not None:
                self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=2048, min_tokens=1024)
            elif top_k is not None:
                self.sampling_params = SamplingParams(temperature=temperature, top_k=top_k, max_tokens=2048, min_tokens=1024)
            else:
                self.sampling_params = SamplingParams(temperature=temperature, max_tokens=2048, min_tokens=1024)

        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
                
        if client is not None:
            self.sampling_params = None
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.stats = stats
        self.nostat = nostat
        self.nodesc = nodesc
        self.types = types
        self.model_name = llm_name


    def fit(self, X):
        self.df = X

        self._build_prompts()
        return self
    
    def _generate_batch(self, num: int):
        pass
    
    def _format_metadata(self):
        cont_cols = [col for col in self.types.keys() if self.metadata[col]["type"]!="str"]
        
        metadata_string = ''
        for col in self.types.keys():
            description = ""
            if self.nostat:
                if 'unique' in self.metadata[col]:
                    description = "This column contains the following categorical values:\n"
                    for val in self.metadata[col]['unique']:
                        description += f'- {val}\n'

            else:
                if 'unique' in self.metadata[col]:
                    description = f"This column contains the following categorical values [{', '.join([str(x) for x in self.metadata[col]['unique']])}] with the probabilities [{', '.join([str(x) for x in self.metadata[col]['probs']])}]."
     

                else:
                    description = "Values in this column have"
                    if "min" in self.stats:
                        description += f"a minimum of {round(self.metadata[col]['min'], 3)}, "
                    if "max" in self.stats:
                        description += f"a maximum of {round(self.metadata[col]['max'], 3)}, "
                    if "mean" in self.stats:
                        description += f"a mean of {round(self.metadata[col]['mean'], 3)}, "
                    if "std" in self.stats:
                        description += f"a standard deviation of {round(self.metadata[col]['std'], 3)}, "
                    if "25%" in self.stats:
                        description += f"the 25% quantile is at {round(self.metadata[col]['25%'], 3)}, "
                    if "50%" in self.stats:
                        description += f"the median lies at {round(self.metadata[col]['50%'], 3)}, "
                    if "75%" in self.stats:
                        description += f"and the 75% quantile is at {round(self.metadata[col]['75%'], 3)}, "
                    if "skew" in self.metadata[col]:
                        description += f" a skewness of {round(self.metadata[col]['skew'], 3)}, "
                    if "kurtosis" in self.metadata[col]:
                        description += f" a kurtosis of {round(self.metadata[col]['kurtosis'], 3)}, "
                        
                    description = description[:-2] + "."
                    
                if "correlations" in self.stats:
                    description += "\n"
                    for other_col, coefficient in self.metadata[col]['correlations'].items():
                        
                        if col in cont_cols:
                            # CONT influenced by CONT
                            if other_col in cont_cols:
                                descriptor = "positively" if coefficient > 0 else "negatively"
                                descriptor = "highly "+descriptor if abs(coefficient) > 0.5 else descriptor if abs(coefficient) > 0.3 else "slightly "+descriptor
                                description += f"{col} is {descriptor} correlated with {other_col} with a pearson correlation coefficient of {coefficient}."
                                if coefficient > 0:
                                    description += f" Higher values of {col} are associated with higher values of {other_col}."
                                else:
                                    description += f" Higher values of {col} are associated with lower values of {other_col}."
                            
                            # CONT influenced by CAT   
                            else:
                                descriptor = " highly " if abs(coefficient) > 0.5 else " " if abs(coefficient) > 0.3 else " slightly "
                                description += f"{col} is {descriptor} correlated with {other_col} with an eta squared coefficient of {coefficient}."
                                description += f" Higher eta squared coefficients mean a larger part of the variance in {col} can be explained by {other_col}."
                                
                        else:
                            # CAT influeced by CONT
                            if other_col in cont_cols:
                                descriptor = " highly " if abs(coefficient) > 0.5 else " " if abs(coefficient) > 0.3 else " slightly "
                                description += f"{col} is{descriptor}correlated with {other_col} with a pseudo r squared score of {coefficient}."
                                description += f" Higher pseudo r squared scores mean {col} can more accurately be predicted by {other_col} using logistic regression."
                            
                            # CAT influenced by CAT    
                            else:
                                descriptor = " highly " if abs(coefficient) > 0.5 else " " if abs(coefficient) > 0.3 else " slightly "
                                description += f"{col} is{descriptor}correlated with {other_col} with a Cramér’s V coefficient of {coefficient}."
                                description += f" Higher Cramér’s V coefficients mean a stronger association of {other_col} and {col}."
                                    
                        description += "\n"

            if self.nodesc:
                metadata_string += f"""# Column Name: {col}
type: {self.metadata[col]['type']}
{description}
"""
            else:            
                metadata_string += f"""# Column Name: {col}
type: {self.metadata[col]['type']}
{self.metadata[col]['description']}
{description}
"""

        return metadata_string

    def extract_df(self, text):
        
        data = {
            col: [] for col in self.metadata.keys() if '_int' not in col
        }

        columns = [col for col in self.metadata.keys() if '_int' not in col]

        for row in text.split("\n"):
            row = row.rstrip(',')
            try:
                if row.count(",") == len(columns) - 1:
                    values = row.split(",")
                    casted_values = []

                    for value, col in zip(values, columns):
                        if self.metadata[col]["type"] == "int":
                            casted_value = int(float(value.replace("'", "").replace('"', "").strip()))
                            if 'unique' in self.metadata[col]:
                                if casted_value not in self.metadata[col]["unique"]:
                                    casted_value = None
                            else:
                                if casted_value < self.metadata[col]["min"]:
                                    casted_value = None #self.metadata[col]["min"]
                                elif casted_value > self.metadata[col]["max"]:
                                    casted_value = None #self.metadata[col]["max"]
                            items = [(casted_value, col)]

                        elif self.metadata[col]["type"] == "float":
                            casted_value = float(value.replace("'", "").replace('"', "").strip())
                            if casted_value < self.metadata[col]["min"]:
                                casted_value = None #self.metadata[col]["min"]
                            elif casted_value > self.metadata[col]["max"]:
                                casted_value = None #self.metadata[col]["max"]
                            items = [(casted_value, col)]

                        elif self.metadata[col]["type"] == "str":
                            casted_value = value.replace("'", "").replace('"', "").strip()
                            if casted_value not in [str(u) for u in self.metadata[col]["unique"]]:
                                items = [(None, col)]
                            else:
                                items = [([str(v) for v in self.metadata[col]["unique"]].index(str(casted_value)), col)]

                        
                        casted_values = casted_values + items

                    if not any([item[0] is None for item in casted_values]) and len(casted_values) == len(columns):
                        for pair in casted_values:
                            if pair[1] in data:
                                data[pair[1]].append(pair[0])
                                
            except (ValueError, OverflowError) as e:
                print(col)
                print(e)
                continue

        generated = pd.DataFrame(data)
        for col in generated.columns:
            if self.metadata[col]["type"] == "str":
                generated[col] = generated[col].astype(int)
            elif self.metadata[col]["type"] == "int":
                generated[col] = generated[col].round().astype(int)
            elif self.metadata[col]["type"] == "float":
                generated[col] = generated[col].astype(float)

        return generated

    def generate(self, count: int):
        # conservative estimate of the number of samples returned per chat
        expected_num_samples_per_chat = 16

        print("Generating", count, "samples.")
        # calculate the number of chats required to generate the desired count
        required_number_of_chats = math.ceil(count / expected_num_samples_per_chat)
        total_chats = required_number_of_chats
        print(f"Required number of chats: {required_number_of_chats} = {count} / {expected_num_samples_per_chat}")

        self.generated = None
        while self.generated is None:
            generated_dataframes = self._generate_batch(num = required_number_of_chats)
            if len(generated_dataframes) > 0:
                self.generated = pd.concat(generated_dataframes, ignore_index=True)

        while len(self.generated) < count:
            print("Generated Length so far:", len(self.generated))

            # Update the expected number of samples per chat based on the current generated length
            expected_num_samples_per_chat = math.floor(len(self.generated) / total_chats)

            # Recalculate the number of chats required to reach the desired count
            required_number_of_chats = math.ceil((count - len(self.generated)) / expected_num_samples_per_chat)
            total_chats += required_number_of_chats

            generated_batch = self._generate_batch(num = required_number_of_chats)
            if len(generated_batch) > 0:
                self.generated = pd.concat([self.generated] + generated_batch, ignore_index=True)

        return self.generated[:count]