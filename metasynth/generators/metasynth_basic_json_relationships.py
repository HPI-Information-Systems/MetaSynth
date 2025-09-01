import json
import re
import signal

import pandas as pd
import asyncio
from metasynth.generators.metasynth_base import MetaSynthBase
from metasynth.generators.llm_utils import get_responses

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def remove_leading_zeros_from_numbers(s):
    # Regular expression to match numbers with leading zeros
    pattern = re.compile(r'\b0+(\d+(\.\d+)?)\b')
    
    # Function to remove leading zeros
    def replace_leading_zeros(match):
        return match.group(1)

    # Use sub method to replace all occurrences in the string
    result = pattern.sub(replace_leading_zeros, s)
    return result

class MetaSynth(MetaSynthBase):
    INTRODUCTION = """# You are a data scientist trying to synthesize new data samples for a table. 
# After recieveing all information, you want to generate synthetic data based on the following column information and relationships between the columns."""

    CALL = """## Task:
- Generate 20 new data rows based on the identified relationships and the column information.
- DO NOT write any code. Only provide the new data samples in a JSON format.

## Instructions for the Model:
1. Consider possible positive or negative correlations between the columns.
2. Data samples should be consistent and realistic. The features of one sample should work together to create a plausible data point.

For every sample, follow this JSON format:
"""

    def extract_df(self, text):
        columns = list(self.types.keys())
        text = remove_leading_zeros_from_numbers(text)
        # Set the signal handler and a 10-second alarm
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)

        try:
            
            pattern = re.compile(r"""
                    \{                           # opening brace
                    (?:                          # non-capturing group for key-value pairs
                        \s*                      # optional whitespace
                        (['"])                   # opening quote (single or double)
                        [^'"]*                   # key (any characters except quotes)
                        \1                       # matching closing quote
                        \s*:\s*                  # colon with optional whitespace
                        (?:                      # non-capturing group for value
                            (['"])               # opening quote for value
                            [^'"]*               # value (any characters except quotes)
                            \2                   # matching closing quote
                            |                    # or
                            -?\d+(\.\d+)?        # number (integer or float, with optional minus sign)
                        )                        # end value group
                        \s*,?                    # optional comma and optional whitespace
                    )+                           # repeat key-value pairs
                    \s*                          # optional whitespace
                    \}                           # closing brace
                """, re.VERBOSE)

            # Find all matches
            matches = [match.group(0) for match in pattern.finditer(text)]
            
            # Cancel the alarm
            signal.alarm(0)
            
        except TimeoutException:
            # If the regex takes too long, return an empty dataframe
            return pd.DataFrame()

        dictionaries = []

        for obj in matches:
            try:
                json_str = obj.replace("'", '"').replace('\n', '').replace('\r', '')
                json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
                dictionary = json.loads(json_str)
                if len(dictionary) == len(columns) and all([col in dictionary for col in columns]):
                    for key in dictionary.keys():
                        value = str(dictionary[key]).replace("'", "").replace('"', "").strip()
                        if self.metadata[key]["type"] == "int":
                            casted_value = int(float(value))
                            if 'unique' in self.metadata[key]:
                                if casted_value not in self.metadata[key]["unique"]:
                                    dictionary[key] = None
                                else:
                                    dictionary[key] = casted_value
                            else:
                                if casted_value < self.metadata[key]["min"] or casted_value > self.metadata[key]["max"]:
                                    dictionary[key] = None
                                else:
                                    dictionary[key] = casted_value

                        elif self.metadata[key]["type"] == "float":
                            casted_value = float(value)
                            if casted_value < self.metadata[key]["min"] or casted_value > self.metadata[key]["max"]:
                                dictionary[key] = None
                            else:
                                dictionary[key] = casted_value

                        elif self.metadata[key]["type"] == "str":
                            casted_value = str(value)
                            if casted_value not in [str(u) for u in self.metadata[key]["unique"]]:
                                dictionary[key] = None
                            else:
                                dictionary[key] = str(self.metadata[key]["unique"].index(str(casted_value)))

                    if not any([v is None for v in dictionary.values()]):
                        dictionaries.append(dictionary)

            except ValueError as e:
                #print(e)
                #print(obj)
                continue

        # Create DataFrame
        return pd.DataFrame(dictionaries)

    def _format_metadata(self):
        cont_cols = [col for col in self.types.keys() if self.metadata[col]["type"]!="str"]
        
        metadata_string = ''
        for col in self.types.keys():
            if 'unique' in self.metadata[col]:
                description = "This column contains the following categorical values with the given probabilities:\n"
                for val, prob in zip(self.metadata[col]['unique'], self.metadata[col]['probs']):
                    prob = round(prob*100)
                    if prob == 0:
                        description += f'- {val} is present in less than 1 out of 100 samples.\n'
                    else:
                        description += f'- {val} is present in {prob} out of 100 samples.\n'

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
                            relationship = self.metadata[col]["relationships"][other_col]
                            description += f"{col} is influenced by {other_col}. "
                            
                            pos_influence = [key for key, value in relationship.items() if value["mean"] >= 0]
                            neg_influence = [key for key, value in relationship.items() if value["mean"] < 0]
                            
                            description += f"There are higher values of {col} when {other_col} is " + ", ".join(pos_influence) + ". "
                            description += f"There are smaller values of {col} when {other_col} is " + ", ".join(neg_influence) + "."
                            
                    else:
                        # CAT influeced by CONT
                        if other_col in cont_cols:
                            relationship = self.metadata[col]["relationships"][other_col]
                            description += f"{col} is influenced by {other_col}. "
                            
                            likely_big = [key for key, frequency in zip(relationship["big"]["unique"], relationship["big"]["probs"]) if frequency >= 0]
                            likely_small = [key for key, frequency in zip(relationship["small"]["unique"], relationship["small"]["probs"]) if frequency >= 0]
                            
                            description += f"The values {', '.join(likely_big)} of {col} are more likely when {other_col} is greater than average. "
                            description += f"The values {', '.join(likely_small)} of {col} are more likely when {other_col} is smaller than average."
                        
                        # CAT influenced by CAT    
                        else:
                            relationship = self.metadata[col]["relationships"][other_col]
                            description += f"{col} is influenced by {other_col}. "
                            for label in relationship.keys():
                                likely = [key for key, frequency in zip(relationship[label]["unique"], relationship[label]["probs"]) if frequency >= 0]
                                description += f"The values {', '.join(likely)} of {col} are more likely when {other_col} has the value {label}. "
                                
                    description += "\n"

                        
            metadata_string += f"""# Column Name: {col}
type: {self.metadata[col]['type']}
{self.metadata[col]['description']}
{description}
"""

        return metadata_string


    def _build_prompts(self):
        metadata_string = self._format_metadata()
        
        # target = [key for key in self.metadata.keys() if '_int' not in key][-1]
        # probs = [round(p * 25) for p in self.metadata[target]["probs"]]
        # classes = [str(c) for c in self.metadata[target]["unique"]]
        # generation_distribution = ", ".join([f"{probs[i]} of the samples from target class '{classes[i]}'" for i in range(len(probs))])+" in a csv format given this header:\n"

        call = self.CALL + str({col: "<placeholder>" for col in self.metadata.keys() if '_int' not in col})

        try:
            chat = [
                    {"role": "system", "content": self.INTRODUCTION},
                    {"role": "user", "content": metadata_string+call},
                ]

            self.full_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        except:
            chat = [
                    {"role": "user", "content": self.INTRODUCTION+"\n"+metadata_string+call},
                ]

            self.full_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)

        self.chat = chat
        print("TOKEN LENGTH", len(self.tokenizer(self.full_prompt)["input_ids"]))

    def _generate_batch(self, num=1):    
        print(self.tokenizer.apply_chat_template(self.chat, tokenize=False))
        
        responses = asyncio.run(get_responses([self.chat for _ in range(num)], llm=self.llm, sampling_params=self.sampling_params, client=self.client, llm_name=self.model_name))
                    
        print(responses[0])
        return [self.extract_df(r) for r in responses]
