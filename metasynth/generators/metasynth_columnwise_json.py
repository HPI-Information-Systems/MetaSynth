import copy
import json
import pandas as pd

import asyncio
from metasynth.generators.metasynth_base import MetaSynthBase
from metasynth.generators.llm_utils import get_responses
import signal
import re
import pickle

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
# You generate 25 new data samples column by column. You will recieve the distribution information and possible correlations with the previously generated columns one by one.
"""

    COLUMNS_INTRODUCTION = """Here are the statistics and correlations for the next columns {column_names}:
"""

    COLUMN_INTRODUCTION = """Here are the statistics and correlations for the next column {column_names}:
"""

    CALL_CAT_FIRST = """## Task:
- Generate 20 new data samples for the column {column_names}
- ONLY generate the values for this new column.
- Do not generate additional information like an index or ID.
- Generate the values row by row, one value per row.
- DO NOT write any code. Only provide the new data samples in a JSON format.

## Instructions for the Model:
Follow the statistics and correlations provided for the column.

For every sample, follow this JSON format. Generate actual values instead of '<placeholder>':
"""

    CALL_CONT_FIRST = """## Task:
- Generate 20 new data samples for the column {column_names}
- ONLY generate the values for this new column.
- Do not generate additional information like an index or ID.
- Generate the values row by row, one value per row.
- DO NOT write any code. Only provide the new data samples in a JSON format.

## Instructions for the Model:
Follow the statistics and correlations provided for the column.

For every sample, follow this JSON format. Generate actual values instead of '<placeholder>':
"""

    CALL_FIRST = """## Task:
- Generate 20 new data samples for the columns {column_names}
- ONLY generate the values for these new columns.
- Do not generate additional information like an index or ID.
- Generate the values row by row.
- DO NOT write any code. Only provide the new data samples in a JSON format.

## Instructions for the Model:
Follow the statistics and correlations provided for the column.

For every sample, follow this JSON format. Generate actual values instead of '<placeholder>':
"""

    CALL_CAT = """## Task:
- Generate 20 new data samples for the column {column_names}
- ONLY generate the values for this new column. Copy the generations from previous columns.
- DO NOT write any code. Only provide the new data samples in a JSON format.

## Instructions for the Model:
Follow the statistics and correlations provided for the column.
Make sure that the generated values work together with the values you generated for the previous columns in every row and create reasonable samples. To do this:
- Think about how the values in this column are related to the values in the previously generated columns.
- Some values of {column_names} might be more likely to occur in the presence of certain values in the previously generated columns. Take this into account when generating new values.

For every sample, follow this JSON format. Generate actual values instead of '<placeholder>':
"""

    CALL_CONT = """## Task:
- Generate 20 new data samples for the column {column_names}
- ONLY generate the values for this new column. Copy the generations from previous columns.
- DO NOT write any code. Only provide the new data samples in a JSON format.

## Instructions for the Model:
Follow the statistics and correlations provided for the column.
Make sure that the generated values work together with the values you generated for the previous columns in every row and create reasonable samples. To do this:
- Think about how the values in this column are related to the values in the previously generated columns.
- The mean of {column_names} might be different depending on certain values in the previously generated columns. Take this into account when generating new values.

For every sample, follow this JSON format. Generate actual values instead of '<placeholder>':
"""

    CALL = """## Task:
- Generate 20 new data samples for the columns {column_names}
- ONLY generate the values for these new columns. Copy the generations from previous columns.
- DO NOT write any code. Only provide the new data samples in a JSON format.

## Instructions for the Model:
Follow the statistics and correlations provided for the new columns.
Make sure that the generated values work together with the values you generated for the previous columns in every row and create reasonable samples. To do this:
- Think about how the values in the new columns are related to the values in the previously generated columns.
- The distribution of the new columns might be shifted depending on certain values in the previously generated columns. Take this into account when generating new values.

For every sample, follow this JSON format. Generate actual values instead of '<placeholder>':
"""

    def _build_prompts(self):

        self.chat = [{"role": "system", "content": self.INTRODUCTION}]


    def get_metadata(self, col, previous_cols):
        cont_cols = [col for col in self.types.keys() if self.metadata[col]["type"]!="str"]
        
        if 'unique' in self.metadata[col]:
            if not self.nostat:
                description = "This column contains the following categorical values with the given probabilities:\n"
                for val, prob in zip(self.metadata[col]['unique'], self.metadata[col]['probs']):
                    prob = round(prob*100)
                    if prob == 0:
                        description += f'- {val} is present in less than 1 out of 100 samples.\n'
                    else:
                        description += f'- {val} is present in {prob} out of 100 samples.\n'
            else:
                description = "This column contains the following categorical values:\n"
                for val in self.metadata[col]['unique']:
                    description += f'- {val}\n'

        else:
            #print("Generating metadata for column:", col, self.metadata[col]["distribution_type"])
            if not self.nostat:
                if any([property in self.stats for property in ["25%", "50%", "75%", "skew", "kurtosis"]]):
                    #print({col: self.metadata[col]})
                    if len(self.metadata[col]['gmm_means']) > 1:
                        description = f"Values in this column follow a multimodel distribution with {len(self.metadata[col]['gmm_means'])} modes."
                        description += " The means of the modes are at "
                        for mean, weight in zip(self.metadata[col]['gmm_means'], self.metadata[col]['gmm_weights']):
                            description += f"{round(mean, 3)} with a weight of {round(weight, 3)}, "
                        description = description[:-2] + ". "
                        description += "The overall distribution is a mixture of these modes with "
                    elif self.metadata[col]["distribution_type"] == "norm":
                        description = "Values in this column follow a normal distribution with "
                    elif self.metadata[col]["distribution_type"] == "uniform":
                        description = "Values in this column follow an equal distribution distribution with " # try equal distribution instead of uniform
                    # elif self.metadata[col]["distribution_type"] == "lognorm":
                    #     description = "Values in this column follow a lognormal distribution with "
                    elif self.metadata[col]["distribution_type"] == "expon":
                        description = "Values in this column follow an exponential distribution with "
                else:
                    description = "Values in this column follow a distribution with "
                
                if "min" in self.stats:
                    description += f"a minimum of {round(self.metadata[col]['min'], 3)}, "
                if "max" in self.stats:
                    description += f"a maximum of {round(self.metadata[col]['max'], 3)}, "
                if "mean" in self.stats and self.metadata[col]["distribution_type"] != "uniform" and len(self.metadata[col]['gmm_means']) == 1:
                    description += f"a mean of {round(self.metadata[col]['mean'], 3)}, "
                if "std" in self.stats and self.metadata[col]["distribution_type"] != "uniform" and len(self.metadata[col]['gmm_means']) == 1:
                    description += f"a standard deviation of {round(self.metadata[col]['std'], 3)}, "
                if "25%" in self.stats and self.metadata[col]["distribution_type"] != "uniform" and len(self.metadata[col]['gmm_means']) == 1:
                    description += f"the 25% quantile is at {round(self.metadata[col]['25%'], 3)}, "
                if "50%" in self.stats and self.metadata[col]["distribution_type"] != "uniform" and len(self.metadata[col]['gmm_means']) == 1:
                    description += f"the median lies at {round(self.metadata[col]['50%'], 3)}, "
                if "75%" in self.stats and self.metadata[col]["distribution_type"] != "uniform" and len(self.metadata[col]['gmm_means']) == 1:
                    description += f"the 75% quantile is at {round(self.metadata[col]['75%'], 3)}, "
                if "skew" in self.stats and self.metadata[col]["distribution_type"] != "uniform" and len(self.metadata[col]['gmm_means']) == 1:
                    description += f" a skewness of {round(self.metadata[col]['skew'], 3)}, "
                if "kurtosis" in self.stats and self.metadata[col]["distribution_type"] != "uniform" and len(self.metadata[col]['gmm_means']) == 1:
                    description += f" a kurtosis of {round(self.metadata[col]['kurtosis'], 3)}, "
                    
                description = description[:-2] + "."
            else:
                description = f"Values in this column are between {round(self.metadata[col]['min'], 3)} and {round(self.metadata[col]['max'], 3)}."
            
        if "correlations" in self.stats and not self.nostat:
            if description.endswith(" with "):
                description = description[:-6] + "."
            description += "\n"
            for other_col, coefficient in self.metadata[col]['correlations'].items():
                if other_col not in previous_cols:
                    continue
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

          
        if self.nodesc:
            return f"""# Column Name: {col}
type: {self.metadata[col]['type']}
{description.rstrip()}
"""
                     
        return f"""# Column Name: {col}
type: {self.metadata[col]['type']}
{self.metadata[col]['description']}
{description.rstrip()}
"""

    def create_assistant_response_weal(self, text):
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

        matches = [match.group(0) for match in pattern.finditer(text.replace("'", '"'))]
        strings = [ ]
        
        for dictionary in matches:
            json_str = dictionary.replace("'", '"').replace('\n', '').replace('\r', '')
            json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
            try:
                data = json.loads(json_str)
                json_string = json.dumps(data, indent=4)
                strings.append(json_string)
            except:
                continue

        return "\n".join(strings)
    
    def create_assistant_response(self, dataframe, columns):
        
        df = dataframe.copy(deep=True)
        
        for col in df.columns:
            if self.types[col] == "str":
                df[col] = df[col].apply(lambda x: self.metadata[col]["unique"][int(x)])
        return "\n".join(json.dumps(dictionary, indent=4) for dictionary in df[columns].to_dict(orient='records'))

    def get_generation_groups(self):
        previous_colums = []
        groups = []
        group = []
        columns = [col for col in self.metadata.keys() if '_int' not in col]
        columns.sort(key=lambda x: len(self.metadata[x]["correlations"]), reverse=False)
        while len(previous_colums) < len(columns):
            for col in columns:
                if col in previous_colums or len(group) >= 8:
                    continue
                if not any([c in self.metadata[col]["correlations"] for c in group]):
                    group.append(col)
                    previous_colums.append(col)
                    
            groups.append(group)
            group = []
        return groups

    def extract_df(self, text, columns):
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
                        \s*                      # optional whitespace
                        (?:,|\n)?                # optional comma or newline
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
            print("Regex took too long")
            return pd.DataFrame()

        dictionaries = []
        
        # if len(matches) == 0:
        #     print("No matches found in the text")
        #     print(text)
        #     print("|---------------------------------|")
        error_text = ""

        for obj in matches:
            try:
                json_str = obj.replace("'", '"')
                # Remove newlines and carriage returns
                json_str = json_str.replace('\n', '').replace('\r', '')
                # Ensure commas between key-value pairs if missing
                json_str = re.sub(r'(\}|\]|\d|")\s*("\w)', r'\1,\2', json_str)
                # Clean up extra commas before closing braces or brackets
                json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
                dictionary = json.loads(json_str)
                if all([col in dictionary for col in columns]):
                    for key in columns:
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
                        dictionaries.append({col: dictionary[col] for col in columns})
                        
                else:
                    error_text += f"\nThere was a problem finding the column {columns} in: {dictionary}\n|---------------------------------|\n"

            
            except ValueError as e:
                print(e)
                print(obj)
                continue
            
        # Create DataFrame
        return pd.DataFrame(dictionaries)
    
    def _generate_batch(self, num=1, allowed_dropout=0.2):    
        groups = self.get_generation_groups()
        print(f"Generating using {len(groups)} groups")
        print(groups)

        # Increase the number of chats to account for dropped chats in every iteratiopn
        expected_dropout = 0.96
        corrected_number_of_chats = int(num / (expected_dropout ** len(groups)))
        
        columns = [col for group in groups for col in group]
        
        all_failed_chats = []
        if self.chat is None:
            successful_chats = [[] for _ in range(corrected_number_of_chats)]
        else:
            successful_chats = [self.chat for _ in range(corrected_number_of_chats)]

        previous_colums = []
        
        for idx, group in enumerate(groups):
            print(group)
            previous_colums.extend(group)
            
            if idx == 0:
                call = self.CALL_FIRST if len(group) > 1 else self.CALL_CAT_FIRST if "unique" in self.metadata[group[0]] else self.CALL_CONT_FIRST
            else:
                call = self.CALL if len(group) > 1 else self.CALL_CAT if "unique" in self.metadata[group[0]] else self.CALL_CONT
                
            introduction = self.COLUMNS_INTRODUCTION if len(group) > 1 else self.COLUMN_INTRODUCTION
            example = str({col: "<placeholder>" for col in previous_colums})
            
            if self.chat is None:
                introduction = self.INTRODUCTION + "\n" + introduction
              
            if len(successful_chats) > 0:
                print("Current chat length", len(successful_chats[0]))
                if len(successful_chats[0]) == 5 and self.chat is not None:
                    successful_chats = [[chat[0], chat[3], chat[4]] for chat in successful_chats]
                    print("limited chat length")
                if len(successful_chats[0]) == 4 and self.chat is None:
                    chats = [[chat[2], chat[3]] for chat in successful_chats]
                    print("limited chat length")
            
            current_chats = copy.deepcopy(successful_chats)
            successful_chats = []
            successful_dataframes = []
            stuck_counter = 0

            while (len(current_chats) / max(1, (len(current_chats) + len(successful_chats)))) > allowed_dropout:
                print((len(current_chats) / (len(current_chats) + len(successful_chats))))
                
                current_chats = [chat + [
                        {
                            "role": "user", 
                            "content": introduction.format(column_names=", ".join(group))+"\n".join([self.get_metadata(col, previous_cols=previous_colums) for col in group])+"\n"+call.format(column_names=", ".join(group))+example
                        }
                    ] for chat in current_chats]

                for _ in range(3):
                    try:
                        print("temp", self.temperature)
                        responses = asyncio.run(get_responses(current_chats, llm=self.llm, sampling_params=self.sampling_params, client=self.client, llm_name=self.model_name, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k))
                        break
                    except Exception as e:
                        print("Error during response generation:", e)
                        print("Retrying")
                        continue

                dataframes = [self.extract_df(response, previous_colums) for response in responses]
                
                failed_chats = [chat + [{"role": "assistant", "content": response}] for chat, response in zip(current_chats,responses) if len(self.extract_df(response, previous_colums)) < 8]
                all_failed_chats.extend(failed_chats)
                
                chats = [chat + [{"role": "assistant", "content": self.create_assistant_response(df, previous_colums)}] for df, chat in zip(dataframes, current_chats) if len(df) >= 8]
                successful_chats.extend(chats)
                successful_dataframes.extend([df for df in dataframes if len(df) >= 8])
                current_chats = [chat[:-1] for df, chat in zip(dataframes, current_chats) if len(df) < 8]
                print("Current number of successful chats", len(successful_chats), "Current number of repeated chats", len(current_chats))
                
                if len(chats) == 0:
                    stuck_counter += 1
                    if stuck_counter >= 3:
                        break
                    
        self.all_failed_chats = all_failed_chats
        # Save the failed chats using pickle
        # with open('failed_chats.pkl', 'wb') as f:
        #     pickle.dump(all_failed_chats, f)
        results = [d for d in successful_dataframes if len(d) >= 5]
        print("final number of reposnses:", len(results))
        return results
