import asyncio
from metasynth.generators.metadata_generator_base import MetadataSynthBase
from metasynth.generators.llm_utils import get_responses


class MetaSynth(MetaSynthBase):
    INTRODUCTION = """# You are a data scientist trying to synthesize new data samples for a table. 
# After recieveing all information, you want to generate synthetic data based on the following column information and identify possible relationships between the columns."""
# The resulting dataset will be used to train a machine learning model to predict the target column. Additionally to the other instructions, prioritize the generation of data samples that have correct relationships of the target column with the other columns."""

    CALL = """## Task:
- Generate 25 new data rows based on the identified relationships and the column information.
- DO NOT write any code. Only provide the new data samples in a csv format.

## Instructions for the Model:
1. Consider possible positive or negative correlations between the columns.
2. Data samples should be consistent and realistic. The features of one sample should work together to create a plausible data point.

Generate 25 additional sample rows of the given table with the CSV header:
"""

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
                if "skew" in self.stats:
                    description += f" a skewness of {round(self.metadata[col]['skew'], 3)}, "
                if "kurtosis" in self.stats:
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
                            descriptor = " strongly " if abs(coefficient) > 0.5 else " " if abs(coefficient) > 0.3 else " slightly "
                            description += f"{col} is{descriptor}influenced by {other_col}. "
                            
                            pos_influence = [key for key, value in relationship.items() if value["mean"] >= 0]
                            neg_influence = [key for key, value in relationship.items() if value["mean"] < 0]
                            
                            description += f"There are higher values of {col} when {other_col} is " + ", ".join(pos_influence) + ". "
                            description += f"There are smaller values of {col} when {other_col} is " + ", ".join(neg_influence) + "."
                            
                    else:
                        # CAT influeced by CONT
                        if other_col in cont_cols:
                            relationship = self.metadata[col]["relationships"][other_col]
                            descriptor = " strongly " if abs(coefficient) > 0.5 else " " if abs(coefficient) > 0.3 else " slightly "
                            description += f"{col} is{descriptor}influenced by {other_col}. "
                            
                            likely_big = [key for key, frequency in zip(relationship["big"]["unique"], relationship["big"]["probs"]) if frequency >= 0]
                            likely_small = [key for key, frequency in zip(relationship["small"]["unique"], relationship["small"]["probs"]) if frequency >= 0]
                            
                            description += f"The values {', '.join(likely_big)} of {col} are more likely when {other_col} is greater than average. "
                            description += f"The values {', '.join(likely_small)} of {col} are more likely when {other_col} is smaller than average."
                        
                        # CAT influenced by CAT    
                        else:
                            relationship = self.metadata[col]["relationships"][other_col]
                            descriptor = " strongly " if abs(coefficient) > 0.5 else " " if abs(coefficient) > 0.3 else " slightly "
                            description += f"{col} is{descriptor}influenced by {other_col}. "
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

        call = self.CALL + ",".join([f"{col}" for col in self.metadata.keys() if '_int' not in col])

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
        #print("TOKEN LENGTH", len(self.tokenizer(self.full_prompt)["input_ids"]))

    def _generate_batch(self, num=1):    
        print(self.tokenizer.apply_chat_template(self.chat, tokenize=False))
        
        responses = asyncio.run(get_responses([self.chat for _ in range(num)], llm=self.llm, sampling_params=self.sampling_params, client=self.client, llm_name=self.model_name))
                    
        print(responses[0])
        return [self.extract_df(r) for r in responses]

