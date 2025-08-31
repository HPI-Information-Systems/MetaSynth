import asyncio
from metasynth.generators.metadata_generator_base import MetadataSynthBase
from metasynth.generators.llm_utils import get_responses


class MetaSynth(MetaSynthBase):
    INTRODUCTION = """# You are a data scientist trying to synthesize new data samples for a table. 
# After receiving all information, you want to generate synthetic data based on the following column information and identify possible relationships between the columns."""

    CALL = """## Task:
- Generate 25 new data rows based on the identified relationships and the column information.
- DO NOT write any code. Only provide the new data samples in a CSV format.

## Instructions for the Model:
1. Consider possible positive or negative correlations between the columns.
2. Data samples should be consistent and realistic. The features of one sample should work together to create a plausible data point.

Generate 25 additional sample rows of the given table with the following header:"""

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

            full_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
            self.chat = chat
        except:
            chat = [
                    {"role": "user", "content": self.INTRODUCTION+"\n"+metadata_string+call},
                ]

            full_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
            self.chat = chat

        print("TOKEN LENGTH", len(self.tokenizer(full_prompt)["input_ids"]))


    def _generate_batch(self, num=1):    
        print(self.tokenizer.apply_chat_template(self.chat, tokenize=False))
        
        responses = asyncio.run(get_responses([self.chat for _ in range(num)], llm=self.llm, sampling_params=self.sampling_params, client=self.client, llm_name=self.model_name))
                    
        print(responses[0])
        return [self.extract_df(r) for r in responses]
