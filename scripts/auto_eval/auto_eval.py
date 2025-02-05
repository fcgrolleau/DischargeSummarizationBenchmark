import pandas as pd
from pathlib import Path
import json
from functools import partial
from pathlib import Path
from agnostic_evaluator_models import API_text_to_text, openai_init, openai_call, anthropic_init, anthropic_call, meta_init, meta_call, lab_key

gpt4o = partial(openai_init, "gpt-4o", lab_key)
claude = partial(anthropic_init, "claude-3-5-sonnet-v2", lab_key)
llama = partial(meta_init, "llama-3-3-70B-instruct", lab_key)
my_llms = [(gpt4o, openai_call), (claude, anthropic_call), (llama, meta_call)]

def llm_output_to_json(llm_output):
    reformated_response =  llm_output.replace("\n", " ").replace("*", " ").replace("-", " ").replace("```", "").replace("json", "")
    reformated_response = reformated_response[:-10] + reformated_response[-10:].replace(",", "")
    for _ in range(5):
        try:
            json.loads(reformated_response)
        except Exception as e:
            e = str(e)
            coma_pos = int(e[e.find("char ")+5:].replace(")", ""))
            reformated_response = reformated_response[:coma_pos] + "," + reformated_response[coma_pos:]
    return json.loads(reformated_response)

def make_fact_eval_prompt(proto_ds, fact):
    prompt = f"""You are an expert AI assistant specializing in internal medicine. Your task is to analyze a provided hospital course summary and determine whether a specific, important fact is explicitly mentioned.
    Output your response as a valid JSON object in the following format:

    ```json
    {{
        "explanation": "Detailed, step-by-step reasoning process explaining why the fact is or is not present in the hospital course summary.",
        "fact_mentioned": integer  // 1 if the fact is explicitly mentioned, 0 if not.
    }}
    
    
    You are an AI assistant specialized in internal medicine. Your task is to analyze the provided hospital course summary and determine whether the specified important fact is mentioned.
    Output the response as valid JSON in the following format:

    {{
    "explanation": "string",
    "fact_mentioned": integer
    }}

    Guidelines:
    1. Reasoning: Provide a clear and concise explanation of your thought process in the "explanation" field. Break down your analysis into logical steps, showing how you searched for the information and what led you to your conclusion. Explain why you believe the fact is or is not mentioned.
    2. Fact Determination: The "fact_mentioned" field must be either 1 or 0.
        - Use 1 only if the important fact is explicitly and unambiguously stated in the hospital course summary.
        - Use 0 if the fact is not explicitly mentioned, even if it could be inferred or is likely to be true based on other information. Ambiguity implies the fact is not explicitly mentioned.
    3. JSON Format:
        Adhere strictly to the JSON format provided. Do not include any surrounding text or markdown.
        Do not nest JSON objects within each other.
        The entire response must be a single, valid JSON object enclosed in curly braces `{"}"}`.
        Ensure proper key-value pairing and use of quotation marks.
    4. No Extraneous Output: Output only the JSON object. Do not include any introductory or concluding sentences, greetings, or other text outside the JSON.

    --- Important Fact to look for ---
    {fact}

    --- Hospital Course Summary ---
    {proto_ds}"""
    return prompt

def make_llm_as_judge_prompt(proto_ds):
    prompt = f"""You are an AI assistant, acting as a senior internal medicine physician evaluating the quality of a hospital course summary. Your task is to analyze the provided summary and assign it a quality score from 1 to 10, where 10 represents the highest possible quality.
    Your response MUST be formatted as valid JSON according to the following schema:

    ```json
    {{
    "explanation": "string",
    "score": integer
    }}

    Instructions:

    1. Reasoning Process (Explanation): In the "explanation" field, meticulously detail your reasoning for the assigned score. Specifically address the following aspects of the hospital course summary:
        - Completeness: Does the summary include all essential elements of a standard hospital course (e.g., presenting problem, pertinent history, key exam findings, labs/imaging results, treatment plan, response to treatment, consultations, discharge plan, discharge medications, follow-up instructions)? Are any crucial details missing?
        - Accuracy: Is the information presented factually correct and consistent with expected clinical findings given the patient's condition? Are there any contradictions or inconsistencies within the summary?
        - Clarity & Conciseness: Is the summary written in clear, concise, and unambiguous language? Is it free of jargon and unnecessary details? Is the timeline of events easy to follow?
        - Organization: Is the summary logically organized, allowing for easy understanding of the patient's hospital stay?
        - Appropriateness of Detail: Does the summary include the right level of detail? Is it overly verbose or too brief to be informative? Does it focus on the most relevant information?
        - Justification of score: Explicitly explain how each of the above aspects influenced the score.
    2. Scoring (Score): Assign a single integer value from 1 to 10, inclusive, to the "score" field. Use the following guidelines to anchor your scoring:
        - 1-3: Unacceptable. The summary is significantly incomplete, inaccurate, poorly written, and/or disorganized. It provides little to no useful information.
        - 4-6: Below Average. The summary has significant flaws in completeness, accuracy, clarity, or organization. Requires substantial revisions.
        - 7-8: Good. The summary is generally well-written, accurate, and complete, with only minor areas for improvement.
        - 9-10: Excellent. The summary is comprehensive, accurate, clear, concise, and well-organized. It provides a high-quality overview of the patient's hospital course.
    3. Output Format: Adhere STRICTLY to the JSON schema provided above. Ensure the JSON object is well-formed and contains only the "explanation" and "score" fields. Do not include any extraneous text or conversational elements outside the JSON object.
    4. Important Restrictions:
        - No Nested JSON: Do NOT embed a JSON object within another JSON object.
        - Single JSON Object: Provide only ONE JSON object as your response.
        - No Trailing Text: Do NOT add any text or comments after the closing curly bracket `{"}"}`.
        
    --- Hospital Course Summary to evaluate ---
    {proto_ds}"""
    return prompt

class AutoEval:
    def __init__(self, llm_eval_pair, proto_model="gpt-4o"):
        self.llm_eval_pair = llm_eval_pair
        self.proto_model = proto_model
        self.test_path = Path("../prototyping/generated_dc_sum/testset")
        self.fact_df_path = Path("../../exports/fact_data/benchmark_creation - all_responses.csv")

        # Read the facts
        fact_df = pd.read_csv(self.fact_df_path)
        self.facts = pd.DataFrame()
        self.facts = fact_df.iloc[:,[3,5,7]].copy()
        self.facts.loc[:,'patient_i'] = fact_df.iloc[:,0].apply(lambda x: int(''.join(filter(str.isdigit, str(x))))) - 1 
        
        # Read the proto summaries
        proto_files = list(self.test_path.rglob(f"**/{proto_model}.md"))
        self.proto_summaries = []
        for file_path in proto_files:
            patient_i = [p for p in file_path.parts if "patient_" in p][0]
            patient_n = int(''.join(filter(str.isdigit, patient_i))) 
            with open(file_path, 'r', encoding='utf-8') as f:
                self.proto_summaries.append((patient_n, f.read().split("\n\n", 1)[1]))
        self.proto_summaries = pd.DataFrame(self.proto_summaries, columns=['patient_i', 'summary'])
        # Include only the proto summaries of patients that are in the fact_df
        self.proto_facts_merged = pd.merge(self.proto_summaries, self.facts, left_on='patient_i', right_on='patient_i', how='right')
        
    def fact_eval(self):
        llm_instance = API_text_to_text(*self.llm_eval_pair)
        self.fact_eval_res = {}
        self.fact_eval_expl = {}
        for patient_i in range(len(self.proto_facts_merged)):
            id = self.proto_facts_merged["patient_i"][patient_i]
            self.fact_eval_res[f'patient_{id}'] = {}
            self.fact_eval_expl[f'patient_{id}'] = {}
            for fact_j in range(3):
                proto_summary = self.proto_facts_merged.iloc[patient_i, 1]
                fact = self.proto_facts_merged.iloc[patient_i,2+fact_j]
                fact_eval_prompt = make_fact_eval_prompt(proto_summary, fact)
                llm_output = llm_instance.gen_txt_to_txt(fact_eval_prompt)
                self.fact_eval_res[f'patient_{id}'][f'fact_{fact_j}'] = float(llm_output_to_json(llm_output).get("fact_mentioned", float('nan')))
                self.fact_eval_expl[f'patient_{id}'][f'fact_{fact_j}'] = llm_output_to_json(llm_output).get("explanation", float('nan'))
                
    def unconditional_eval(self):
        llm_instance = API_text_to_text(*self.llm_eval_pair)
        self.unc_eval_res = {}
        self.unc_eval_expl = {}
        for patient_i in range(len(self.proto_facts_merged)):
            id = self.proto_facts_merged["patient_i"][patient_i]
            self.unc_eval_res[f'patient_{id}'] = {}
            self.unc_eval_expl[f'patient_{id}'] = {}
            proto_summary = self.proto_facts_merged.iloc[patient_i, 1]
            unconditional_eval_prompt = make_llm_as_judge_prompt(proto_summary)
            llm_output = llm_instance.gen_txt_to_txt(unconditional_eval_prompt)
            self.unc_eval_res[f'patient_{id}'] = float(llm_output_to_json(llm_output).get("score", float('nan')))
            self.unc_eval_expl[f'patient_{id}'] = llm_output_to_json(llm_output).get("explanation", float('nan'))

if __name__ == "__main__": 
    autoeval_ins = AutoEval((gpt4o, openai_call), "gpt-4o")
    autoeval_ins.facts
    autoeval_ins.proto_summaries
    autoeval_ins.proto_facts_merged
    autoeval_ins.fact_eval()
    autoeval_ins.fact_eval_res
    autoeval_ins.fact_eval_expl
    autoeval_ins.unconditional_eval()
    autoeval_ins.unc_eval_res
    autoeval_ins.unc_eval_expl