import pandas as pd
import vertexai
from vertexai.preview.generative_models import GenerativeModel

hashes = "#####################"
next_note = "---NEXT NOTE---"
note_header = "UNJITTERED NOTE DATE"

def simplify_dates(text, note_type):
    real_start = text.find(note_header) + len(note_header)
    real_h_p = note_type + text[real_start:]
    return real_h_p.replace(hashes, "").strip()

def extract_h_p(text):
    # Find the first occurrence of multiple #
    start = text.find(hashes) + len(hashes)
    # Find the second occurrence
    end = text.find(hashes, start)
    # Return the content between them, stripped of whitespace
    full_h_p = text[start:end].strip()
    return simplify_dates(full_h_p, "H&P").strip()

def extract_last_progress_note(text):
    # Remember that the last note is the first one to be found in the text
    
    # Find first hash occurrence
    first_hash = text.find(hashes)
    # Find second hash occurrence
    start = text.find(hashes, first_hash + len(hashes)) + len(hashes)
    # Find next note marker
    end = text.find(next_note, start)
    full_last_note = text[start:end].strip()    
    return  simplify_dates(full_last_note, "LAST PROGRESS NOTE").strip()

def extract_other_progress_notes(text):    
    # Select all progress notes except the first that appears in the text (the last by date)
    other_progress_notes = text.split("---NEXT NOTE---")[1:]
    # reverse the list to get the first note (by date) first in the list
    other_progress_notes = other_progress_notes[::-1]
    return [simplify_dates(note, "PROGRESS NOTE NO " + str(i+1)) for i, note in enumerate(other_progress_notes)]

content = """1.  Reason for Admission: Clearly state the primary reason for the patient's hospitalization. 
2.  Relevant Medical History: Briefly summarize significant pre-existing medical conditions.
3.  Relevant Surgical History: Note any prior surgeries relevant to the current admission.
4.  Primary Diagnosis: State the definitive diagnosis established during the hospital stay.
5.  Secondary Diagnoses: List any additional diagnoses made during the hospitalization. 
6.  Key Diagnostic Investigations and their Results:** Mention important tests conducted and their findings.  
7.  Therapeutic Procedures Performed:**  Describe any procedures, including surgeries, performed. 
8.  Medications: Detail new medications started, dosage changes, and medications discontinued (along with reasons).
9.  Patient's Condition at Discharge: Summarize the patient's status upon leaving the hospital."""

content_and_requirements = f"""
Content:
{content}

Requirements:
- Ensure the summary is concise yet comprehensive.
- Professional Tone: Employ language appropriate for a medical document.
- Medical Terminology: Use precise medical terminology while ensuring clarity.
- Acronyms: Avoid acronyms unless they are standard in medical documentation (e.g., ECG).
"""

def make_prompt_1(example_input):
    prompt = f"""
You are an internal medicine specialist. Your task is to read the following first (History & Physical) and last progress note from a patient's hospital stay and to generate a concise, professional "Hospital Course Summary" for that patient.

Guidelines: Your Hospital Course Summary should include the following sections, as applicable to the patient's case, and adhere to the following requirements:
{content_and_requirements}
---
Patient History and Physical (H&P):
{extract_h_p(example_input)}
---
Patient last Progress Note:
{extract_last_progress_note(example_input)}
---
Provide your "Hospital Course Summary" for the patient below following the guidelines and additional requirements.
"""
    return prompt

def make_prompt_2(example_input, draft, note_no):
    prompt = f"""
You are an internal medicine specialist tasked with creating a concise yet comprehensive Hospital Course Summary for a patient based on their medical records.
You have written a first draft of the Hospital Course Summary using the initial History & Physical and final progress note. Your task is to improve this draft by incorporating relevant details found in the additional progress note provided.

Guidelines: Your revised Hospital Course Summary should include the following sections, as applicable to the patient's case, and adhere to the following requirements:
{content_and_requirements}
Your Draft "Hospital Course Summary":
{draft}

Extra Information from Other Progress Notes:
---
{extract_other_progress_notes(example_input)[note_no]}
---
Provide Your Improved "Hospital Course Summary" below:
"""
    return prompt

def make_prompt_3(example_input, draft):
    prompt = f"""
You are a specialized AI assistant trained in internal medicine, tasked with refining a Hospital Course Summary. Your goal is to ensure the summary is both comprehensive and strictly accurate, based only on the patient's medical records provided.

Task:
You will receive a first-draft Hospital Course Summary and the complete medical record notes for a patient. Your task is to revise and improve the summary by:
- Verification: Confirming that every statement in the summary is directly supported by the information in the provided medical records.
- Elimination: Removing any information from the summary that cannot be directly verified in the provided medical records.
- Accuracy: Ensuring that all medical terminology is precise and correct.
- Clarity: Maintaining a concise, professional tone suitable for medical documentation.
- Structure: Ensuring the summary follows these sections:
{content}

Guidelines: 
- Strict Adherence: Only include information explicitly stated in the provided medical records. Do not add inferences or assumptions.
- Medical Terminology: Use precise medical terms and avoid non-standard abbreviations.
- Professional Tone: Write in a formal, professional style.
- Conciseness: Keep the summary focused and avoid unnecessary details.
- No External Knowledge: Rely solely on the medical records provided; do not use any other knowledge.

Input:
- Your Draft "Hospital Course Summary":
{draft}

- Entire sequence of notes for that patient: 
---
{extract_h_p(example_input)}
{"\n\n".join(extract_other_progress_notes(example_input))}
---

Output:
Provide Your Improved "Hospital Course Summary" below:
"""
    return prompt

def generate_summary(gen_model, example_input):
    drafts = []
    
    # generate first draft
    response = gen_model.generate_content([make_prompt_1(example_input)])
    previous_draft = response.candidates[0].content.parts[0].text
    drafts.append((0, previous_draft))

    # generate improved drafts by iteratively incorporating details from progress notes 1,2,3, ... not including the last note
    no_of_notes = len(extract_other_progress_notes(example_input))
    for i in range(no_of_notes):
        response = gen_model.generate_content([make_prompt_2(example_input, previous_draft, i)])
        previous_draft = response.candidates[0].content.parts[0].text
        drafts.append((i+1, previous_draft))
        
    response = gen_model.generate_content([make_prompt_3(example_input, previous_draft)])
    final_draft = response.candidates[0].content.parts[0].text
    return drafts, final_draft

class DC_summarizer:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        # For HIPAA compliance, everything remains in our Google Cloud Project
        PROJECT_ID = 'som-nero-phi-jonc101'
        self.model_instance = vertexai.init(project=PROJECT_ID)
        self.gen_model = GenerativeModel(model_name)
        print(f"""A total of {2+len(extract_other_progress_notes(example_input))} notes need to be summarized for this patient.""")  

    def summarize(self, example_input):
        self.drafts, self.final_draft = generate_summary(self.gen_model, example_input)
    
if __name__ == "__main__":    
    # Load the trainset
    trainset = pd.read_pickle('../../pickle/train_test_dfs/benchmark/trainset.pkl')
    
    # Select an example input
    ex_i = 242
    example_input = trainset["inputs"].iloc[ex_i]
    
    # Print the example input (physician's H&P and progress notes)
    print(example_input)
    
    # Instantiate the DC_summarizer class
    DC_summary_example = DC_summarizer()
    
    # Generate the drafts and final draft
    DC_summary_example.summarize(example_input)

    # Print the final draft
    print(DC_summary_example.final_draft)
    
    # Compare to the physician's summary
    print(trainset["brief_hospital_course"].iloc[ex_i])