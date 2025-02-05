from vertexai.preview.generative_models import GenerativeModel
import time 
import pandas as pd
import os

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

content = """Format 1: Standard Section-Based Summary:
First, provide the summary organized into the following numbered sections:

1. Reason for Admission: Clearly state the primary reason for the patient's hospitalization, as documented.
2. Relevant Medical History: Briefly summarize significant pre-existing medical conditions relevant to the admission, as documented.
3. Relevant Surgical History: Note any prior surgeries relevant to the current admission, as documented.
4. Primary Diagnosis: State the definitive diagnosis established and documented during the hospital stay.
5. Secondary Diagnoses: List any additional diagnoses made and documented during the hospitalization.
6. Key Diagnostic Investigations and Results: Mention important diagnostic tests conducted and their significant findings, as documented.
7. Therapeutic Procedures Performed: Describe any therapeutic procedures, including surgeries, performed and documented.
8. Medications: Detail new medications started, dosage changes, and medications discontinued (along with documented reasons).
9. Patient's Condition at Discharge: Summarize the patient's clinical status upon leaving the hospital, as documented.

Format 2: Problem-Based Summary:
Immediately following the Section-Based Summary, provide a summary organized by medical problem, using the following template:

Hospital Course/Significant Findings by Problem:

Problem #1: [Problem Name - e.g., Pneumonia, Heart Failure Exacerbation, etc.]
  Current Clinical Status: [Brief summary of status related to this problem at discharge]
  Discharge Plan and Goals: [Documented discharge plan for this problem, e.g., medications, follow-up]
  Outstanding/Pending Issues: [Any unresolved issues or pending investigations related to this problem at discharge]

Problem #2: [Problem Name]
  Current Clinical Status:
  Discharge Plan and Goals:
  Outstanding/Pending Issues:

... (Etc. for Problem #i) ...

Conclusion Paragraph:
Finally, after completing both Format 1 and Format 2 summaries, provide a concise, one-paragraph conclusion summarizing the overall hospital course and key outcomes."""

content_and_requirements = f"""
{content}

Requirements:
- Ensure the summary is concise yet comprehensive.
- Professional Tone: Employ language appropriate for a medical document.
- Medical Terminology: Use precise medical terminology while ensuring clarity.
- Acronyms: Avoid acronyms unless they are standard in medical documentation (e.g., ECG).
"""

def make_prompt_1(example_input):
    prompt = f"""
Role: You are an expert AI assistant specializing in internal medicine and medical documentation. Your task is to read the following first (History & Physical) and last progress note from a patient's hospital stay and to generate a concise, professional "Hospital Course Summary" for that patient.

Output Format: You are required to provide the Hospital Course Summary in two distinct formats, followed by a concluding paragraph. Please ensure you provide both formats in the order specified below:
{content_and_requirements}

---
Patient History and Physical (H&P):
{extract_h_p(example_input)}
---
Patient last Progress Note:
{extract_last_progress_note(example_input)}
---
Provide your "Hospital Course Summary" for the patient below following the guidelines and output format requirements.
"""
    return prompt

def make_prompt_2(example_input, draft, note_no):
    prompt = f"""
Role: You are an expert AI assistant specializing in internal medicine and medical documentation. Your task is to read the following medical record notes from a patient's hospital stay and create a concise yet comprehensive Hospital Course Summary for a patient

You have written a first draft of the Hospital Course Summary using the initial History & Physical and final progress note. Your task is to improve this draft by incorporating relevant details found in the additional progress note provided.

Output Format: You are required to provide the Hospital Course Summary in two distinct formats, followed by a concluding paragraph. Please ensure you provide both formats in the order specified below:
{content_and_requirements}

Your Draft "Hospital Course Summary":
{draft}

Extra Information from Other Progress Notes:
---
{extract_other_progress_notes(example_input)[note_no]}
---
Provide Your Improved "Hospital Course Summary" for the patient below following the guidelines and output format requirements.
"""
    return prompt

def make_prompt_3(example_input, draft):
    prompt = f"""
Role: You are an expert AI assistant specializing in internal medicine and medical documentation. Your primary function is to meticulously refine Hospital Course Summaries to ensure they are accurate, comprehensive within the confines of the provided records, and clinically relevant. You operate strictly based on the provided patient medical records and must not introduce external knowledge or make inferences.

Task: You will receive a first-draft Hospital Course Summary and the complete set of medical record notes for a patient's hospitalization. Your task is to revise and enhance the summary by:
- Verification: Confirming that every statement in the summary is directly supported by the information in the provided medical records.
- Elimination: Removing any information from the summary that cannot be directly verified in the provided medical records.
- Accuracy: Ensuring that all medical terminology is precise and correct.
- Clarity: Maintaining a concise, professional tone suitable for medical documentation.
- Structure: Ensuring the summary follows these sections:
{content}

Output Format: You are required to provide the Hospital Course Summary in two distinct formats, followed by a concluding paragraph. Please ensure you provide both formats in the order specified below:
{content_and_requirements}

Your Draft "Hospital Course Summary":
{draft}

Input:
- Your Draft "Hospital Course Summary":
{draft}

- Entire sequence of notes for that patient: 
---
{extract_h_p(example_input)}
{"\n\n".join(extract_other_progress_notes(example_input))}
\n\n{extract_last_progress_note(example_input)}
---

Output:
Provide Your Improved "Hospital Course Summary" for the patient below following the guidelines and output format requirements.
"""
    return prompt

def model_init():
    # Overwrite this function to use another model
    model_name="gemini-2.0-flash-exp"
    return {"loaded_model": GenerativeModel(model_name)}

def model_call(input_txt, **kwargs):
    # Overwrite this function to use another model
    ready_model = kwargs["loaded_model"]
    response = ready_model.generate_content([input_txt])
    return response.candidates[0].content.parts[0].text    

def generate_summary(gen_txt_to_txt, example_input):
    drafts = []
    
    # generate first draft
    previous_draft = gen_txt_to_txt(make_prompt_1(example_input))
    drafts.append((0, previous_draft))

    # generate improved drafts by iteratively incorporating details from progress notes 1,2,3, ... not including the last note
    no_of_notes = len(extract_other_progress_notes(example_input))
    for i in range(no_of_notes):
        previous_draft = gen_txt_to_txt(make_prompt_2(example_input, previous_draft, i))
        drafts.append((i+1, previous_draft))
        
    final_draft = gen_txt_to_txt(make_prompt_3(example_input, previous_draft))
    return drafts, final_draft

class DC_summarizer:
    def __init__(self, model_init, model_call):
        self.model_init = model_init
        self.model_call = model_call
        self.model_init_dict = model_init()
        
    def _gen_txt_to_txt(self, input_txt):
        return self.model_call(input_txt, **self.model_init_dict)

    def summarize(self, example_input, verbose=True):
        self.example_input = example_input
        self.no_of_notes = 2+len(extract_other_progress_notes(example_input))
        if verbose:
            print(f"""A total of {self.no_of_notes} notes need to be summarized for this patient.""")
        start = time.time()
        self.drafts, self.final_draft = generate_summary(self._gen_txt_to_txt, example_input)
        end = time.time()
        self.time_to_summarize = end - start # in seconds
    
if __name__ == "__main__":
    # For HIPAA compliance, everything remains in our Google Cloud Project
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../mykeys/grolleau_application_default_credentials.json'
    os.environ['GCLOUD_PROJECT'] = 'som-nero-phi-jonc101'
    
    # Load the trainset
    trainset = pd.read_pickle('../../pickle/train_test_dfs/trainset.pkl')
    
    # Select an example input
    ex_i = 15
    example_input = trainset["inputs"].iloc[ex_i]
    
    # Print the example input (physician's H&P and progress notes)
    print(example_input)
    
    # Instantiate the DC_summarizer class
    DC_summary_example = DC_summarizer(model_init, model_call)
    
    # Generate the drafts and final draft
    DC_summary_example.summarize(example_input)

    # Print the final draft
    print(DC_summary_example.final_draft)
    
    # Compare to the physician's summary
    print(trainset["brief_hospital_course"].iloc[ex_i])