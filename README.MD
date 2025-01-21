# Discharge Summarization Benchmark

This benchmark challenges participants to develop the best generative AI system for creating draft discharge summaries. Performance will be evaluated against a rigorous, physician-guided benchmark with clearly defined goals.

## Motivation

Generating high-quality draft discharge summaries with Generative AI is important because manual summarization is:

* **Time-consuming:** A tedious and repetitive task that takes away valuable time.
* **Leads to burnout:**  Contributes to "pajama time" (work done after hours).
* **Reduces patient care time:** Diverts physicians' time away from direct patient interaction and communication with other healthcare providers.

With the ever-increasing volume of medical information, computational approaches are essential for effectively managing this challenge.

## Benchmark Access (Stanford Affiliates Only)

This benchmark uses sensitive patient data from Stanford Health Care and is currently **restricted to Stanford researchers** due to HIPAA regulations.  Access requires a Stanford affiliation and installation of [Cardinal Key](https://uit.stanford.edu/service/cardinalkey/installation) for secure data access.

**To request access:**

For Stanford affiliates: Contact François Grolleau (grolleau@stanford.edu or @François Grolleau on Slack) with your intended use case.

Upon approval, you will receive access instructions for the HIPAA-compliant Google Drive folder containing the benchmark data:

👉 <a href="https://drive.google.com/drive/folders/1SJp9WQEiQ5PMGmHgS--tJt1FRBG33bbZ" target="_blank" rel="noopener">**Benchmark Data (Access Upon Approval)**</a>

## Data Description

The benchmark comprises:

* **Training Set (1500 Stanford Health Care patients):** Paired input clinical notes and corresponding physician-edited "Brief Hospital Course" summaries (silver labels).
* **Test Set (200 Stanford Health Care patients):** Input clinical notes only.

The methodology for data selection and curation is described in `dataset_preparation.md` (forthcoming).

**Data Format:**

The data is provided as a pandas DataFrame with the following columns:

* **`inputs`**: Contains the History and Physical (H&P - the first note taken at admission) and all progress notes written by physicians during the patient's hospital stay order from last note to first note.  The text is structured as follows:
> #####################
>
>h_p: UNJITTERED NOTE DATE: 2023-01-25 12:49:00
>Stanford Hospital and Clinics Preoperative History & Physical 24 Hour Interval Note
>... (H&P content) ...
>
>#####################
>
>progress_notes: UNJITTERED NOTE DATE: 2023-02-01 16:14:00
>Attending Physician Addendum:
>... (Progress note content) ...
>
>---NEXT NOTE---
>UNJITTERED NOTE DATE: 2023-02-01 10:57:00
>Coordinated Care Discharge Plans ...
>... (Progress note content) ...
>
>---NEXT NOTE---
>UNJITTERED NOTE DATE: 2023-02-01 07:16:00
>... (Progress note content) ...

* **`brief_hospital_course` (Training set only)**: This column contains the hospital course summary extracted from the general discharge summaries. Example:
>Reason for Hospitalization   Hepatic metastasis
>Brief History of Present Illness   XXX XXX is a 48 Y old woman with BRCA-negative, stage IIIC high-grade serous ovarian cancer, who underwent an exploratory laparotomy, modified radical hysterectomy with bilateral salpingo-oophorectomy, bilateral pelvic and periaortic lymphadenectomy with mobilization of the splenic flexure, omentectomy, diaphragm peritoneal resection, excision of liver lesions with a splenectomy in XX/XX/XXXX by Dr. XXX and Dr. XXX at XXX Hospital. She then was treated with 2 cycles of carboplatin and Taxol and was referred to Dr. XXX. He recommended that she undergo intraperitoneal chemotherapy, and she was treated with 7 cycles of intraperitoneal chemotherapy with cisplatin and Taxol.
>... (Brief hospital course content) ...

## Submission Instructions

1. Generate brief hospital courses for the test set.
2. Create a pickle file containing a pandas DataFrame with columns `inputs` (matching the test set `inputs`) and `predicted_brief_hospital_course`.
3. Upload your submission to the designated Submission Folder:

👉 <a href="https://drive.google.com/drive/folders/1-LjLGC8KLA6TgzBbNW1HdDlhZ54Vi046" target="_blank" rel="noopener">**Submit your predictions (Access Upon Approval)**</a>

Evaluation results will be available within 48 hours of submission.

## Acceptable Approaches

We welcome any approach for hospital course summarization, including open-weight models from Hugging Face (prompted, fine-tuned, chain-of-thought, in-context learning, etc.). Proprietary models are acceptable as long as HIPAA compliance is maintained (e.g., do not send PHI to standard LLM APIs; secure versions may be permissible).

## Evaluation Metrics

We employ a combination of automatic and rubric-based evaluation:

* **Automatic Evaluation:** NLP metrics including BLEU, ROUGE-L, and BERT-Score.
* **Rubric-Based Evaluation:**  Evaluation using pre-specified criteria defined by clinicians. An LLM is used to evaluate submitted solutions (`predicted_brief_hospital_course`) against these criteria.

## Contributing

Contributions are welcome! Code contributions (e.g., your solution's code) can be submitted via pull request. Physicians can contribute to rubric development by requesting access to our evaluation platform: 

👉 <a href="https://drive.google.com/drive/folders/XXXX" target="_blank" rel="noopener">**Contribute Your Expertise: Rubric Development (Access Upon Approval)**</a>  

Physician input is crucial for creating clinically valid rubrics.

## Contact

For any questions, suggestions, or further information, please contact François Grolleau: grolleau@stanford.edu

## Project Information

**Author:** François Grolleau

**Affiliation:** Stanford Center for Biomedical Informatics Research, Stanford University

**Last Updated:** January 20, 2025
