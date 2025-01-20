# Discharge Summarization Benchmark

This repository hosts a benchmark for generating draft discharge summaries using Generative AI.  Well-defined goals and evaluation benchmarks are crucial for driving scientific progress in this area.

## Motivation

Generating high-quality draft discharge summaries with Generative AI is important because manual summarization is:

* **Time-consuming:** A tedious and repetitive task that takes away valuable time.
* **Leads to burnout:**  Contributes to "pajama time" (work done after hours).
* **Reduces patient care time:** Diverts physicians' time away from direct patient interaction and communication with other healthcare providers.

With the ever-increasing volume of medical information, computational approaches are essential for effectively managing this challenge.

## Benchmark Access (Stanford Affiliates Only)

This benchmark uses sensitive patient data from Stanford Health Care and is **restricted to Stanford researchers** due to HIPAA regulations.  Access requires a Stanford affiliation and installation of Cardinal Key for secure data access.

**To request access:**

For Stanford affiliates: Contact François Grolleau (grolleau@stanford.edu or @François Grolleau on Slack) with your intended use case.

Upon approval, you will receive access instructions for the HIPAA-compliant Google Drive folder containing the benchmark data:

[**Benchmark Data (Access Upon Approval)**](https://drive.google.com/drive/folders/1SJp9WQEiQ5PMGmHgS--tJt1FRBG33bbZ){target="_blank"}

## Data Description

The benchmark comprises:

* **Training Set (1500 patients):** Paired input clinical notes and corresponding physician-edited "Brief Hospital Course" summaries (silver labels).
* **Test Set (200 patients):** Input clinical notes only.

The methodology for data selection and curation is described in [dataset_preparation.md](dataset_preparation.md) (forthcoming).

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

* **`brief_hospital_course`**: This column contains the hospital course summary extracted from the general discharge summaries. Example:
>Reason for Hospitalization   Hepatic metastasis
>Brief History of Present Illness   Imogene Most is a 48 Y old woman with BRCA-negative, stage IIIC high-grade serous ovarian cancer, who underwent an exploratory laparotomy, modified radical hysterectomy with bilateral salpingo-oophorectomy, bilateral pelvic and periaortic lymphadenectomy with mobilization of the splenic flexure, omentectomy, diaphragm peritoneal resection, excision of liver lesions with a splenectomy in 08/26/2016 by Dr. Craft and Dr. Burnell at Good Samaritan Hospital.      She then was treated with 2 cycles of carboplatin and Taxol and was referred to Dr. XXX. He recommended that she undergo intraperitoneal chemotherapy, and she was treated with 7 cycles of intraperitoneal chemotherapy with cisplatin and Taxol.
>... (Brief hospital course content) ...

## Submission Instructions

1. Generate predictions for the test set.
2. Create a pickle file containing a pandas DataFrame with columns `inputs` (matching the test set `inputs`) and `predicted_brief_hospital_course`.
3. Upload your submission to the designated Submission Folder:

[**Submit your predictions (Access Upon Approval)**](https://drive.google.com/drive/folders/1-LjLGC8KLA6TgzBbNW1HdDlhZ54Vi046){target="_blank"}

Evaluation results will be available within 48 hours of submission.

## Acceptable Approaches

We welcome any approach for hospital course summarization, including open-weight models from Hugging Face (prompted, fine-tuned, chain-of-thought, in-context learning, etc.). Proprietary models are acceptable as long as HIPAA compliance is maintained (e.g., do not send PHI to standard LLM APIs; secure versions may be permissible).

## Evaluation Metrics

We employ a combination of automatic and rubric-based evaluation:

* **Automatic Evaluation:** NLP metrics such as BLEU, ROUGE-L, and BERT-Score.
* **Rubric-Based Evaluation:**  Evaluation using pre-specified criteria defined by clinicians. An LLM is used to evaluate submitted solutions (`predicted_brief_hospital_course`) against these criteria.

## Contributing

Contributions are welcome! Code contributions (e.g., your solution's code) can be submitted via pull request.  Physicians interested in contributing to rubric development, please contact us.

## Contact

For any questions, suggestions, or further information, please contact François Grolleau: grolleau@stanford.edu

**Author:** François Grolleau

**Affiliation:** Stanford Center for Biomedical Informatics Research, Stanford University

**Last Updated:** January 17, 2025
