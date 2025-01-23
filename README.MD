# Discharge Summarization Benchmark

This benchmark challenges participants to develop state-of-the-art generative AI systems for automatically creating draft "Brief Hospital Course" summaries from patient clinical notes. Performance will be evaluated against a rigorous, physician-guided benchmark with clearly defined evaluation metrics.

## Motivation

Generating high-quality draft discharge summaries is crucial for improving healthcare efficiency and patient care. Manual summarization is:

* **Time-consuming:** Physicians spend valuable time on this tedious and repetitive task.
* **Burnout Inducing:**  Contributes to after-hours work ("pajama time"), increasing physician burnout.
* **Detracts from Patient Care:** Diverts physicians' time from direct patient interaction and care coordination.

Automated summarization offers a solution to this challenge, allowing physicians to focus on patient care.  With the ever-growing volume of medical information, computational approaches are essential for effectively managing this information overload.

## Benchmark Access (Stanford Affiliates Only)

This benchmark utilizes sensitive patient data from Stanford Health Care and is therefore **restricted to researchers with a Stanford affiliation** due to HIPAA regulations. Access requires a Stanford affiliation and installation of [Cardinal Key](https://uit.stanford.edu/service/cardinalkey/installation) for secure data access.

**To request access:**

Stanford affiliates should contact François Grolleau (grolleau@stanford.edu or @François Grolleau on Slack) with a brief description of their intended use case. Upon approval, you will receive access instructions for the HIPAA-compliant Google Drive folder containing the benchmark data:

👉 <a href="https://drive.google.com/drive/folders/1SJp9WQEiQ5PMGmHgS--tJt1FRBG33bbZ" target="_blank" rel="noopener">**Benchmark Data (Access Upon Approval)**</a>

## Data

The benchmark comprises:

* **Training Set (1500 Patients):** Paired input clinical notes and corresponding physician-edited "Brief Hospital Course" summaries (silver labels).
* **Test Set (200 Patients):** Input clinical notes only.

Detailed information about data selection and curation can be found in `dataset_preparation.md` (forthcoming).

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
>Brief History of Present Illness   XXX XXX is a 52 Y old woman with BRCA-negative, stage IIIB high-grade serous ovarian cancer, who underwent an exploratory laparotomy, modified radical hysterectomy with bilateral salpingo-oophorectomy 
>... (Brief hospital course content) ...

## Submission Instructions

1. Generate "Brief Hospital Course" summaries for the test set.
2. Create a pickle file containing a pandas DataFrame with columns `inputs` (matching the test set `inputs`) and `predicted_brief_hospital_course`.
3. Upload your submission to the designated Submission Folder:

👉 <a href="https://drive.google.com/drive/folders/1-LjLGC8KLA6TgzBbNW1HdDlhZ54Vi046" target="_blank" rel="noopener">**Submit your predictions (Access Upon Approval)**</a>

Evaluation results will be available within 48 hours of submission.

## Acceptable Approaches

We welcome any approach for hospital course summarization, including open-weight models from Hugging Face (prompted, fine-tuned, chain-of-thought, in-context learning, etc.). Proprietary models are acceptable as long as HIPAA compliance is maintained (e.g., do not send PHI to standard LLM APIs; secure versions may be permissible).

## Evaluation

We use a combination of automated metrics and a physician-guided rubric to evaluate submitted summaries:

* **Rubric-Based Evaluation:**  Generated summaries are assessed against clinician-defined criteria for clinical accuracy, safety, and relevance.  An LLM applies these criteria to the `predicted_brief_hospital_course` summaries, enabling consistent and efficient evaluation  while maintaining alignment with physician expertise.
* **Automatic Metrics:** Standard NLP metrics such as BLEU, ROUGE-L, and BERT-Score are used, along with task-specific clinical NLP metrics.  Details on these metrics will be provided in `evaluation_metrics.md`.

## Contributing

We welcome contributions!

* **Code Contributions:** Please feel free to share your solutions via pull requests to the `scripts/solutions` folder.
* **Physician Expertise (Rubric Development):** Physicians can contribute to rubric development by requesting access to our evaluation platform: 

👉 <a href="https://docs.google.com/spreadsheets/d/13Dy6mv3NW6eiAsEpKdY37eajPFXaF4CdknEg5Owg9zo/edit?gid=1738738529#gid=1738738529" target="_blank" rel="noopener">**Contribute Your Expertise: Rubric Development (Access Upon Approval)**</a>  

Physician input is crucial for creating clinically valid rubrics.

## Contact

For any questions, suggestions, or further information, please contact François Grolleau: grolleau@stanford.edu

## Project Information

**Author:** François Grolleau

**Affiliation:** Stanford Center for Biomedical Informatics Research, Stanford University

**Last Updated:** January 20, 2025
