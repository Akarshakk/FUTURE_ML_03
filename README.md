# Intelligent Resume / Candidate Screening System 🔍

A Machine Learning-based resume screening and candidate ranking system built using Natural Language Processing (NLP) and Streamlit. This tool automates the process of evaluating resumes against job descriptions, identifying key skills, computing match scores, and performing skill gap analysis.

## Deployed App Link : https://akarshakk-future-ml-03-app-j2kyli.streamlit.app

## 🚀 Features

- **Automated Resume Parsing:** Reads and extracts text from both PDF and TXT resumes.
- **NLP Skill Extraction:** Uses `spaCy` to intelligently identify noun phrases and potential skills from unstructured text.
- **Job Matching & Ranking:** Uses `Scikit-learn`'s TF-IDF Vectorizer and Cosine Similarity to score how well a candidate fits a specific job role.
- **Skill Gap Analysis:** Cross-references resume skills with job requirements to display exact matched and missing skills.
- **Interactive Dashboard:** A sleek, user-friendly UI built with Streamlit.

## 🛠️ Technology Stack

- **Python:** Core language
- **Streamlit:** Frontend dashboard
- **spaCy & NLTK:** Natural Language Processing (NLP)
- **Scikit-learn:** TF-IDF and Cosine Similarity (Machine Learning)
- **PyPDF2:** PDF text extraction
- **Pandas:** Data manipulation

## 📂 Project Structure

- `app.py`: The main Streamlit dashboard application.
- `ml_pipeline.py`: Contains the core NLP text processing, extraction, and scoring logic.
- `sample_job_descriptions.csv`: A lightweight dataset of 2,000 job descriptions used for fast lookups.
- `extract_sample.py`: A utility script used to sample data from the larger raw dataset.
- `sample_resume.txt`: A mock Software Engineer resume for testing purposes.
- `requirements.txt`: List of Python dependencies.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akarshakk/FUTURE_ML_03.git
   cd FUTURE_ML_03
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the required NLP models:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

## 🎯 How to Use

1. Launch the app in your browser (usually `http://localhost:8501`).
2. Select a target **Job Title** from the sidebar.
3. Upload a candidate's resume (PDF or TXT format).
4. Instantly view the candidate's Match Score, Matched Skills, and Missing Skills!

## ⚠️ Note on Dataset

The original `job_descriptions.csv` was 1.7GB, which is too large for GitHub and can slow down Streamlit loading times. A preprocessing script (`extract_sample.py`) was used to generate the lightweight `sample_job_descriptions.csv` (2,000 records) included in this repository.

---
*Created as part of Machine Learning Task 3 (Future Interns)*
