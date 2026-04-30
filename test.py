from ml_pipeline import clean_text, extract_skills, calculate_similarity, get_skill_gap
import pandas as pd

# 1. Test Data Loading
df = pd.read_csv('sample_job_descriptions.csv')
job = df.iloc[0]
jd_text = str(job.get('Job Description', ''))
jd_skills_text = str(job.get('skills', ''))

# 2. Test Resume Loading
with open('sample_resume.txt', 'r') as f:
    resume_text = f.read()

# 3. Pipeline testing
score = calculate_similarity(resume_text, jd_text)
print(f"Similarity Score: {score}")

resume_skills = extract_skills(resume_text)
print(f"Extracted Resume Skills: {resume_skills[:10]}...")

matched, missing = get_skill_gap(resume_skills, jd_skills_text)
print(f"Matched Skills: {matched}")
print(f"Missing Skills: {missing}")

print("ML Pipeline works successfully!")
