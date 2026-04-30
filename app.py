import streamlit as st
import pandas as pd
import tempfile
import os
from ml_pipeline import extract_text_from_pdf, clean_text, extract_skills, calculate_similarity, get_skill_gap

# Set page configuration
st.set_page_config(
    page_title="Candidate Screening System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium feel
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #6c757d;
    }
    .skill-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .skill-matched {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .skill-missing {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .section-title {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load Sample Job Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('sample_job_descriptions.csv')
    except Exception:
        # Fallback if there's an issue with the sample
        try:
            return pd.read_csv('job_descriptions.csv', nrows=2000)
        except Exception:
            # Create a dummy df to prevent crashing if neither exists
            return pd.DataFrame({
                'Job Title': ['Software Engineer', 'Data Scientist', 'Product Manager'],
                'Job Description': ['Develop web apps using python and react', 'Build machine learning models using python, sklearn, tensorflow', 'Manage product lifecycle and team collaboration'],
                'skills': ['Python React Web Development', 'Python Machine Learning Sklearn Tensorflow', 'Agile Scrum Leadership Strategy']
            })

df = load_data()

# App Header
st.title("🔍 Intelligent Resume Screening System")
st.markdown("Automate candidate evaluation using Natural Language Processing (NLP). Match resumes with job descriptions, rank candidates, and identify skill gaps.")

# Sidebar - Controls
st.sidebar.header("Configuration")
st.sidebar.markdown("Upload a resume and select a job role to see the analysis.")

# 1. Job Selection
st.sidebar.subheader("1. Select Job Role")
job_titles = df['Job Title'].dropna().unique()
selected_job_title = st.sidebar.selectbox("Job Title", job_titles)

# Filter jobs by selected title to let user choose a specific posting if there are multiple
job_postings = df[df['Job Title'] == selected_job_title]
if not job_postings.empty:
    selected_posting_idx = st.sidebar.selectbox("Specific Posting (Index)", range(len(job_postings)))
    selected_job = job_postings.iloc[selected_posting_idx]
    
    jd_text = str(selected_job.get('Job Description', ''))
    jd_skills_text = str(selected_job.get('skills', ''))
    
    st.sidebar.markdown("**Job Description Preview:**")
    st.sidebar.info(jd_text[:200] + "...")
else:
    jd_text = ""
    jd_skills_text = ""

# 2. Resume Upload
st.sidebar.subheader("2. Upload Candidate Resume")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file and jd_text:
    # Process Resume
    with st.spinner('Analyzing resume...'):
        # Extract text based on file type
        if uploaded_file.name.endswith('.pdf'):
            # Save uploaded file temporarily to read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_pdf_path = temp_file.name
            
            resume_text = extract_text_from_pdf(temp_pdf_path)
            os.remove(temp_pdf_path)
        else:
            resume_text = uploaded_file.read().decode('utf-8')
        
        # ML Pipeline Execution
        match_score = calculate_similarity(resume_text, jd_text)
        resume_skills = extract_skills(resume_text)
        matched_skills, missing_skills = get_skill_gap(resume_skills, jd_skills_text)
        
    # --- UI Layout for Results ---
    st.markdown(f"<h2 class='section-title'>Analysis Results for {selected_job_title}</h2>", unsafe_allow_html=True)
    
    # Top Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Match Score</div>
            <div class="metric-value">{match_score}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Skills Matched</div>
            <div class="metric-value" style="color: #007bff;">{len(matched_skills)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Skills Missing</div>
            <div class="metric-value" style="color: #dc3545;">{len(missing_skills)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Skills Analysis Section
    st.markdown("<h3 class='section-title'>Skill Gap Analysis</h3>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("✅ Matched Skills")
        if matched_skills:
            html_matched = "".join([f"<span class='skill-badge skill-matched'>{skill.title()}</span>" for skill in matched_skills[:30]])
            st.markdown(html_matched, unsafe_allow_html=True)
            if len(matched_skills) > 30:
                st.markdown("*(Showing top 30 matched skills)*")
        else:
            st.info("No matching skills found.")
            
    with col_b:
        st.subheader("⚠️ Missing Requirements")
        if missing_skills:
            html_missing = "".join([f"<span class='skill-badge skill-missing'>{skill.title()}</span>" for skill in missing_skills[:30]])
            st.markdown(html_missing, unsafe_allow_html=True)
            if len(missing_skills) > 30:
                st.markdown("*(Showing top 30 missing skills)*")
        else:
            st.success("No missing skills identified! Excellent match.")

    # Extracted Info Section
    with st.expander("View Extracted Resume Information"):
        st.markdown("**Raw Extracted Skills (Noun Phrases):**")
        st.write(", ".join([s.title() for s in resume_skills[:50]]))
        st.markdown("**Resume Preview:**")
        st.text(resume_text[:1000] + "\n\n... (truncated)")

elif not uploaded_file:
    # Landing Page State
    st.info("👈 Please upload a resume and select a job role from the sidebar to begin analysis.")
    
    st.markdown("""
    ### How it Works:
    1. **Text Extraction:** Uses PyPDF2 to parse text from the uploaded resume.
    2. **NLP Processing:** Uses spaCy and NLTK to clean text, remove stopwords, and extract meaningful noun phrases (skills).
    3. **Similarity Scoring:** Uses Scikit-learn's TF-IDF Vectorizer and Cosine Similarity to compare the resume against the job description.
    4. **Gap Analysis:** Cross-references extracted resume words with required job skills to identify matches and missing requirements.
    """)
    
    # Provide a sample resume download button if we create one later
    # st.download_button("Download Sample Resume", data=open('sample_resume.pdf', 'rb'), file_name='sample_resume.pdf')
