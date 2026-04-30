import re
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd

# Download necessary NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import stopwords

# Load English NLP model from spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def clean_text(text):
    """Cleans text by removing special characters, stopwords, and lowercasing."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def extract_skills(text):
    """
    Extracts potential skills from text using NLP noun chunks and named entities.
    Since 'skills' are not a standard entity type, we approximate by extracting Noun Phrases.
    For better results, we cross-reference with common tech and business terms.
    """
    doc = nlp(text)
    skills = set()
    
    # Extract noun chunks (phrases like "machine learning", "data analysis")
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        # Filter out very long chunks or single character chunks
        if 2 < len(phrase) < 30:
            # Basic filtering to remove obvious non-skills (e.g. pronouns)
            if not any(stop_word in phrase.split() for stop_word in ['i', 'me', 'my', 'he', 'she', 'it', 'we', 'they']):
                skills.add(phrase)
                
    # Fallback/Additional: extract single words that are NOUNs or PROPNs
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
            skills.add(token.text.lower())
            
    return list(skills)

def calculate_similarity(resume_text, job_desc_text):
    """
    Calculates the cosine similarity between the resume and the job description using TF-IDF.
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_desc_text)
    
    if not resume_clean or not jd_clean:
        return 0.0
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        # Cosine similarity between the first (resume) and second (JD) document
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)  # Return as percentage
    except ValueError:
        return 0.0

def get_skill_gap(resume_skills, jd_skills_text):
    """
    Compares resume skills to a parsed JD skills list to find gaps.
    """
    # Simply using clean_text to find overlapping words
    # A more robust system would map synonymous skills.
    jd_clean = clean_text(jd_skills_text).split()
    jd_skills = set(jd_clean)
    
    # Create a set of words from the resume skills
    resume_words = set()
    for skill in resume_skills:
        for word in skill.split():
            resume_words.add(word)
            
    # Find missing skills (words present in JD skills but not in Resume)
    missing_skills = jd_skills - resume_words
    matched_skills = jd_skills.intersection(resume_words)
    
    # Re-group some common phrases if possible, but single words work for simple demonstration
    return list(matched_skills), list(missing_skills)
