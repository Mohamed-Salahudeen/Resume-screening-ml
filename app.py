import streamlit as st
import pandas as pd
import re
import string
import nltk
from io import BytesIO

# File Processing
import PyPDF2
from docx import Document
from PIL import Image
import pytesseract

# ML & NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================================================================================
# 1. SETUP AND CONFIGURATION
# =================================================================================

# FIX: Explicitly download the required NLTK data packages first.
# This prevents the AttributeError by ensuring the data is available and not corrupted.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

stop_words = set(nltk.corpus.stopwords.words('english'))

# =================================================================================
# 2. HELPER FUNCTIONS (BACKEND LOGIC)
# =================================================================================


# --- Text Extraction Functions ---
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception:
        return ""


def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception:
        return ""


def extract_text_from_image(file):
    try:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error processing image: {e}. Is Tesseract installed and in your system's PATH?")
        return ""


def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except Exception:
        return ""


# --- NLP & Analysis Functions ---
def is_resume_suspicious(text, email, phone):
    MIN_WORD_COUNT = 50
    word_count = len(str(text).split())
    if word_count < MIN_WORD_COUNT or (email == "Not Found" and phone == "Not Found"):
        return True
    return False


def extract_contact_info(text):
    # More robust regex for names (looks for 2-3 capitalized words)
    name_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})'
    name = re.search(name_pattern, text)
    
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    
    # More flexible phone regex
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone = re.search(phone_pattern, text)
    
    return (
        name.group(0).strip() if name else "Not Found",
        email.group(0) if email else "Not Found",
        phone.group(0) if phone else "Not Found"
    )


def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = nltk.word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words and word.isalpha()])


def calculate_similarity(jd_text, resume_text):
    if not jd_text or not resume_text: return 0.0
    corpus = [jd_text, resume_text]
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        if tfidf_matrix.shape[1] == 0:  # Check for empty vocabulary
            return 0.0
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        return 0.0


# =================================================================================
# 3. STREAMLIT UI (FRONTEND)
# =================================================================================
st.set_page_config(page_title="AI Resume Screening", page_icon="🎯", layout="wide")
st.title("🎯 AI Resume Screening")
st.markdown("Upload resumes in any format (`PDF`, `DOCX`, `CSV`, `JPG`, etc.), filter out suspicious entries, and shortlist top candidates.")
st.markdown("---")

# --- UI Layout ---
st.header("1. Job Description")
job_description = st.text_area("Paste the job description here:", height=200, placeholder="e.g., Senior Data Scientist...")
st.header("2. Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload a mix of resumes (PDF, DOCX, TXT, CSV, JPG, PNG)",
    type=['pdf', 'docx', 'txt', 'csv', 'jpg', 'png', 'jpeg'],
    accept_multiple_files=True
)

# --- Conditional UI for CSV Column Mapping ---
csv_files = [f for f in uploaded_files if f.name.endswith('.csv')]
csv_column_map = {}
if csv_files:
    st.info("CSV file detected. Please map the column containing resume text.")
    try:
        df_sample = pd.read_csv(csv_files[0])
        resume_text_col = st.selectbox(
            "**Which column in your CSV contains the resume text?**",
            options=df_sample.columns.tolist()
        )
        csv_column_map['text_column'] = resume_text_col
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

st.header("3. Configure Shortlisting")
filter_fakes = st.toggle("Filter out suspicious resumes", value=True, help="Removes resumes that are very short or lack contact info.")
top_percentage = st.slider("Show only the top % of candidates", 5, 100, 20, 5, format="%d%%")

if st.button("Shortlist Candidates", type="primary", use_container_width=True):
    if not job_description:
        st.warning("Please provide a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume file.")
    else:
        with st.spinner("Processing files and analyzing resumes... This may take a moment. ⏳"):
            all_resumes_data = []
            
            # --- Step 1: Extract text from all files ---
            for file in uploaded_files:
                filename = file.name
                file_content = BytesIO(file.getvalue())  # Use BytesIO for universal handling
                if filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file_content)
                    all_resumes_data.append({'source': filename, 'text': text})
                elif filename.endswith('.docx'):
                    text = extract_text_from_docx(file_content)
                    all_resumes_data.append({'source': filename, 'text': text})
                elif filename.endswith('.txt'):
                    text = extract_text_from_txt(file_content)
                    all_resumes_data.append({'source': filename, 'text': text})
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    text = extract_text_from_image(file_content)
                    all_resumes_data.append({'source': filename, 'text': text})
                elif filename.endswith('.csv'):
                    if csv_column_map.get('text_column'):
                        df = pd.read_csv(file_content)
                        if csv_column_map['text_column'] in df.columns:
                            for i, row in df.iterrows():
                                all_resumes_data.append({
                                    'source': f"{filename} - Row {i+1}",
                                    'text': str(row[csv_column_map['text_column']])
                                })
            
            # --- Step 2: Process the extracted text ---
            processed_jd = preprocess_text(job_description)
            valid_results = []
            suspicious_count = 0
            
            for resume in all_resumes_data:
                raw_text = resume['text']
                if not raw_text or raw_text.isspace():
                    suspicious_count += 1
                    continue
                
                name, email, phone = extract_contact_info(raw_text)
                
                if filter_fakes and is_resume_suspicious(raw_text, email, phone):
                    suspicious_count += 1
                    continue
                    
                processed_resume = preprocess_text(raw_text)
                score = calculate_similarity(processed_jd, processed_resume)
                
                valid_results.append({
                    "Candidate Name": name, "Source": resume['source'], "Similarity Score": score,
                    "Email": email, "Phone Number": phone, "Resume Text": raw_text
                })
            
            # --- Step 3: Display Results ---
            if valid_results:
                st.markdown("---")
                st.header("🏆 Shortlisted Candidates")
                
                sorted_results = sorted(valid_results, key=lambda x: x["Similarity Score"], reverse=True)
                num_to_show = max(1, int(len(sorted_results) * (top_percentage / 100.0)))
                final_results = sorted_results[:num_to_show]
                
                st.success(f"Processed {len(all_resumes_data)} entries. Shortlisted **{len(final_results)} candidates** (Top {top_percentage}%). {suspicious_count} entries were filtered out.")
                
                for i, result in enumerate(final_results):
                    st.markdown(f"### **Rank {i+1}: {result['Candidate Name']}** (`{result['Source']}`)")
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**📧 Email:** {result['Email']} | **📞 Phone:** {result['Phone Number']}")
                        with col2:
                            st.metric(label="**Match Score**", value=f"{result['Similarity Score'] * 100:.2f}%")
                        
                        with st.expander("📄 View Full Resume Text"):
                            st.text(result['Resume Text'])
            else:
                st.warning(f"No valid resumes found to display after filtering. {suspicious_count} entries were removed from a total of {len(all_resumes_data)}.")
