## ATS (Applicant Tracking System) Aligner

### Overview
The ATS Aligner is a powerful web application designed to help job seekers create tailored resumes and cover letters that align with specific job descriptions. By leveraging advanced AI models and natural language processing techniques, the application ensures that users can effectively showcase their qualifications and experiences to meet the requirements of potential employers.

### Features
- **Resume Generation**: Automatically generate a resume optimized for a specific job description.
- **Cover Letter Creation**: Create personalized cover letters that highlight key qualifications and experiences.
- **Job Description Analysis**: Analyze job descriptions to extract keywords and requirements, ensuring that resumes and cover letters are tailored accordingly.
- **Hallucination Tracking**: Identify and log potential inaccuracies in generated content to maintain the integrity of the application.
- **User-Friendly Interface**: Built with Streamlit for an intuitive user experience.

### Technologies Used
- **Streamlit**: A framework for building web applications in Python.
- **PyPDF2**: A library for reading and extracting text from PDF files.
- **Requests**: A library for making HTTP requests to external APIs.
- **Langchain**: A framework for building applications with language models.
- **Natural Language Processing**: Techniques for analyzing and generating human language.

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ats-aligner.git
   cd ats-aligner
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set Up Environment Variables**
   Create a `.env` file in the root directory and add your API keys:
   ```plaintext
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

### Usage

1. **Run the Application (Primary UI)**
   ```bash
   streamlit run streamlit_app.py
   ```

   Note: `app.py` is a legacy UI kept for reference.

2. **Navigate to the Application**
   Open your web browser and go to `http://localhost:8501`.

3. **Input Job Description**
   Choose the method of input (Text or URL) for the job description.

4. **Upload Resume**
   Upload your resume in PDF format.

5. **Generate Content**
   Use the action buttons to generate a tailored resume or cover letter based on the job description.
