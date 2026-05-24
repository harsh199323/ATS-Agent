try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import os
import re
try:
    import requests
except Exception:
    requests = None
import shutil
import subprocess
try:
    import PyPDF2 as pdf
except Exception:
    pdf = None
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    # Prefer community chat models; different installs expose providers differently.
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None

try:
    # Local Llama/C++ provider
    from langchain_community.chat_models.llamacpp import ChatLlamaCpp
except Exception:
    try:
        from langchain_community.chat_models import ChatLlamaCpp
    except Exception:
        ChatLlamaCpp = None

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
    )
except Exception:
    StrOutputParser = None
    ChatPromptTemplate = None
    SystemMessagePromptTemplate = None
    HumanMessagePromptTemplate = None

from config import AVAILABLE_MODELS, SCORING_WEIGHTS, EMBEDDING_MODEL_NAME, SKILL_LEXICON

MAX_JD_CHARS = 5000
MAX_RESUME_CHARS = 8000
MAX_PROFILE_CHARS = 4000

# Create a helper to instantiate a ChatOllama model with common parameters
def get_llm_instance(model_name: str):
    """Instantiate a local chat model (Ollama or LlamaCPP) if available.

    Tries providers in order: ChatOllama, ChatLlamaCpp. Returns None when
    no supported provider is installed.
    """
    # Prefer Deepseek HTTP API when an API key is present.
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    if deepseek_key:
        try:
            if requests is not None:
                class DeepseekHTTPModel:
                    def __init__(self, model: str):
                        self.model = model
                        self.is_deepseek = True

                    def run_prompt(self, prompt_text: str, timeout: int = 60) -> str:
                        try:
                            headers = {
                                "Authorization": f"Bearer {deepseek_key}",
                                "Content-Type": "application/json",
                            }
                            payload = {
                                "model": self.model,
                                "messages": [{"role": "user", "content": prompt_text}],
                                "max_tokens": 512,
                                "temperature": 0.5,
                            }
                            resp = requests.post(deepseek_url, json=payload, headers=headers, timeout=timeout)
                            resp.raise_for_status()
                            body = resp.json()
                            # Support common response shapes used by chat-completion APIs
                            if isinstance(body, dict):
                                # OpenAI-like
                                if "choices" in body and body["choices"]:
                                    first = body["choices"][0]
                                    # Chat-style
                                    if isinstance(first.get("message"), dict):
                                        return first["message"].get("content", "").strip()
                                    # Text-style
                                    if "text" in first:
                                        return first.get("text", "").strip()
                                # direct text field
                                if "text" in body:
                                    return body.get("text", "").strip()
                            # Fallback to raw text
                            return str(body)
                        except Exception as e:
                            return f"Deepseek API error: {str(e)}"

                # Quick health-check: ensure the API key is usable before returning
                try:
                    test_client = DeepseekHTTPModel(model_name)
                    health = test_client.run_prompt("Say hello", timeout=6)
                    low = (health or "").lower()
                    if low and not low.startswith("deepseek api error") and "payment required" not in low and "error:" not in low:
                        return test_client
                except Exception:
                    pass
        except Exception:
            pass

    # Try Ollama provider first
    if ChatOllama is not None:
        try:
            return ChatOllama(
                model=model_name,
                temperature=0.5,
                top_k=40,
                top_p=0.85,
                repeat_penalty=1.15,
                num_ctx=2048,
                num_predict=512
            )
        except Exception:
            pass

    # Try LlamaCPP local provider
    if ChatLlamaCpp is not None:
        try:
            # ChatLlamaCpp wrappers may accept different argument names; try common ones.
            try:
                return ChatLlamaCpp(model=model_name, n_ctx=2048, n_predict=512, temperature=0.5)
            except TypeError:
                return ChatLlamaCpp(model=model_name, num_ctx=2048, num_predict=512, temperature=0.5)
        except Exception:
            pass

    # If neither LangChain provider is installed, fall back to the Ollama CLI if present.
    try:
        if model_name and shutil.which("ollama"):
            # Lightweight wrapper that calls the local ollama CLI. This avoids a hard
            # dependency on the langchain ChatOllama class while still enabling local
            # model usage when ollama is installed and running.
            class OllamaCLIModel:
                def __init__(self, model: str):
                    self.model = model
                    # marker used by generate_content to select CLI path
                    self.is_ollama_cli = True

                def run_prompt(self, prompt_text: str, timeout: int = 60) -> str:
                    try:
                        # Use ollama run <model> "prompt text" to generate text.
                        # Pass the prompt as a positional argument to avoid flag
                        # incompatibilities across ollama versions.
                        # Capture raw bytes and decode explicitly to avoid
                        # platform-dependent decoding errors (cp1252 on Windows).
                        proc = subprocess.run(
                            ["ollama", "run", self.model, prompt_text],
                            capture_output=True,
                            text=False,
                            timeout=timeout
                        )
                        out_bytes = proc.stdout or b""
                        err_bytes = proc.stderr or b""
                        try:
                            out = out_bytes.decode("utf-8")
                        except Exception:
                            out = out_bytes.decode("utf-8", errors="replace")
                        out = out.strip()
                        if out:
                            return out
                        # If stdout is empty, decode and return stderr to aid debugging.
                        try:
                            err = err_bytes.decode("utf-8")
                        except Exception:
                            err = err_bytes.decode("utf-8", errors="replace")
                        return err.strip() or ""
                    except Exception as e:
                        return f"Ollama CLI error: {str(e)}"

                def run_prompt_stream(self, prompt_text: str, timeout: int = 60):
                    """Stream output from the ollama CLI as it arrives.

                    Yields decoded UTF-8 text chunks.
                    """
                    try:
                        proc = subprocess.Popen(
                            ["ollama", "run", self.model, prompt_text],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=1
                        )
                        # Read raw bytes and decode incrementally
                        assert proc.stdout is not None
                        while True:
                            chunk = proc.stdout.read(1024)
                            if not chunk:
                                break
                            try:
                                text = chunk.decode("utf-8")
                            except Exception:
                                text = chunk.decode("utf-8", errors="replace")
                            yield text
                        proc.wait(timeout=timeout)
                    except Exception as e:
                        yield f"Ollama CLI stream error: {str(e)}"

            return OllamaCLIModel(model_name)
    except Exception:
        pass

    return None

# Instantiate models based on config (tolerant to missing LLM libs).
llm_embedding = get_llm_instance(AVAILABLE_MODELS.get("embedding"))
llm_general_light = get_llm_instance(AVAILABLE_MODELS.get("general", {}).get("light"))
llm_general_heavy = get_llm_instance(AVAILABLE_MODELS.get("general", {}).get("heavy"))
# For technical tasks, here we choose "data_science" as an example; you may extend this logic.
llm_technical_light = get_llm_instance(
    AVAILABLE_MODELS.get("technical", {}).get("data_science", {}).get("light")
)
llm_technical_heavy = get_llm_instance(
    AVAILABLE_MODELS.get("technical", {}).get("data_science", {}).get("heavy")
)
llm_creative_polish = get_llm_instance(AVAILABLE_MODELS.get("creative_polish"))

output_parser = StrOutputParser() if StrOutputParser is not None else None

_NLP = None
_EMBEDDING_MODEL: Optional[Any] = None

_SIMPLE_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "our", "will", "from", "that",
    "this", "are", "was", "were", "has", "have", "had", "into", "over", "under",
    "about", "across", "more", "most", "such", "including", "within", "into",
    "using", "used", "use", "per", "via", "all", "any", "each", "other"
}

# Define system prompts for each task.
RESUME_SYSTEM_PROMPT = """You are an expert ATS analyst. Create a resume that exactly matches the job description using ONLY provided information.

Rules:
1. Extract keywords from job description
2. Verify every claim against the resume
3. Never invent information
4. Structure:
   [Contact Info]
   [Tailored Summary]
   [Skills (prioritized by JD keywords)]
   [Experience (reworded to match JD)]
   [Education]
5. Tag each match: [MATCH] 
6. Reject unverifiable information"""

COVER_LETTER_SYSTEM_PROMPT = """Create cover letters demonstrating exact qualification matches.

Requirements:
1. Open with specific JD reference
2. Map resume items to JD requirements
3. Highlight unique value
4. Closing with JD-specific motivation

Rules:
- Each claim must show [RESUME EVIDENCE]
- Each JD connection must show [JD REQUIREMENT]
- No unverifiable information"""

ANALYSIS_SYSTEM_PROMPT = """Analyze resume-JD match with precision:

1. Calculate match percentage (0-100%)
2. List missing keywords
3. Identify resume strengths
4. Suggest improvements
5. Provide overall assessment

Format:
Match Percentage: X%
Missing Keywords: [list]
Strengths: [list]
Improvements: [list]
Assessment: [text]"""

def track_hallucinations(response: str) -> str:
    """Analyze response for potential hallucinations."""
    hallucinations = []
    if "[INFERENCE]" in response:
        hallucinations.append("Made inferences beyond source material")
    if "invent" in response.lower() or "assume" in response.lower():
        hallucinations.append("Used speculative language")
    if "Not specified" in response:
        hallucinations.append("Identified missing information")
    # (In a Streamlit app, you can log these to session state.)
    return response


def find_name_in_resume(resume: str) -> Optional[str]:
    """Try to heuristically find the candidate's name in the resume text.

    Looks for a short line with 1-3 capitalized words and returns it.
    """
    for line in resume.splitlines():
        ln = line.strip()
        if not ln or len(ln) > 60:
            continue
        # Match typical name-like lines e.g. "Jane Doe" or "John A. Smith"
        if re.match(r"^[A-Z][a-z]+(?:[ \-][A-Z][a-z\.]+){0,2}$", ln):
            return ln
    return None


def extract_resume_contacts(resume: str) -> Dict[str, set]:
    """Extract emails and phone-like tokens from the resume for verification."""
    emails = set(re.findall(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", resume))
    phones = set(re.findall(r"\+?\d[\d\-\s\(\)]{6,}\d", resume))
    # Lowercase copies for simple containment checks
    return {"emails": emails, "phones": phones}


def sanitize_output_against_resume(output: str, resume: str) -> str:
    """Remove or blank out invented contact info and replace placeholder names.

    - Removes lines containing Address/Phone/Email if those specific values are not
      present in the provided resume text.
    - Replaces common placeholders like 'John Doe' or 'Candidate' with the name
      found in the resume when possible.
    """
    contacts = extract_resume_contacts(resume)
    found_name = find_name_in_resume(resume)

    sanitized_lines: List[str] = []
    for line in output.splitlines():
        low = line.lower().strip()
        # Remove contact lines if the contained value does not appear in resume
        if low.startswith("address:") or low.startswith("phone:") or low.startswith("email:"):
            # extract the value portion
            parts = line.split(":", 1)
            if len(parts) > 1:
                val = parts[1].strip()
                # if an email present and not in resume, skip the line
                if re.search(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", val):
                    if val not in contacts["emails"]:
                        continue
                # phone-like
                if re.search(r"\+?\d[\d\-\s\(\)]{6,}\d", val):
                    if val not in contacts["phones"]:
                        continue
            # otherwise keep the line if no clear mismatch
            sanitized_lines.append(line)
            continue

        # Replace common placeholder names with the real name when available
        if found_name:
            # replace exact matches of John Doe / Candidate etc.
            if re.search(r"\bJohn Doe\b", line):
                line = re.sub(r"\bJohn Doe\b", found_name, line)
            if re.search(r"\bCandidate\b", line, flags=re.IGNORECASE):
                line = re.sub(r"\bCandidate\b", found_name, line, flags=re.IGNORECASE)

        sanitized_lines.append(line)

    return "\n".join(sanitized_lines)

def generate_content(prompt_template: str, inputs: Dict[str, Any], system_prompt: str, model_instance: ChatOllama) -> str:
    """Generate content using the provided model instance and LangChain prompt chaining."""
    try:
        # If the model instance is the Ollama CLI wrapper, format the prompt
        # and call the CLI directly. This keeps a lightweight dependency surface
        # and works even when langchain ChatOllama is not installed.
        if model_instance is None:
            return generate_content_fallback(prompt_template, inputs, system_prompt)

        # If the model instance exposes a run_prompt method (Deepseek HTTP wrapper
        # or Ollama CLI wrapper), use it directly.
        if hasattr(model_instance, "run_prompt"):
            # Render the user-facing prompt by formatting template with inputs
            try:
                formatted = prompt_template.format(**inputs)
            except Exception:
                # If formatting fails, fall back to a safe join of inputs
                formatted = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
            full_prompt = system_prompt + "\n\n" + formatted
            out = model_instance.run_prompt(full_prompt)
            return track_hallucinations(out)

        # Otherwise try to use LangChain prompt chaining if available.
        if (
            ChatPromptTemplate is None
            or SystemMessagePromptTemplate is None
            or HumanMessagePromptTemplate is None
            or output_parser is None
        ):
            return generate_content_fallback(prompt_template, inputs, system_prompt)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(prompt_template)
        ])
        chain = prompt | model_instance | output_parser
        response = chain.invoke(inputs)
        return track_hallucinations(response)
    except Exception as e:
        return f"Generation error: {str(e)}"


def stream_generate_content(prompt_template: str, inputs: Dict[str, Any], system_prompt: str, model_instance: Any):
    """Yield generation chunks as they become available.

    - If model_instance exposes run_prompt_stream, yield chunks from it.
    - Otherwise fall back to a single final string via generate_content.
    """
    try:
        # Render prompt
        try:
            formatted = prompt_template.format(**inputs)
        except Exception:
            formatted = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
        full_prompt = system_prompt + "\n\n" + formatted

        if hasattr(model_instance, "run_prompt_stream"):
            for chunk in model_instance.run_prompt_stream(full_prompt):
                yield chunk
            return

        # Fallback: single-shot generation
        out = generate_content(prompt_template, inputs, system_prompt, model_instance)
        yield out
    except Exception as e:
        yield f"Generation stream error: {str(e)}"


def stream_generate_resume(jd: str, resume: str, profile: str, keywords: list, low_memory: bool = False):
    """Stream resume generation. Yields intermediate chunks and final sanitized output."""
    prompt = """
    Create resume optimized for this job:
    Job Description: {jd}
    Current Resume: {resume}
    Additional Profile: {profile}
    Keywords to emphasize: {keywords}
    """
    inputs = {
        "jd": clamp_text(jd, MAX_JD_CHARS),
        "resume": clamp_text(resume, MAX_RESUME_CHARS),
        "profile": clamp_text(profile, MAX_PROFILE_CHARS),
        "keywords": ", ".join(keywords),
    }
    # select model
    model = llm_technical_heavy if not low_memory else llm_technical_light
    # Stream generation
    buffer = ""
    for chunk in stream_generate_content(prompt, inputs, RESUME_SYSTEM_PROMPT, model):
        buffer += chunk
        yield chunk

    # After stream ends, sanitize and yield final replacement marker
    try:
        sanitized = sanitize_output_against_resume(buffer, resume)
        # If sanitized differs, yield the sanitized version as the final chunk
        if sanitized != buffer:
            yield "\n" + sanitized
    except Exception:
        return


def stream_generate_cover_letter(jd: str, resume: str, low_memory: bool = False):
    prompt = """
    Write cover letter for:
    Job Description: {jd}
    Resume: {resume}
    """
    inputs = {"jd": clamp_text(jd, MAX_JD_CHARS), "resume": clamp_text(resume, MAX_RESUME_CHARS)}
    model = llm_general_heavy if not low_memory else llm_general_light
    buffer = ""
    for chunk in stream_generate_content(prompt, inputs, COVER_LETTER_SYSTEM_PROMPT, model):
        buffer += chunk
        yield chunk

    try:
        sanitized = sanitize_output_against_resume(buffer, resume)
        if sanitized != buffer:
            yield "\n" + sanitized
    except Exception:
        return


def generate_content_fallback(prompt_template: str, inputs: Dict[str, Any], system_prompt: str) -> str:
    """Deterministic fallback generator when LLMs are unavailable.

    This produces a concise, template-driven resume / cover letter / analysis
    using extracted keywords and simple heuristics so the UI remains useful
    without heavy LLM dependencies.
    """
    try:
        jd = inputs.get("jd", "") or inputs.get("job_description", "")
        resume = inputs.get("resume", "")
        profile = inputs.get("profile", "")
        keywords = inputs.get("keywords") or inputs.get("keywords", "")
        if isinstance(keywords, str):
            keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        else:
            keywords_list = list(keywords) if keywords is not None else []

        # Resume fallback: create a tailored summary + prioritized skills + evidence bullets
        if system_prompt == RESUME_SYSTEM_PROMPT or "resume" in system_prompt.lower():
            top_kw = keywords_list[:8]
            summary_parts = []
            if profile:
                summary_parts.append(profile.strip())
            if top_kw:
                summary_parts.append("Key skills: " + ", ".join(top_kw))
            summary = " - ".join(summary_parts) if summary_parts else "Tailored candidate summary."

            # Pull up to 6 short experience lines from the resume
            bullets = []
            for line in resume.splitlines():
                ln = line.strip()
                if not ln:
                    continue
                if len(bullets) >= 6:
                    break
                # keep reasonably short lines
                bullets.append(ln if len(ln) <= 200 else ln[:197] + "...")

            skills_section = ("Skills: " + ", ".join(top_kw)) if top_kw else "Skills: [none extracted]"
            generated = f"[Contact Info]\n{summary}\n\n{skills_section}\n\n[Experience]\n"
            for b in bullets:
                generated += f"- {b}\n"
            generated += "\n[Notes] This is a deterministic fallback summary. For richer results, install an LLM provider."
            return generated

        # Cover letter fallback
        if system_prompt == COVER_LETTER_SYSTEM_PROMPT or "cover" in system_prompt.lower():
            opening = "Dear Hiring Team,\n\nI am writing to express interest in the role described."
            focus = "I bring experience in " + (", ".join(keywords_list[:3]) if keywords_list else "relevant areas") + "."
            evidence = "From my resume: "
            # use first two resume lines as evidence
            evidence_lines = [ln for ln in (resume.splitlines()[:2]) if ln.strip()]
            if evidence_lines:
                evidence += "; ".join(evidence_lines)
            closing = "\n\nSincerely,\nCandidate"
            return opening + "\n\n" + focus + "\n\n" + evidence + closing

        # Analysis fallback
        if system_prompt == ANALYSIS_SYSTEM_PROMPT or "analy" in system_prompt.lower():
            # compute simple keyword overlap
            jd_tokens = set([w.lower() for w in re.findall(r"\w{3,}", jd)])
            resume_tokens = set([w.lower() for w in re.findall(r"\w{3,}", resume)])
            overlap = jd_tokens & resume_tokens
            pct = int((len(overlap) / max(1, len(jd_tokens))) * 100)
            missing = sorted(list(jd_tokens - resume_tokens))[:20]
            strengths = sorted(list(overlap))[:10]
            return (
                f"Match Percentage: {pct}%\n"
                f"Missing Keywords: {', '.join(missing) if missing else 'None'}\n"
                f"Strengths: {', '.join(strengths) if strengths else 'None'}\n"
                f"Improvements: Add more JD keywords to experience and skills sections."
            )

        # Fallback default
        return "Generation unavailable: no LLM found and no deterministic fallback matched."
    except Exception as e:
        return f"Generation fallback error: {str(e)}"

def generate_with_fallback(
    prompt_template: str,
    inputs: Dict[str, Any],
    system_prompt: str,
    primary_model: ChatOllama,
    fallback_model: ChatOllama
) -> str:
    """Retry with a lighter model when the primary model exceeds memory."""
    result = generate_content(prompt_template, inputs, system_prompt, primary_model)
    lower = result.lower() if isinstance(result, str) else ""
    # If the model indicates memory issues or returns an error, try the fallback.
    if (
        "requires more system memory" in lower
        or lower.startswith("generation error")
        or "deepseek api error" in lower
        or "error:" in lower
    ):
        return generate_content(prompt_template, inputs, system_prompt, fallback_model)
    return result

def input_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF file."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        return f"PDF error: {str(e)}"

def extract_text_from_url(url: str) -> str:
    """Extract text from a job posting URL with basic hardening."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (ATS-Aligner/1.0; +https://example.com)"
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text" not in content_type and "html" not in content_type:
            return "URL error: unsupported content type"
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        clean = soup.get_text(separator=" ", strip=True)
        return clean[:5000]
    except Exception as e:
        return f"URL error: {str(e)}"

def get_nlp() -> Optional[Any]:
    """Load spaCy model with a lightweight fallback."""
    global _NLP
    if _NLP is None:
        try:
            os.environ.setdefault("THINC_NO_PYTORCH", "1")
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                _NLP = spacy.load("en_core_web_sm")
        except Exception:
            return None
    return _NLP

def get_embedding_model() -> Optional[Any]:
    """Load sentence-transformer model once, or return None if unavailable."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:
            return None
    return _EMBEDDING_MODEL

def normalize_text(text: str) -> str:
    """Normalize text for matching and vectorization."""
    cleaned = re.sub(r"[^\w\s\-\+\.]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()

def get_tfidf_vectorizer() -> Tuple[Optional[Any], Optional[Any]]:
    """Return TF-IDF vectorizer and stopwords if sklearn is available."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
        return TfidfVectorizer, ENGLISH_STOP_WORDS
    except Exception:
        return None, None

def simple_tokenize(text: str, stopwords: Optional[set] = None) -> List[str]:
    """Tokenize text with a basic regex and stopword filter."""
    stop = stopwords or _SIMPLE_STOPWORDS
    return [w for w in re.findall(r"\w{3,}", text.lower()) if w not in stop]

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two 2D arrays."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

def extract_phrases(doc: Any) -> List[str]:
    """Extract noun-phrase and entity phrases."""
    phrases: List[str] = []
    for chunk in doc.noun_chunks:
        tokens = [t.lemma_.lower() for t in chunk if not t.is_stop and not t.is_punct]
        if 2 <= len(tokens) <= 5:
            phrases.append(" ".join(tokens))
    for ent in doc.ents:
        tokens = [t.lemma_.lower() for t in ent if not t.is_stop and not t.is_punct]
        if 2 <= len(tokens) <= 5:
            phrases.append(" ".join(tokens))
    return phrases

def extract_keywords(text: str, n: int = 15) -> Dict[str, List[str]]:
    """Extract top keywords and phrases from text."""
    nlp = get_nlp()
    if nlp is None:
        _, stopwords = get_tfidf_vectorizer()
        stop = set(stopwords) if stopwords else _SIMPLE_STOPWORDS
        words = [w for w in re.findall(r"\w{3,}", text.lower()) if w not in stop]
        counts: Dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        top_keywords = sorted(counts, key=counts.get, reverse=True)[:n]
        return {"keywords": top_keywords, "phrases": []}

    doc = nlp(text)
    token_counts: Dict[str, int] = {}
    phrase_counts: Dict[str, int] = {}

    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.lower()
        if len(lemma) < 3:
            continue
        token_counts[lemma] = token_counts.get(lemma, 0) + 1

    for phrase in extract_phrases(doc):
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    top_keywords = sorted(token_counts, key=token_counts.get, reverse=True)[:n]
    top_phrases = sorted(phrase_counts, key=phrase_counts.get, reverse=True)[:n]
    return {"keywords": top_keywords, "phrases": top_phrases}

def extract_years_experience(text: str) -> int:
    """Extract the maximum years-of-experience requirement if present."""
    matches = re.findall(r"(\d+)\s*\+?\s*(?:years|yrs)", text.lower())
    return max((int(m) for m in matches), default=0)

def extract_skill_entities(text: str) -> Dict[str, Any]:
    """Extract skills/tools and years of experience from text."""
    lowered = text.lower()
    skills = set()
    for skill in SKILL_LEXICON:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lowered):
            skills.add(skill)
    years = extract_years_experience(lowered)
    return {"skills": skills, "years": years}

def split_sections(text: str, max_lines: int = 3, max_sections: int = 20) -> List[str]:
    """Split text into compact sections for semantic similarity."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: List[str] = []
    current: List[str] = []

    def is_heading(line: str) -> bool:
        return (line.isupper() and len(line) <= 60) or (line.endswith(":") and len(line) <= 80)

    for line in lines:
        if is_heading(line) and current:
            sections.append(" ".join(current))
            current = [line]
            continue
        current.append(line)
        if len(current) >= max_lines:
            sections.append(" ".join(current))
            current = []

    if current:
        sections.append(" ".join(current))

    return sections[:max_sections]

def compute_semantic_similarity(jd_text: str, resume_text: str, top_k: int = 5) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute semantic similarity and return top section matches."""
    jd_sections = split_sections(jd_text)
    resume_sections = split_sections(resume_text)
    if not jd_sections or not resume_sections:
        return 0.0, []

    model = get_embedding_model()
    if model is None:
        vectorizer_cls, _ = get_tfidf_vectorizer()
        if vectorizer_cls is not None:
            vectorizer = vectorizer_cls(stop_words="english", ngram_range=(1, 2), max_features=5000)
            tfidf = vectorizer.fit_transform(jd_sections + resume_sections)
            jd_vecs = tfidf[: len(jd_sections)].toarray()
            resume_vecs = tfidf[len(jd_sections) :].toarray()
            sim_matrix = cosine_matrix(jd_vecs, resume_vecs)
        else:
            sim_matrix = np.zeros((len(jd_sections), len(resume_sections)))
            for i, jd_section in enumerate(jd_sections):
                jd_tokens = set(simple_tokenize(jd_section))
                for j, resume_section in enumerate(resume_sections):
                    resume_tokens = set(simple_tokenize(resume_section))
                    union = jd_tokens | resume_tokens
                    sim_matrix[i, j] = (len(jd_tokens & resume_tokens) / len(union)) if union else 0.0
    else:
        jd_emb = model.encode(jd_sections, normalize_embeddings=True)
        resume_emb = model.encode(resume_sections, normalize_embeddings=True)
        sim_matrix = cosine_matrix(np.array(jd_emb), np.array(resume_emb))

    top_pairs: List[Dict[str, Any]] = []
    max_scores = sim_matrix.max(axis=1)
    avg_max = float(max_scores.mean())

    for i, score in enumerate(max_scores):
        j = int(sim_matrix[i].argmax())
        top_pairs.append({
            "jd_section": jd_sections[i],
            "resume_section": resume_sections[j],
            "score": float(score)
        })

    top_pairs = sorted(top_pairs, key=lambda x: x["score"], reverse=True)[:top_k]
    return avg_max, top_pairs

def compute_tfidf_similarity(jd_text: str, resume_text: str) -> float:
    """Compute TF-IDF cosine similarity as a keyword compliance signal."""
    cleaned_jd = normalize_text(jd_text)
    cleaned_resume = normalize_text(resume_text)
    if not cleaned_jd or not cleaned_resume:
        return 0.0
    vectorizer_cls, _ = get_tfidf_vectorizer()
    if vectorizer_cls is None:
        jd_tokens = set(simple_tokenize(cleaned_jd))
        resume_tokens = set(simple_tokenize(cleaned_resume))
        union = jd_tokens | resume_tokens
        return (len(jd_tokens & resume_tokens) / len(union)) if union else 0.0
    vectorizer = vectorizer_cls(stop_words="english", ngram_range=(1, 2), max_features=5000)
    tfidf = vectorizer.fit_transform([cleaned_jd, cleaned_resume]).toarray()
    sim = cosine_matrix(tfidf[:1], tfidf[1:2])
    return float(sim[0][0])

def compute_match_scores(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """Compute hybrid match scores and diagnostics."""
    semantic_score, top_matches = compute_semantic_similarity(jd_text, resume_text)
    tfidf_score = compute_tfidf_similarity(jd_text, resume_text)

    jd_entities = extract_skill_entities(jd_text)
    resume_entities = extract_skill_entities(resume_text)
    jd_skills = jd_entities["skills"]
    resume_skills = resume_entities["skills"]
    overlap = jd_skills.intersection(resume_skills)
    skill_overlap = (len(overlap) / len(jd_skills)) if jd_skills else 0.0

    jd_years = jd_entities["years"]
    resume_years = resume_entities["years"]
    experience_match = 0.0
    if jd_years > 0:
        experience_match = min(resume_years / jd_years, 1.0) if resume_years > 0 else 0.0

    ontology_boost = 1.0 if skill_overlap >= 0.30 else 0.0

    overall = (
        SCORING_WEIGHTS["semantic"] * semantic_score +
        SCORING_WEIGHTS["skill_overlap"] * skill_overlap +
        SCORING_WEIGHTS["tfidf"] * tfidf_score +
        SCORING_WEIGHTS["experience"] * experience_match +
        SCORING_WEIGHTS["ontology"] * ontology_boost
    )
    overall_score = max(0.0, min(100.0, overall * 100))

    missing_skills = sorted(jd_skills - resume_skills)
    return {
        "overall_score": overall_score,
        "semantic_score": semantic_score,
        "tfidf_score": tfidf_score,
        "skill_overlap": skill_overlap,
        "experience_match": experience_match,
        "ontology_boost": ontology_boost,
        "missing_skills": missing_skills,
        "top_matches": top_matches
    }

def clamp_text(text: str, max_chars: int) -> str:
    """Reduce large inputs to keep model calls responsive."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:max_chars]

def generate_resume(jd: str, resume: str, profile: str, keywords: list, low_memory: bool = False) -> str:
    """Generate a precision resume using JD, resume, profile, and extracted keywords."""
    jd_entities = extract_skill_entities(jd)
    resume_entities = extract_skill_entities(resume)
    missing_skills = sorted(jd_entities["skills"] - resume_entities["skills"])

    prompt = """
    Create resume optimized for this job:
    Job Description: {jd}
    Current Resume: {resume}
    Additional Profile: {profile}
    Keywords to emphasize: {keywords}
    Missing skills (do not add if absent): {missing_skills}
    
    Requirements:
    1. Include all contact information
    2. Tailor professional summary
    3. Reorder skills by relevance
    4. Rewrite experience bullets to match JD
    5. Keep to 1 page
    """
    inputs = {
        "jd": clamp_text(jd, MAX_JD_CHARS),
        "resume": clamp_text(resume, MAX_RESUME_CHARS),
        "profile": clamp_text(profile, MAX_PROFILE_CHARS),
        "keywords": ", ".join(keywords),
        "missing_skills": ", ".join(missing_skills) if missing_skills else "None"
    }
    # Instruct models explicitly not to invent contact PII; enforce via sanitizer.
    prompt = prompt + "\n\nDo not invent contact information. Only include contact fields that appear in the provided resume. If none are present, omit contact lines. Keep the resume concise."
    if low_memory:
        result = generate_content(prompt, inputs, RESUME_SYSTEM_PROMPT, llm_technical_light)
    else:
        # Try heavy model first, then fall back to light if memory is insufficient.
        result = generate_with_fallback(prompt, inputs, RESUME_SYSTEM_PROMPT, llm_technical_heavy, llm_technical_light)

    try:
        return sanitize_output_against_resume(result, inputs.get("resume", ""))
    except Exception:
        return result

def generate_cover_letter(jd: str, resume: str, low_memory: bool = False) -> str:
    """Generate a cover letter based on the job description and resume."""
    prompt = """
    Write cover letter for:
    Job Description: {jd}
    Resume: {resume}
    
    Requirements:
    1. Professional opening
    2. Highlight 3 key qualifications
    3. Show evidence from resume
    4. Strong closing
    """
    inputs = {
        "jd": clamp_text(jd, MAX_JD_CHARS),
        "resume": clamp_text(resume, MAX_RESUME_CHARS)
    }
    prompt = prompt + "\n\nDo not invent contact information or personal details. Use only information present in the provided resume. If necessary, state clearly when details are missing."
    if low_memory:
        result = generate_content(prompt, inputs, COVER_LETTER_SYSTEM_PROMPT, llm_general_light)
    else:
        # Try heavy model first, then fall back to light if memory is insufficient.
        result = generate_with_fallback(prompt, inputs, COVER_LETTER_SYSTEM_PROMPT, llm_general_heavy, llm_general_light)

    try:
        return sanitize_output_against_resume(result, inputs.get("resume", ""))
    except Exception:
        return result

def analyze_match(jd: str, resume: str, low_memory: bool = False) -> str:
    """Analyze the match between a resume and a job description."""
    prompt = """
    Analyze resume match with job:
    Job Description: {jd}
    Resume: {resume}
    
    Provide:
    1. Match percentage (0-100%)
    2. Missing keywords
    3. Resume strengths
    4. Suggested improvements
    """
    # For analysis, use light model regardless of mode.
    inputs = {"jd": clamp_text(jd, MAX_JD_CHARS), "resume": clamp_text(resume, MAX_RESUME_CHARS)}
    return generate_content(prompt, inputs, ANALYSIS_SYSTEM_PROMPT, llm_general_light)
