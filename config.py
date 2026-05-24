# config.py
# Configuration file for ATSModelOrchestrator models

AVAILABLE_MODELS = {
    "embedding": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    "general": {
        "light": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
        "heavy": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
    },
    "technical": {
        "software_engineering": {
            "light": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
            "heavy": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
        },
        "data_science": {
            "light": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
            "heavy": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
        },
        "data_engineering": {
            "light": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
            "heavy": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
        },
        "machine_learning": {
            "light": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
            "heavy": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
        },
        "ai": {
            "light": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
            "heavy": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
        }
    },
    "creative_polish": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
}

# Hybrid scoring weights (must sum to 1.0)
SCORING_WEIGHTS = {
    "semantic": 0.45,
    "skill_overlap": 0.25,
    "tfidf": 0.15,
    "experience": 0.10,
    "ontology": 0.05
}

# Sentence-transformer model for semantic similarity
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Skill lexicon for entity extraction and overlap scoring
SKILL_LEXICON = [
    "python",
    "fastapi",
    "langchain",
    "semantic kernel",
    "autogen",
    "react",
    "typescript",
    "node.js",
    "nodejs",
    "docker",
    "kubernetes",
    "azure",
    "azure ai search",
    "mongodb",
    "cosmos db",
    "pgvector",
    "chroma",
    "milvus",
    "sql server",
    "mysql",
    "postgresql",
    "git",
    "github",
    "jira",
    "datadog",
    "pytest",
    "playwright",
    "rag",
    "retrieval augmented generation",
    "llm",
    "llms",
    "embeddings",
    "prompt engineering",
    "fine tuning",
    "finetuning",
    "mlops",
    "llmops",
    "mcp",
    "model context protocol",
    "vector database",
    "vector databases",
    "azure databricks",
    "pyspark",
    "spark",
    "airflow",
    "dagster",
    "opensearch"
]
