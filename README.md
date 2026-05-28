# Epstein Files RAG Explorer 🔍

An open-source Retrieval-Augmented Generation (RAG) platform to explore and analyze the unsealed Jeffrey Epstein court documents. Built with LangChain, ChromaDB, and Streamlit.

![Screenshot](pic.png)

## 🚀 Features
- **Open Stack**: Fully open-source tools and models.
- **Local & Fast**: Support for local execution via Ollama or high-speed cloud inference via Groq/OpenRouter.
- **Automated Ingestion**: Easily download and index curated parquet data from Hugging Face.
- **Strict Guardrails**: Designed to stay strictly within the context of the investigative documents.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Python 3.10+** (Recommend using a virtual environment).
- **Ollama** (Optional): If you want to run LLMs completely locally. Download at [ollama.com](https://ollama.com/).

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/AbhisumatK/Epstein_Files_RAG
cd Epstein_Files_RAG
python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the `.env.example` to `.env` and configure your providers:
```bash
cp .env.example .env
```
Fill in your API keys in `.env`:
- **Groq API**: Get yours at [console.groq.com](https://console.groq.com/).
- **OpenRouter API**: Get yours at [openrouter.ai](https://openrouter.ai/).
- **Ollama**: No key needed, just ensure it's running.

### 4. Data Ingestion
The Epstein dataset is massive (>200GB). By default, the ingestion script downloads only the first **0.5 GB** chunk for testing.
```bash
python ingest.py
```
- **Estimated Time**: ~3-5 minutes for the first chunk (depending on your bandwidth).
- **How to Tweaks**: Open `ingest.py` and change `num_files=1` to a higher number (e.g., `num_files=10` for ~5GB) to index more data.

### 5. Launch the Application
Start the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📊 Dataset Info
- **Source**: [Nikity/Epstein-Files](https://huggingface.co/datasets/Nikity/Epstein-Files) on Hugging Face.
- **Format**: Apache Parquet files containing extracted text from investigative files.
- **Note**: The 0.5 GB limit (one parquet file) is used to ensure quick setup and low memory usage. The full dataset contains hundreds of thousands of documents.

## 🛡️ Guardrails
This application includes specialized system prompts to ensure the assistant stays strictly within the investigative context. It will refuse out-of-scope requests (like general knowledge or unrelated tasks) to maintain the integrity of the analysis.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

