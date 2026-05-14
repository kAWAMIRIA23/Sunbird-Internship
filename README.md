# Sunbird AI Internship Assessment

This repository contains my submission for the Sunbird AI Internship Assessment. The project is divided into two major sections:

1. Python programming exercises solved using pytest.
2. A Generative AI web application powered by the Sunbird AI APIs.

---

# Project Overview

The project demonstrates:

* Python problem-solving skills
* Unit testing with pytest
* API integration
* Generative AI pipeline orchestration
* Speech-to-Text (STT)
* Summarization
* Translation into Ugandan local languages
* Text-to-Speech (TTS)
* Gradio frontend deployment

---

# Repository Structure

```text id="6lzwq4"
.
├── exercises/
│   └── basics.py
│
├── tests/
│
├── GEN_AI/
│   ├── app.py
│   ├── backend/
│   │   ├── pipeline.py
│   │   ├── sunbird_client.py
│   │   └── utils.py
│   │
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

# Part 1 — Python Programming Exercises

## Description

This section contains two Python exercises designed to test understanding of Python fundamentals and problem-solving.

The tasks are implemented in:

```text id="3fq8kv"
exercises/basics.py
```

The automated tests are located in:

```text id="jlwmf9"
tests/
```

---

## Functions Implemented

### 1. `collatz`

Implements the Collatz sequence logic.

### 2. `distinct_numbers`

Returns distinct/unique numbers from a collection.

---

## Running the Tests

### Create a virtual environment

```bash id="8y7k9n"
python -m venv venv
```

---

### Activate the environment

#### Linux / Mac

```bash id="8yfx4v"
source venv/bin/activate
```

#### Windows

```bash id="c4kdfy"
venv\Scripts\activate.bat
```

---

### Install dependencies

```bash id="v7fwyf"
pip install -r requirements.txt
```

---

### Run pytest

```bash id="5pdu1l"
pytest
```

---

# Part 2 — Generative AI Application

## Description

This application is a multilingual Generative AI pipeline powered entirely by the Sunbird AI APIs.

The system accepts either:

* text input
* audio input

and processes it through a complete AI pipeline.

---

# AI Pipeline Architecture

```text id="j51sji"
Input (Text or Audio)
        ↓
Speech-to-Text (optional)
        ↓
Summarization
        ↓
Translation
        ↓
Text-to-Speech
        ↓
Generated Output
```

---

# Features

## Text Input

Users can directly enter text for processing.

---

## Audio Upload

Users can upload audio files which are transcribed using Speech-to-Text.

---

## Summarization

The application summarizes the provided text using the Sunflower LLM.

---

## Translation

Summaries can be translated into Ugandan local languages including:

* Luganda
* Runyankole
* Ateso
* Lugbara
* Acholi

---

## Text-to-Speech

Translated text is converted into playable speech audio.

---

## Intermediate Outputs

The application displays:

* transcript
* summary
* translated summary
* generated audio

---

# Technologies Used

| Component             | Technology      |
| --------------------- | --------------- |
| Frontend              | Gradio          |
| Backend               | Python          |
| APIs                  | Sunbird AI APIs |
| HTTP Requests         | requests        |
| Environment Variables | python-dotenv   |
| Testing               | pytest          |

---

# Sunbird AI APIs Used

* Speech-to-Text (STT)
* Sunflower LLM
* Translation
* Text-to-Speech (TTS)

API Documentation:

* https://docs.sunbird.ai/guides/speech-to-text
* https://docs.sunbird.ai/guides/text-to-speech
* https://docs.sunbird.ai/guides/sunflower-chat
* https://docs.sunbird.ai/api-reference/introduction

---

# Environment Variables

Create a `.env` file inside the `GEN_AI` directory.

Example:

```env id="j9vw88"
SUNBIRD_API_TOKEN=your_api_token_here
```

---

# `.env.example`

```env id="7wpvtj"
SUNBIRD_API_TOKEN=your_token_here
```

---

# Running the GenAI Application

## Navigate to the project

```bash id="jlwm2u"
cd GEN_AI
```

---

## Install dependencies

```bash id="ls24u5"
pip install -r requirements.txt
```

---

## Start the application

```bash id="3d86bi"
python app.py
```

or

```bash id="pm4dcw"
gradio app.py
```

depending on configuration.

---

# Deployment

The application is deployed publicly using Hugging Face Spaces.

Deployed Link:

```text id="9m8fr7"
https://huggingface.co/spaces/mariakawa/sunbird
```

---

# Known Limitations

* Audio files longer than 5 minutes are rejected.
* Processing speed depends on network conditions and API response times.
* Large audio files may increase processing latency.
* TTS generation may take longer for lengthy translations.

---

# Security Notes

* API tokens are stored securely using environment variables.
* `.env` files are excluded from Git tracking.
* No secrets are committed to the repository.

---

# Conclusion

This project demonstrates practical software engineering skills including:

* Python programming
* Automated testing
* API integration
* AI pipeline orchestration
* Frontend development
* Error handling
* Deployment and documentation

The application combines speech, language, and translation technologies into a unified multilingual AI workflow designed around Ugandan local language accessibility.
