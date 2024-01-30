## 🦒 Colab

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jpxvCK1W05ykh1wHNjXMu7S2NZJmYnK0?authuser=1#scrollTo=m1huS-fj4qOz) | ML-QA Chatbot 🚀


# Chatbot Interaction Demo

## Overview
This Space demonstrates how to interact with a chatbot using open accessible models. It offers a seamless experience of converting speech to text, generating chatbot responses, and then converting these responses back to speech.

## Models Used

### Speech to Text
- **Model**: Whisper-large-v3
- **Source**: [Whisper-large-v3 on Hugging Face](https://huggingface.co/spaces/openai/whisper)
- **Description**: This ASR (Automatic Speech Recognition) model transcribes recorded audio to text. It is integrated through a [Gradio client](https://www.gradio.app/docs/client).

### Chat Model
- **Model**: Mixtral-8x7B-Instruct-v0.1
- **Source**: [Mixtral-8x7B-Instruct-v0.1 on Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- **Description**: This Large Language Model (LLM) Mistral serves as the chat model, generating responses based on the transcribed text.

### Text to Speech
- **Model**: Facebook Seamless T2ST
- **Source**: [Facebook Seamless T2ST on GitHub](https://github.com/facebookresearch/seamless_communication)
- **Description**: This Multilingual TTS (Text-to-Speech) model generates audio from the chatbot's text responses. The model is hosted locally.

## Preview
  ![ML-QA VoiceChatbot App](imgs/data.gif)
  
## Installation

Use the following commands to install the necessary SDK and dependencies:

Local installation:

```bash
git clone https://github.com/AMEERAZAM08/ML-QA.git
cd ML-QA
pip install -r requirements.txt
python app.py
```

