import io
import json
import mmap
import numpy
import torchaudio
import torch
#import spaces #comment out if you using Zero-GPU
import datetime
from collections import defaultdict
from pathlib import Path
from seamless_communication.inference import Translator
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import tempfile
import os
print(os.system("python -m pip install --upgrade pip"))
# print(os.system("pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118"))
from huggingface_hub import InferenceClient
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# Initialize a Translator object with a multitask model, vocoder on the GPU.
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"


AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600  # limit to 1 hour YouTube files

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card="vocoder_v2",
    device=device,
    dtype=dtype,
    apply_mintox=True,
)
device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)
client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)


def translate_seamless(input_text):
    # tgt_langs = ("spa", "fra", "deu", "ita", "hin", "cmn")
    text_output, speech_output = translator.predict(
        input=input_text,
        task_str="t2st",
        tgt_lang="eng",
        src_lang="eng",
  )
    return text_output,speech_output
device = 0 if torch.cuda.is_available() else "cpu"



# Initialize a Translator object with a multitask model, vocoder on the GPU.
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"


AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600  # limit to 1 hour YouTube files


def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt

    
# @spaces.GPU()  #comment out if you using Zero-GPU
def generate(
    prompt, history, system_prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(f"{system_prompt}, {prompt}", history)
    # print("formatted_prompt ",formatted_prompt)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    response_list=[]
    dummy_list=""
    for response in stream:
        output += response.token.text
        if len(dummy_list.split(" ")) < 50:
            dummy_list+=response.token.text
        else:
            response_list.append(dummy_list)
            dummy_list=""
        # yield [output]
    response_list.append(dummy_list)
    voice_stream=[]
    from tqdm import tqdm 
    for stream_response in tqdm(response_list):
        text_output, speech_output= translate_seamless(stream_response)
        voice_stream.append(speech_output.audio_wavs[0][0].to(torch.float32).cpu())
    # text_output, speech_output= translate_seamless(output)
    out_file = f"./voice.wav"
    # print(type(speech_output.audio_wavs[0][0]))
    torchaudio.save(out_file, torch.cat(voice_stream,1), speech_output.sample_rate)
    print("audio Saved")
    # torchaudio.save(out_file, speech_output.audio_wavs[0][0].to(torch.float32).cpu(), speech_output.sample_rate)
    # return history,sentence,out_file
    history[-1][1]=output
    return history,out_file


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def generate_speech(history):
    # yield (history, sentence,"/home/rnd/Documents/Ameer/Dream/voice.wav")
    return generate(history[-1][0],history, system_prompt="Machine learning Based Answer in 50 words")

def transcribe(wav_path):
    text = pipe(wav_path, batch_size=BATCH_SIZE, generate_kwargs={"task": "translate"}, return_timestamps=True)["text"]
    return text

def add_file(history, file):
    history = [] if history is None else history
    try:
        text = transcribe(file)
        print("Transcribed text:", text)
    except Exception as e:
        print(str(e))
        gr.Warning("There was an issue with transcription, please try writing for now")
        # Apply a null text on error
        text = "Transcription seems failed, please tell me a joke about chickens"

    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


EXAMPLES = [
    [[],"What is the difference between supervised and unsupervised machine learning?"],
    [[],"What is the difference between a generative and discriminative model?"],
    [[],"What is the difference between a probability and a likelihood?"],
    
]




MODELS = ["Mistral 7B Instruct"]

OTHER_HTML=f"""<div>
</div>
"""
DESCRIPTION = """# ML QA Voice ChatBot"""
with gr.Blocks(title="ML QA") as demo:
    gr.Markdown(DESCRIPTION)
    gr.Markdown(OTHER_HTML)
    # with gr.Row():
    #     model_selected = gr.Dropdown(
    #         label="Select Instuct LLM Model to Use",
    #         info="Mistral, Zephyr: Mistral uses inference endpoint, Zephyr is 5 bit GGUF",
    #         choices=MODELS,
    #         max_choices=1,

    #         value=MODELS[0],
    #     )
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
    )
    with gr.Row():
        txt = gr.Textbox(
            scale=3,
            show_label=False,
            placeholder="Enter text and press enter, or speak to your microphone",
            container=False,
            interactive=True,
        )
        txt_btn = gr.Button(value="Submit text", scale=1)
        btn = gr.Audio(source="microphone", type="filepath", scale=4)

    def stop():
        print("Audio STOP")
        set_audio_playing(False)
        
    with gr.Row():
        sentence = gr.Textbox(visible=False)
        audio = gr.Audio(
            value=None,
            label="Generated audio response",
            streaming=True,
            autoplay=True,
            interactive=False,
            show_label=True,
        )
        
        audio.end(stop)
        
    with gr.Row():
        gr.Examples(
        EXAMPLES,
        [chatbot, txt],
        [chatbot, txt],
        add_text,
        cache_examples=False,
        run_on_click=False, # Will not work , user should submit it 
    )   

    def clear_inputs(chatbot):
        return None
    clear_btn = gr.ClearButton([chatbot, audio])
    
    # chatbot_role.change(fn=clear_inputs, inputs=[chatbot], outputs=[chatbot])
    # model_selected.change(fn=clear_inputs, inputs=[chatbot], outputs=[chatbot])
    
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        generate_speech,  [chatbot], [chatbot,audio]
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        generate_speech,  [chatbot], [chatbot,audio]
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    file_msg = btn.stop_recording(
        add_file, [chatbot, btn], [chatbot, txt], queue=False
    ).then(
        generate_speech,  [chatbot], [chatbot,audio]
    )

    file_msg.then(lambda: (gr.update(interactive=True),gr.update(interactive=True,value=None)), None, [txt, btn], queue=False)

    gr.Markdown(
        """
This Space demonstrates how to speak to a chatbot, based solely on open accessible models.
It relies on following models :
Speech to Text : [Whisper-large-v3](https://huggingface.co/spaces/openai/whisper) as an ASR model, to transcribe recorded audio to text. It is called through a [gradio client](https://www.gradio.app/docs/client).
LLM Mistral    : [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) as the chat model. 
Text to Speech : [Facebook Seamless T2ST](https://github.com/facebookresearch/seamless_communication) as a Multilingual TTS model, to generate the chatbot answers. This time, the model is hosted locally.
Note:
- By using this demo you agree to the terms of the Open Source Not for Comercial Use Just Demo Purpuse
- Responses generated by chat model should not be assumed correct or taken serious, as this is a demonstration example only"""
    )
demo.queue()
demo.launch(debug=True,share=True)
