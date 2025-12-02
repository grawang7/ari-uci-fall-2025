import os

from google import genai
from google.genai import types
from PyPDF2 import PdfReader
import gradio as gr
from dotenv import load_dotenv

from vectara import score

import requests
load_dotenv()

from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
url = "https://api.vectara.io/v2/hallucination_correctors/correct_hallucinations"
payload = {}

def extract_pdf_text(path):
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

def create_summary(input):
    prompt = (
        f"""You are a professional healthcare chat bot. 
        You are asked the question 'Provide a summary of the patient's health profile given the description of the patient's symptoms. 
        There may be more than one patient. This is the description: {input}"""
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(temperature=0.3)
    )
    raw = "".join(part.text for part in resp.candidates[0].content.parts if part.text)
    clean_json = raw.strip().removeprefix("```json").removesuffix("```").strip()    
    return clean_json

def file_uploaded(path, user_input):
    if not path:
        return adapater_user(user_input)
    return adapater_file(path)

def adapater_file(pdf_input):
    txt = extract_pdf_text(pdf_input)
    summary = create_summary(txt)
    result = score([(txt, summary)])
    payload = {
    "generated_text": summary,
    "documents":[
        {
            "text": txt
        }
    ],
    "model_name": "vhc-large-1.0",
    } 
    while float(result[0]) < 0.8:
        response = requests.post(url, json=payload, headers={"X-Api-Key": "zut_IVJ0kti5gEtJ9ntB6itiAGR_DhHM9JGpt2x8-Q"})
        resp_json = response.json()
        result = score([(txt, resp_json["corrected_text"])])
        payload["generated_text"] = resp_json["corrected_text"]
    return str(result[0])

def adapater_user(user_input):
    summary = create_summary(user_input)
    result = score([(user_input, summary)])
    payload = {
    "generated_text": summary,
    "documents":[
        {
            "text":user_input
        }
    ],
    "model_name": "vhc-large-1.0",
    }
    while float(result[0]) < 0.8:
        response = requests.post(url, json=payload, headers={"X-Api-Key": "zut_IVJ0kti5gEtJ9ntB6itiAGR_DhHM9JGpt2x8-Q"})
        resp_json = response.json()
        result = score([(user_input, resp_json["corrected_text"])])
        payload["generated_text"] = resp_json["corrected_text"]
    return str(result[0])

with gr.Blocks() as demo:
    gr.Markdown("Hallucination Detector")
    with gr.Row():
        pdf_path = gr.File(label="Upload Ground Truth", file_count="single", type="filepath")
    user_input = gr.Textbox(label="Enter Text Here")
    detect_hall = gr.Button("Run Model")
    result = gr.Markdown(label="Scoring")

    
    detect_hall.click(fn=file_uploaded, inputs=[pdf_path, user_input], outputs=result)

demo.launch()