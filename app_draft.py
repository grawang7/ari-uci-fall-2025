import gradio as gr
from transformers import pipeline, AutoTokenizer
from pdfminer.high_level import extract_text
import google.generativeai as genai
import kagglehub
from kagglehub import KaggleDatasetAdapter
import requests
import json
import config

def parse_dataset(row: int):
    # Set the path to the file you'd like to load
    file_path = "Disease_symptom_and_patient_profile_dataset.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "uom190346a/disease-symptoms-and-patient-profile-dataset",
    file_path,
    # Provide any additional arguments like 
    # sql_query or pandas_kwargs. See the 
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    print(f"first {row} records:", df.head(row))
    return df.head(row)

def convert_table_to_text(table_data) -> dict:
    # currently only 1 row of table data
    keys = table_data.keys()
    values = table_data.iloc[-1]
    return dict(zip(keys, values))

def generate_sentence_from_data(data: dict) -> str:
    if data["Fatigue"]:
        fatigue = "tired"
    else:
        fatigue = "not tired"

    if data["Fever"]:
        fever = "fever"
    else:
        fever = "no fever"

    if data["Difficulty Breathing"]:
        breathing = "have"
    else:
        breathing = "don't have"

    if data["Cough"]:
        cough = "have"
    else:
        cough = "don't have"

    query = f"""I am a {fatigue} {data["Age"]} year old {data["Gender"]} with {data["Cholesterol Level"]} cholesterol, {data["Blood Pressure"]} blood pressure, {data["Disease"]}, and with {fever}. I also {breathing} difficulty breathing and I {cough} a cough."""
    return query


def extract_summary(text, number) -> str:
    genai.configure(api_key=config.Grace_Google_API)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # response = model.generate_content(f'You are a chat bot answering questions using data. You must stick to the answers provided solely by the text in the passage provided. ' \
        # f'You are asked the question "Provide a concise summary of the following passage, covering the core pieces of information described. {text}"',)
        response = model.generate_content(f"""You are a professional healthcare chat bot. 
        You are asked the question "Provide a summary in paragraph form (no bullet points) of the patient's health profile given the description of the patient's symptoms. You are given {number} patients' information."
        This is the description: {text}"""
        )
        print(response.text)
        return response.text
    except Exception as e:
        print("API key call failed:", e)


def create_corpus_on_vectara() -> int:
    url = "https://api.vectara.io/v2/corpora"

    payload = json.dumps({
        "key": "temp-corpus",
        "name": "ai safety research incubator corpus",
        "description": "Documents with important information for my prompt.",
        "save_history": False,
        "queries_are_answers": False,
        "documents_are_questions": False,
        "encoder_name": "boomerang-2023-q3",
        "filter_attributes": [
            {
            "name": "Title",
            "level": "document",
            "description": "The title of the document.",
            "indexed": True,
            "type": "text"
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': config.Grace_Vectara_API
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.status_code

def get_corrections(text, summary, hallucination_score) -> json:
    url = "https://api.vectara.io/v2/hallucination_correctors/correct_hallucinations"

    payload = json.dumps({
        "generated_text": f'{summary}',
        "documents": [
            {
            "text": f'{text}'
            }
        ],
        "model_name": "vhc-large-1.0",
        "query": f"please identify what is causing hallucination score of {hallucination_score} (from vectara's hhem). suggest improvements to increase the hallucination score, i.e., formatting, missing information, incorrect information, etc."
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': config.Grace_Vectara_API2 # personal or query/index key?
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.status_code)
    return response.json()


def vectara(filepath, user_input, number: int = 1) -> float:
    if filepath:
        real = extract_text(filepath)
    else:
        real = user_input # TODO: intermediate step to format data before passing into model? can be automated??
    
    real = generate_sentence_from_data(convert_table_to_text(parse_dataset(1)))
    real = ""
    for i in range(1, number+1):
        real += generate_sentence_from_data(convert_table_to_text(parse_dataset(i))) + " "
    print(real)
        
    summary = extract_summary(real, number)
    pair = (real, summary)
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    input_pair = [prompt.format(text1=pair[0], text2=pair[1])]

    # Use text-classification pipeline to predict
    classifier = pipeline(
            "text-classification",
            model='vectara/hallucination_evaluation_model',
            tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
            trust_remote_code=True
            )
    full_scores = classifier(input_pair, top_k=None) # List[List[Dict[str, float]]]

    # Optional: Extract the scores for the 'consistent' label
    simple_score = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']
    score = simple_score[0] # 0.5

    get_corrections_response = get_corrections(real, summary, score)

    print(score)
    print(get_corrections_response)

    prev_score = score

    while score < 0.8:
        summary = extract_summary(get_corrections_response['corrected_text'], number)
        pair = (get_corrections_response['corrected_text'], summary)
        full_scores = classifier(input_pair, top_k=None)
        simple_score = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']
        prev_score = score
        score = simple_score[0]
        get_corrections_response = get_corrections(get_corrections_response['corrected_text'], summary, score)
        print(score)
        print(get_corrections_response)
        if (prev_score == score):
            break

    return [summary, score, get_corrections_response['corrected_text']]


if __name__ == '__main__':
    # corpus_creation_status = create_corpus_on_vectara()
    # print(corpus_creation_status)

    demo = gr.Interface(
        fn=vectara,
        inputs=[gr.Textbox(label="filepath"), gr.Textbox(label="user input text", lines=3), gr.Number(label="number of patients data included", value=1)],
        outputs=[gr.Textbox(label="summary", lines=7), gr.Number(label="hallucination score"), gr.Textbox(label="corrections", lines=7)],
        title="Vectara Hallucination Evaluation",
        description="Input the path to a PDF file to evaluate the hallucination score of its summary generated by Google Gemini-2.5-flash model using Vectara's hallucination evaluation model.",
    )
    
    demo.launch()
