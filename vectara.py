from transformers import pipeline, AutoTokenizer
import torch

def score(pairs):
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    input_pairs = [prompt.format(text1=pair[0], text2=pair[1]) for pair in pairs]

    classifier = pipeline(
                "text-classification",
                model='vectara/hallucination_evaluation_model',
                tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
                trust_remote_code=True,
            )
    full_scores = classifier(input_pairs, top_k=None)
    simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']
    return simple_scores
