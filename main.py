from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from transformers import pipeline,AutoTokenizer, AutoModelForTokenClassification
from typing import List, Union
import docx
import nltk
from googletrans import Translator
import re
import spacy
import torch

app = FastAPI()

translator = Translator()

# Load medium SpaCy model
nlp = spacy.load("en_core_web_md")

# Regex patterns for structured entities
patterns = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "site": r"\b(?:https://|www\.)[^\s/$.?#].[^\s]*\b",
    "identification number": r"""
    (?<=\s)
    (?:\+\d{1,3}|\(\+\d{1,3}\))?   # Optional country code (+1, (+98))
    [-.\s]?                         # Optional separator
    (?:\(?\d{2,5}\)?)?              # Optional area code with parentheses (e.g., (415))
    [-.\s]?                         # Optional separator
    \d{1,4}                         # First group of digits
    [-.\s]?                         # Optional separator
    \d{4}
    (?=\s)
    """
}

# Load NER model using Hugging Face Transformers
# ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

#persian models:
models = {
    1:"ner_parsbert",
    2:"ner_distilbert",
    3:"ner_hafez"
}

# model_parsbert = AutoModelForTokenClassification.from_pretrained("resources/ner_parsbert")
# tokenizer_parsbert = AutoTokenizer.from_pretrained("resources/ner_parsbert")
# ner_parsbert = pipeline("ner", model=model_parsbert, tokenizer=tokenizer_parsbert)

# ner_xlm_roberta = pipeline("ner", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english")
# # Save the model and tokenizer
# ner_xlm_roberta.model.save_pretrained("resources/ner_xlm_roberta")
# ner_xlm_roberta.tokenizer.save_pretrained("resources/ner_xlm_roberta")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ",device)

model_xlm_roberta = AutoModelForTokenClassification.from_pretrained("resources/ner_xlm_roberta").to(device)
tokenizer_xlm_roberta = AutoTokenizer.from_pretrained("resources/ner_xlm_roberta")
ner_xlm_roberta = pipeline("ner", model=model_xlm_roberta, tokenizer=tokenizer_xlm_roberta, device=device)



model_parsbert = AutoModelForTokenClassification.from_pretrained("resources/ner_parsbert").to(device)
tokenizer_parsbert = AutoTokenizer.from_pretrained("resources/ner_parsbert")
ner_parsbert = pipeline("ner", model=model_parsbert, tokenizer=tokenizer_parsbert, device=device)
# ner_parsbert = pipeline("ner", model="HooshvareLab/bert-base-parsbert-ner-uncased")
# Save the model and tokenizer
# ner_parsbert.model.save_pretrained("resources/ner_parsbert")
# ner_parsbert.tokenizer.save_pretrained("resources/ner_parsbert")

# model_distilbert = AutoModelForTokenClassification.from_pretrained("resources/ner_distilbert").to(device)
# tokenizer_distilbert = AutoTokenizer.from_pretrained("resources/ner_distilbert")
# ner_distilbert = pipeline("ner", model=model_distilbert, tokenizer=tokenizer_distilbert, device=device)
# # ner_distilbert = pipeline("ner", model="HooshvareLab/distilbert-fa-zwnj-base-ner")
# # ner_distilbert.model.save_pretrained("resources/ner_distilbert")
# # ner_distilbert.tokenizer.save_pretrained("resources/ner_distilbert")
#
# model_hafez = AutoModelForTokenClassification.from_pretrained("resources/ner_hafez").to(device)
# tokenizer_hafez = AutoTokenizer.from_pretrained("resources/ner_hafez")
# ner_hafez = pipeline("ner", model=model_hafez, tokenizer=tokenizer_hafez, device=device)
# # ner_hafez = pipeline("ner", model="ViravirastSHZ/Hafez-NER")
# # ner_hafez.model.save_pretrained("resources/ner_hafez")
# # ner_hafez.tokenizer.save_pretrained("resources/ner_hafez")

# Create sentence tokenizer
nltk.download('punkt_tab')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# convert labels to united ones
united_labes = {
    'B-LOC':'B-location',
    'B-MISC':'B-misc',
    'B-ORG':'B-organization',
    'I-LOC':'I-location',
    'I-MISC':'I-misc',
    'I-ORG':'I-organization',
    'I-PER':'I-person',
}


def extract_entities_with_regex(text,sen):
    entities = []

    # for entity_type, pattern in patterns.items():
    #     for match in re.finditer(pattern, text):
    #         entities.append({
    #             "lang/sentence": sen,
    #             "entity": entity_type,
    #             "start": match.start(),
    #             "end": match.end(),
    #             "word": match.group(),
    #         })
    # Compile patterns with re.VERBOSE for multi-line support
    compiled_patterns = {
        key: re.compile(pattern, re.VERBOSE) if key == "identification number" else re.compile(pattern)
        for key, pattern in patterns.items()
    }
    for entity_type, pattern in compiled_patterns.items():
        for match in re.finditer(pattern, text):
            entities.append({
                "lang/sentence": sen,
                "entity": entity_type,
                "start": match.start(),
                "end": match.end(),
                "word": match.group(),
            })
    return entities

def split_into_sentences(text: str) -> List[str]:
    final_sentences = []
    first_sentences = text.split('\n')
    # Split text into sentences
    for mtext in first_sentences:
      final_sentences.extend(sentence_tokenizer.tokenize(mtext.strip()))

    return final_sentences


@app.post("/process-text")
async def process_text(
    text: str = Form(...)
):
    """
    Endpoint to process raw text and extract named entities.
    """
    all_entities = []
    sentences = split_into_sentences(text)
    for sentence in sentences:
        entities=[]
        print(sentence)
        # Step 1: structured entities using regex
        regex_entities = extract_entities_with_regex(sentence, "en/ " + sentence)
        entities.extend(regex_entities)

        detection = translator.detect(sentence)
        if detection.lang == 'en':
            # Step 2: Add SpaCy results for date/event
            doc = nlp(sentence)
            print("spacy:",doc)
            for new_entity in doc.ents:
                print("spacy: ",new_entity,new_entity.label_)
                if new_entity.label_ in {"DATE", "EVENT"}:
                    for entity in entities:
                        if new_entity.start_char < entity["end"] and new_entity.end_char > entity["start"]:
                            break
                    else:
                        entities.append({
                            "lang/sentence": "en/" + sentence,
                            "entity": new_entity.label_.lower(),
                            "start": new_entity.start_char,
                            "end": new_entity.end_char,
                            "word": new_entity.text,
                        })


            # step 3: Add language model result:
            result = ner_xlm_roberta(sentence)
            for new_entity in result:
                for entity in entities:
                    if new_entity["start"] < entity["end"] and new_entity["end"] > entity["start"]:
                        break

                else:
                    if entities:
                        print(entities[-1])
                        print(entities[-1]["end"])
                        print('011')
                        print(entities[-1]["end"] == new_entity["start"])
                        print(entities[-1]["entity"][2:] == new_entity["entity"][2:])
                    if entities and entities[-1]["end"] == new_entity["start"] and entities[-1]["entity"][2:] == new_entity["entity"][2:]:
                        entities[-1]["word"] += new_entity["word"]
                        entities[-1]["end"] = new_entity["end"]
                    else:
                        entities.append({
                            "lang/sentence":"en/ "+sentence,
                            "entity": new_entity["entity"],
                            "start": new_entity["start"],
                            "end": new_entity["end"],
                            "word": new_entity["word"],
                        })


        elif detection.lang == 'fa':
            result = ner_parsbert(sentence)
            for new_entity in result:
                for entity in entities:
                    if new_entity["start"] < entity["end"] and new_entity["end"] > entity["start"]:
                        break
                else:
                    if entities and entities[-1]["end"] == new_entity["start"] and entities[-1]["entity"][2:] == new_entity["entity"][2:]:
                        entities[-1]["word"] += new_entity["word"]
                        entities[-1]["end"] = new_entity["end"]
                    else:
                        entities.append({
                            "lang/sentence": "fa/" + sentence,
                            "entity": new_entity["entity"],
                            "start": new_entity["start"],
                            "end": new_entity["end"],
                            "word": new_entity["word"],
                        })
        else:
            return f"can't process {detection.lang} language texts!"

        all_entities.extend(entities)

    return {"entities": all_entities}

@app.post("/process-file")
async def process_file(file: UploadFile):
    """
    Endpoint to process uploaded text or Word file and extract named entities.
    """
    # Read content from the uploaded file
    if file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Process .docx files
        doc = docx.Document(file.file)
        text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    elif file.content_type == "text/plain":
        # Process plain text files
        text = (await file.read()).decode("utf-8")
    else:
        return {"error": "Unsupported file type. Please upload a .txt or .docx file."}

    all_entities = []
    sentences = split_into_sentences(text)
    for sentence in sentences:
        entities = []
        print(sentence)
        # Step 1: structured entities using regex
        regex_entities = extract_entities_with_regex(sentence, "en/ " + sentence)
        entities.extend(regex_entities)

        detection = translator.detect(sentence)
        if detection.lang == 'en':
            # Step 2: Add SpaCy results for date/event
            doc = nlp(sentence)
            print("spacy:", doc)
            for new_entity in doc.ents:
                print("spacy: ", new_entity, new_entity.label_)
                if new_entity.label_ in {"DATE", "EVENT"}:
                    for entity in entities:
                        if new_entity.start_char < entity["end"] and new_entity.end_char > entity["start"]:
                            break
                    else:
                        entities.append({
                            "lang/sentence": "en/" + sentence,
                            "entity": new_entity.label_.lower(),
                            "start": new_entity.start_char,
                            "end": new_entity.end_char,
                            "word": new_entity.text,
                        })

            # step 3: Add language model result:
            result = ner_xlm_roberta(sentence)
            for new_entity in result:
                for entity in entities:
                    if new_entity["start"] < entity["end"] and new_entity["end"] > entity["start"]:
                        break
                else:
                    if entities and entities[-1]["end"] == new_entity["start"] and entities[-1]["entity"][2:] == new_entity["entity"][2:]:
                        entities[-1]["word"] += new_entity["word"]
                        entities[-1]["end"] = new_entity["end"]
                    else:
                        entities.append({
                            "lang/sentence": "en/ " + sentence,
                            "entity": new_entity["entity"],
                            "start": new_entity["start"],
                            "end": new_entity["end"],
                            "word": new_entity["word"],
                        })


        elif detection.lang == 'fa':
            result = ner_parsbert(sentence)
            for new_entity in result:
                for entity in entities:
                    if new_entity["start"] < entity["end"] and new_entity["end"] > entity["start"]:
                        break
                else:
                    if entities and entities[-1]["end"] == new_entity["start"] and entities[-1]["entity"][2:] == new_entity["entity"][2:]:
                        entities[-1]["word"] += new_entity["word"]
                        entities[-1]["end"] = new_entity["end"]
                    else:
                        entities.append({
                            "lang/sentence": "fa/" + sentence,
                            "entity": new_entity["entity"],
                            "start": new_entity["start"],
                            "end": new_entity["end"],
                            "word": new_entity["word"],
                        })
        else:
            return f"can't process {detection.lang} language texts!"

        all_entities.extend(entities)
    return {"entities": all_entities}

@app.get("/")
async def root():
    return {"message": "NER API is running. Use '/process-text' or '/process-file' endpoints."}
