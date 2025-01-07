from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from transformers import pipeline,AutoTokenizer, AutoModelForTokenClassification
from typing import List, Union
import docx
import nltk
from googletrans import Translator

app = FastAPI()

translator = Translator()

# Load NER model using Hugging Face Transformers
# ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

#persian models:
models = {
    1:"ner_parsbert",
    2:"ner_distilbert",
    3:"ner_hafez"
}

model_parsbert = AutoModelForTokenClassification.from_pretrained("resources/ner_parsbert")
tokenizer_parsbert = AutoTokenizer.from_pretrained("resources/ner_parsbert")
ner_parsbert = pipeline("ner", model=model_parsbert, tokenizer=tokenizer_parsbert)
# ner_parsbert = pipeline("ner", model="HooshvareLab/bert-base-parsbert-ner-uncased")
# Save the model and tokenizer
# ner_parsbert.model.save_pretrained("resources/ner_parsbert")
# ner_parsbert.tokenizer.save_pretrained("resources/ner_parsbert")

model_distilbert = AutoModelForTokenClassification.from_pretrained("resources/ner_distilbert")
tokenizer_distilbert = AutoTokenizer.from_pretrained("resources/ner_distilbert")
ner_distilbert = pipeline("ner", model=model_distilbert, tokenizer=tokenizer_distilbert)
# ner_distilbert = pipeline("ner", model="HooshvareLab/distilbert-fa-zwnj-base-ner")
# ner_distilbert.model.save_pretrained("resources/ner_distilbert")
# ner_distilbert.tokenizer.save_pretrained("resources/ner_distilbert")

model_hafez = AutoModelForTokenClassification.from_pretrained("resources/ner_hafez")
tokenizer_hafez = AutoTokenizer.from_pretrained("resources/ner_hafez")
ner_hafez = pipeline("ner", model=model_hafez, tokenizer=tokenizer_hafez)
# ner_hafez = pipeline("ner", model="ViravirastSHZ/Hafez-NER")
# ner_hafez.model.save_pretrained("resources/ner_hafez")
# ner_hafez.tokenizer.save_pretrained("resources/ner_hafez")

# Create sentence tokenizer
nltk.download('punkt_tab')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def split_into_sentences(text: str) -> List[str]:
    final_sentences = []
    first_sentences = text.split('\n')
    # Split text into sentences
    for mtext in first_sentences:
      final_sentences.extend(sentence_tokenizer.tokenize(mtext.strip()))

    return final_sentences

class NERResponse(BaseModel):
    entity: str
    type: str
    start: int
    end: int
    word: str

class NERResult(BaseModel):
    entities: List[NERResponse]

@app.post("/process-text", response_model=NERResult)
async def process_text(
    text: str = Form(...), model:int = 0
):
    """
    Endpoint to process raw text and extract named entities.
    """
    entities = []
    sentences = split_into_sentences(text)
    for sentence in sentences:
        lang = translator.detect(text)
        if lang == 'en':

        elif lang == 'fa':
            result = ner_parsbert(sentence)
            entities.append({
                "entity": result["entity"],
                "start": result["start"],
                "end": result["end"],
                "word": result["word"],
            })
        else:
            return f"can't process {lang} language texts!"


    return {"entities": entities}

@app.post("/process-file", response_model=NERResult)
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

    # Extract NER from the text
    ner_results = ner_pipeline(text)

    # Format response
    entities = [
        {
            "entity": result["entity"],
            "type": result["entity"],
            "start": result["start"],
            "end": result["end"],
            "word": result["word"],
        }
        for result in ner_results
    ]
    return {"entities": entities}

@app.get("/")
async def root():
    return {"message": "NER API is running. Use '/process-text' or '/process-file' endpoints."}
