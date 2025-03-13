from transformers import pipeline
nlp = pipeline("sentiment-analysis")
print(nlp("Transformers is working!"))