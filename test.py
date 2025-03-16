from transformers import pipeline

from googletrans import Translator

# 1. Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis")
print("Sentiment Analysis:")
print(sentiment_pipeline("I feel great today!"))  # Analyze sentiment of a sentence

# 2. Summarization
summarizer = pipeline("summarization")
print("\nSummarization:")
text = """Transformers are a type of neural network architecture that has proven to be very effective in natural language processing tasks. 
They have been used in state-of-the-art models such as BERT and GPT."""
print(summarizer(text, max_length=30, min_length=10, do_sample=False))  # Summarize a paragraph

# 3. Question Answering
qa_pipeline = pipeline("question-answering")
print("\nQuestion Answering:")
qa_input = {
    'question': "What are transformers used for?",
    'context': text
}
print(qa_pipeline(qa_input))  # Answer a question based on context

# 4. Translation (English to Korean)
translator = Translator()
result = translator.translate("I am learning machine translation.", src='en', dest='ko')
print(result.text)
#translator = pipeline("translation_en_to_fr")
#print("\nTranslation (EN → FR):")
#print(translator("Machine learning is amazing!", max_length=40))  # Translate sentence to French

# 5. Named Entity Recognition (NER)
ner_pipeline = pipeline("ner", grouped_entities=True)
print("\nNamed Entity Recognition:")
print(ner_pipeline("Barack Obama was the 44th president of the United States."))  # Identify entities

# 6. Text Generation
generator = pipeline("text-generation", model="gpt2")
print("\nText Generation:")
print(generator("Once upon a time,", max_length=30, num_return_sequences=1))  # Generate continuation
