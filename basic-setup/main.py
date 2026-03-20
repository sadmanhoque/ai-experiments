from transformers import pipeline

model = pipeline("summarization", model="facebook/bart-large-cnn")
response = model("text to summarize")
print(response)
