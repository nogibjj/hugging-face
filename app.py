from transformers import pipeline
import gradio as gr

model = pipeline("summarization")

def predict(prompt):
    summary = model(prompt)[0]["summarize_text"]
    return summary

gr.Interface(fn=predict, inputs="text to summarize", outputs="summary output").launch()