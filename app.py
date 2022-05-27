from transformers import pipeline
import gradio as gr

model = pipeline("summarize")

def predict(prompt):
    summary = model(prompt)[0]["summarize_text"]
    return summary

gr.Interface(fn=predict, inputs="text", outputs="text").launch()