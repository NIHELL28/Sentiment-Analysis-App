import gradio as gr
from transformers import pipeline

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(text)[0]
    return f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})"

# Custom CSS
custom_css = """
#interface-container {background-color: #f0f8ff;}
#title {color: #0f4c75; font-size: 32px; font-weight: bold;}
.gr-textbox label {color: #3282b8;}
.gr-text-output label {color: #bbe1fa;}
button {background-color: #1b262c; color: #ffffff;}
"""

# Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter Text", placeholder="Type your sentence here..."),
    outputs=gr.Text(label="Sentiment Analysis Result"),
    title="🌐 Sentiment Analysis App",
    description="🔍 Enter a sentence and the model will predict if it's POSITIVE or NEGATIVE.",
    theme="compact",
    css=custom_css
)

# Launch the Gradio app
iface.launch()
