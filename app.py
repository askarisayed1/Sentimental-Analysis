from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define the mapping of sentiment labels to human-readable strings
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

@app.route("/sentiment", methods=["POST"])
def predict_sentiment():
    input_text = request.json["text"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_label_idx = outputs.logits.argmax(dim=1).item()
    sentiment_score = outputs.logits.softmax(dim=1)[0][predicted_label_idx].item()

    predicted_label = SENTIMENT_LABELS[predicted_label_idx]

    response = {
        "text": input_text,
        "sentiment": predicted_label,
        "score": sentiment_score
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
