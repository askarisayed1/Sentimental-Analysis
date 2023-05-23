# Import the necessary libraries
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Create an instance of the Flask application
app = Flask(__name__)

# Define the model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define the mapping of sentiment labels to human-readable strings
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

@app.route("/sentiment", methods=["POST"])
def predict_sentiment():
    """
    Endpoint for sentiment prediction.

    Accepts a POST request with a JSON body containing an input text.
    Returns a JSON response with the predicted sentiment and score.

    Example Input:
    {
        "text": "I am too good"
    }

    Example Output:
    {
        "text": "I am too good",
        "sentiment": "Positive",
        "score": 0.9603369832038879
    }
    """
    # Retrieve the input text from the request body
    input_text = request.json["text"]

    # Tokenize the input text and prepare it for model input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Pass the input through the model to get the sentiment prediction
    outputs = model(**inputs)
    predicted_label_idx = outputs.logits.argmax(dim=1).item()
    sentiment_score = outputs.logits.softmax(dim=1)[0][predicted_label_idx].item()
    predicted_label = SENTIMENT_LABELS[predicted_label_idx]

    # Prepare the response JSON
    response = {
        "text": input_text,
        "sentiment": predicted_label,
        "score": sentiment_score
    }

    # Return the response as JSON
    return jsonify(response)

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
