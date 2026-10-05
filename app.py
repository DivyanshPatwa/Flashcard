from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN"),
    provider="auto"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_flashcards():

    data = request.json

    text = data["text"]
    count = int(data.get("count", 5))

    if count not in [5, 10, 15]:
        count = 5

    format_lines = "\n\n".join(
        f"Q{i}: Question\nA{i}: Answer"
        for i in range(1, count + 1)
    )

    prompt = f"""
Create exactly {count} simple flashcards from the following study material.

Study material:
{text}

Give the output only in this format:

{format_lines}
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1000
    )

    result = response.choices[0].message.content

    return jsonify({"flashcards": result})


if __name__ == "__main__":
    app.run(debug=True)