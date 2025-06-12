from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Azure OpenAI setup
endpoint = os.getenv("ENDPOINT_URL", "https://azure-chat-bot-demo.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1-mini")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"  # Or 2025-01-01-preview if needed

# Select auth method
if api_key:
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
else:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

# System message for tone
system_prompt = {
    "role": "system",
    "content": (
        "You're a witty, respectful chatbot. Use light sarcasm, helpful advice, "
        "and clever humor. Never rude. Think Tony Stark meets a mentor. If asked silly "
        "questions, answer with style. Keep replies 1–3 short paragraphs."
    )
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    messages = [system_prompt, {"role": "user", "content": user_message}]

    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=800,
            temperature=1,
        )
        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
