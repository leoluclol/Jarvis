import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Carica le variabili d'ambiente dal file .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

for m in client.models.list():
    print(m.id)