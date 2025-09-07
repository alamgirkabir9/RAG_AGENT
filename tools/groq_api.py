import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_groq(prompt: str) -> str:
    """
    Sends the prompt to Groq Llama 3.3 70B and returns the output.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    #return response.choices[0].message.content
    return f"{response.choices[0].message.content}\n(from groq chat model)"


