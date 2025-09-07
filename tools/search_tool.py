import requests
import os
from dotenv import load_dotenv

load_dotenv()

class WebSearchTool:
    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")

    def search(self, query: str = "latest news") -> str:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": 5,
            "language": "en"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()

            articles = results.get("articles", [])
            if not articles:
                return "⚠️ No news found."

            output = f"📰 News results for '{query}':\n\n"
            for article in articles:
                output += f"- {article['title']} ({article['url']})\n{article['description']}\n\n"
            #return output.strip()
            return f"{output.strip()} \n(from web)"


        except Exception as e:
            return f"❌ News search error: {str(e)}"
