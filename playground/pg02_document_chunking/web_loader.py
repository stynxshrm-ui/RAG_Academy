"""Load web-pages"""
import requests
from bs4 import BeautifulSoup

def load_web(url: str) -> str:
    """Extracts text with double newlines between blocks for easy splitting."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noisy elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        return soup.get_text(separator='\n\n', strip=True)

    except Exception as e:
        return f"Error: {e}"


