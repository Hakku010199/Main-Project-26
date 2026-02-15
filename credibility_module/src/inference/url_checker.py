from src.inference.article_extractor import extract_article_text
from src.inference.predictor import predict_credibility

def check_news_url(url):
    text = extract_article_text(url)

    if not text or len(text) < 50:
        return {
            "url": url,
            "error": "No readable article content found."
        }

    result = predict_credibility(text)

    return {
        "url": url,
        "credibility": result["prediction"],
        "confidence": result["confidence"]
    }
