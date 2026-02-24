from src.inference.article_extractor import extract_article_text
from src.inference.predictor import predict
from src.techniques.technique_detector import detect_technique

def check_url_for_propaganda(url):
    text = extract_article_text(url)

    if text.startswith("Error"):
        return text

    result = predict(text)

    if result["label"] == "Propaganda":
        category = detect_technique(text)
    else:
        category = "None"

    result["technique"] = category
    return result
