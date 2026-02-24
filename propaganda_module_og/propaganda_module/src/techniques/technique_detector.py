def detect_technique(text):
    text_lower = text.lower()

    if any(word in text_lower for word in ["corrupt", "evil", "destroy"]):
        return "Loaded Language"

    if any(word in text_lower for word in ["fear", "threat", "danger"]):
        return "Emotional Appeal"

    if any(word in text_lower for word in ["always", "never", "everyone"]):
        return "Exaggeration"

    return "Manipulated Information"
