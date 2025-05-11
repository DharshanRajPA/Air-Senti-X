import random
from textblob import TextBlob

__all__ = ["get_emotion", "calculate_urgency"]

def get_emotion(text):
    keyword_emotions = {
        "Joy": ["happy", "joy", "delighted", "pleased", "cheerful", "smile"],
        "Anger": ["angry", "furious", "irate", "outraged", "annoyed"],
        "Frustration": ["frustrated", "disappointed", "upset", "sour"],
        "Hope": ["hope", "optimistic", "expectant"],
        "Calm": ["calm", "serene", "peaceful", "relaxed"]
    }
    
    text_lower = text.lower()
    for emotion, keywords in keyword_emotions.items():
        if any(keyword in text_lower for keyword in keywords):
            return emotion
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.4:
        return "Joy"
    elif polarity < -0.4:
        return "Anger"
    else:
        return "Calm"

def calculate_urgency(text):
    exclamations = text.count("!")
    upper_words = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
    
    subjectivity = TextBlob(text).sentiment.subjectivity

    raw_score = (0.1 * exclamations) + (0.05 * upper_words) + (0.3 * subjectivity)
    urgency = min(1.0, raw_score)
    
    return round(urgency, 2)

if __name__ == "__main__":
    test_text = "I am REALLY upset! This is unacceptable!!!"
    print("Test Text:", test_text)
    print("Emotion:", get_emotion(test_text))
    print("Urgency:", calculate_urgency(test_text))
