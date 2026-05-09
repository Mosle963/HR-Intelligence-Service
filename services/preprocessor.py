import spacy
import re

# In a production environment, you'd run 'python -m spacy download en_core_web_sm' 
# during your Docker build or server setup script.
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    # Fallback for local dev, but for the 'hosted' version, we'll ensure it's pre-installed
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

ALLOWED_TOKENS = {'c++', 'c#', '.net'}

def preprocess(text: str) -> str:
    """
    Cleans a single string for storage or prediction.
    Ensures 'The Brain' and the DB are always in sync.
    """
    if not text:
        return ""
    
    # 1. Standardize spacing and case
    text = text.lower().replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        token_text = token.text.lower()

        # Check special technical tokens
        if token_text in ALLOWED_TOKENS:
            tokens.append(token_text)
            continue

        # Filter out noise
        if token.is_stop or token.is_punct or token.is_space:
            continue
        
        # Handle alphanumeric (html5, v10)
        has_letter = any(char.isalpha() for char in token_text)
        has_digit = any(char.isdigit() for char in token_text)
        if has_letter and has_digit:
            tokens.append(token_text)
            continue

        # Standard words -> Lemmatize
        if token.is_alpha:
            tokens.append(token.lemma_)
            continue
            
    return " ".join(tokens)