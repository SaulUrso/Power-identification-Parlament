import spacy
nlp = spacy.load("en_core_web_sm")


def preprocess_document_for_count(testo):
    doc = nlp(testo)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens
