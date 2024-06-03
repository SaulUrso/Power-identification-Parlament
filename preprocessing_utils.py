import spacy


def preprocess_document_for_count(doc):
    nlp = spacy.load("en_core_web_sm")
    doc = doc.strip()
    doc_spacy = nlp(doc)
    tokens = [token.lemma_ for token in doc_spacy if not token.is_stop and not token.is_punct]
    return tokens
