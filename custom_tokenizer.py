from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stops = [word.lower() for word in set(stopwords.words('english'))]

def tokenize_and_clean(text):
    """
    Applies tokenization at whitespace and punctuation, removes digits from words, removes stopwords
    :param text: string to be tokenized
    :return: list of tokens
    """
    tokens = []
    for token in word_tokenize(text):
        if not token.isdigit():  # keep purely numerical strings unchanged
            token = ''.join(c for c in token if c.isalpha())
        if token.isalnum(): #and token.lower() not in stops:  # if token != ""
            tokens.append(token.lower())
    return tokens

