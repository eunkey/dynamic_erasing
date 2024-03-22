import nltk
nltk.download('punkt')  # Download the punkt tokenizer models if not already downloaded
from nltk.tokenize import sent_tokenize

def split_into_sentences(text):
    """
    Split the input text into a list of sentences using NLTK's sent_tokenize function.
    
    Args:
    text (str): Input text to split into sentences.
    
    Returns:
    list: List of sentences extracted from the input text.
    """
    sentences = sent_tokenize(text)
    return sentences
