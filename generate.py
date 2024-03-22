import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

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

# Load llama2-7b model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define input_ids and labels
input_text = """Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 'Venite Ad Me Omnes'. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary. To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? Saint Bernadette Soubirous"""
sentences = split_into_sentences(input_text)
input_ids = tokenizer(sentences[0], return_tensors="pt").input_ids
attention_mask = tokenizer(sentences[0], return_tensors="pt").attention_mask

for sentence in sentences[1:]:
    sentence = " "+sentence
    if random.random() < 0.5:
        # Flatten the tensors
        input_ids = input_ids.view(-1)
        attention_mask = attention_mask.view(-1)
        input_ids2 = tokenizer(sentence[:random.randrange(0,len(sentence))], return_tensors="pt").input_ids.view(-1)

        input_ids = torch.cat((input_ids, input_ids2), dim=0)
        attention_mask = torch.cat((attention_mask, torch.zeros((input_ids2.size()))), dim=0)

        # Reshape the concatenated tensor to the desired shape
        input_ids = input_ids.view(1, -1)
        attention_mask = attention_mask.view(1, -1)
    
    input_ids = input_ids.view(-1)
    attention_mask = attention_mask.view(-1)
    input_ids = torch.cat((input_ids, tokenizer(sentence, return_tensors="pt").input_ids.view(-1)), dim=0)
    attention_mask= torch.cat((attention_mask, tokenizer(sentence, return_tensors="pt").attention_mask.view(-1)), dim=0)
    input_ids = input_ids.view(1, -1)
    attention_mask = attention_mask.view(1, -1)

print(input_ids)
print(attention_mask)

# input_ids = tokenizer(input_text, return_tensors="pt").input_ids
# attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask

labels = input_ids.clone()

# Generate predictions

model = AutoModelForCausalLM.from_pretrained(model_name)

with torch.no_grad():

    outputs = model(input_ids=input_ids, attention_mask = attention_mask,labels=labels)
    logits = outputs.logits

# Calculate perplexity
loss = outputs.loss
perplexity = torch.exp(loss)
print("Perplexity:", perplexity.item())
