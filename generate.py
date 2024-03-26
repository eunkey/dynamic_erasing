import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from datasets import load_dataset

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
dataset = load_dataset("rajpurkar/squad")

sentences = split_into_sentences(input_text)
input_ids = tokenizer(sentences[0], return_tensors="pt").input_ids
attention_mask = tokenizer(sentences[0], return_tensors="pt").attention_mask

for sentence in sentences[1:]:
    sentence = " "+sentence
    if random.random() < 0.2:
        # Flatten the tensors
        input_ids = input_ids.view(-1)
        attention_mask = attention_mask.view(-1)
        
        input_ids2 = tokenizer(sentence, return_tensors="pt").input_ids.view(-1)
        input_ids2 = input_ids2[:random.randrange(0, input_ids2.size()[0])]

        input_ids = torch.cat((input_ids, input_ids2), dim=0)
        attention_mask = torch.cat((attention_mask, torch.zeros((input_ids2.size()[0]))), dim=0)

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
