import os
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import nltk
import torch
import random
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, dataset_type, tokenizer):
        self.original_dataset = load_dataset(dataset_name)[dataset_type]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if dataset_name == "rajpurkar/squad":
            self.dataset = self._process_squad()
        
    def _process_squad(self):
        dataset = []
        
        for i in range(len(self.original_dataset)):
            input_ids = self.tokenizer("Question: "+self.original_dataset[i]["question"]+"\nContext: ", return_tensors="pt").input_ids
            attention_mask = self.tokenizer("Question: "+self.original_dataset[i]["question"]+"\nContext: ", return_tensors="pt").attention_mask
            
            input_sentences = sent_tokenize(self.original_dataset[i]["context"]+"\nAnswer")
            for sentence in input_sentences:
                sentence = " "+sentence
                if random.random() < 0.2:
                    # Flatten the tensors
                    input_ids = input_ids.view(-1)
                    attention_mask = attention_mask.view(-1)
                
                    input_ids2 = self.tokenizer(sentence, return_tensors="pt").input_ids.view(-1)
                    input_ids2 = input_ids2[:random.randrange(0, input_ids2.size()[0])]

                    input_ids = torch.cat((input_ids, input_ids2), dim=0)
                    attention_mask = torch.cat((attention_mask, torch.zeros((input_ids2.size()[0]))), dim=0)
                
                    # Reshape the concatenated tensor to the desired shape
                    input_ids = input_ids.view(1, -1)
                    attention_mask = attention_mask.view(1, -1)
                input_ids = input_ids.view(-1)
                attention_mask = attention_mask.view(-1)
                input_ids = torch.cat((input_ids, self.tokenizer(sentence, return_tensors="pt").input_ids.view(-1)), dim=0)
                attention_mask= torch.cat((attention_mask, self.tokenizer(sentence, return_tensors="pt").attention_mask.view(-1)), dim=0)
                input_ids = input_ids.view(1, -1)
                attention_mask = attention_mask.view(1, -1)
            
            input_ids = input_ids.view(-1)
            attention_mask = attention_mask.view(-1)
            input_ids = torch.cat((input_ids, self.tokenizer("Answer: "+self.original_dataset[i]["answers"]["text"][0], return_tensors="pt").input_ids.view(-1)), dim=0)
            attention_mask = torch.cat((attention_mask, self.tokenizer("Answer: "+self.original_dataset[i]["answers"]["text"][0], return_tensors="pt").attention_mask.view(-1)), dim=0)
            input_ids = input_ids.view(1, -1)
            attention_mask = attention_mask.view(1, -1)
            
            labels = input_ids.clone()
            
            
            dataset.append([input_ids, attention_mask, labels])
        
        return dataset
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids, attention_mask, labels= self.dataset[idx]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}