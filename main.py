import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import (
DataCollatorWithPadding, ErnieModel, BertForSequenceClassification, AutoTokenizer, 
AutoModel, BertTokenizer, ErnieForSequenceClassification, TrainingArguments, Trainer)
from datasets import load, dataset_dict

# Documentation used: huggingface.co/docs/transformers

# TODO: Debug ernie run, not working. DataCollator object error
# TODO: Setup more models, do 3 different NLP tasks across both models

# ------------- ERNIE Pytorch -----------------

tokenizer_ernie = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-large-en", num_labels=2)
model_ernie = ErnieForSequenceClassification.from_pretrained("nghuyong/ernie-2.0-large-en", num_labels=2)


# ------------- BERT Pytorch -----------------

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased", num_labels=2)
model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ------------- Helper Functions -----------------

def encode_process_ernie(input):
    return tokenizer_ernie(input["text"], truncation=True)

def encode_process_bert(input):
    return tokenizer_bert(input["text"], truncation=True)

# ------------- Datasets -----------------

data = load.load_dataset("imdb") # Load the dataset 

data_bert = data.map(encode_process_bert, batched=True) # Process for BERT model
data_collator_bert = DataCollatorWithPadding(tokenizer=tokenizer_bert) # Do data collation with the BERT tokenizer
data_tokenized_bert = data_bert.map(encode_process_bert, batched=True) # Tokenize data with BERT tokenizer

data_ernie = data.map(encode_process_bert, batched=True) # Same as above, only for the ernie tokenizer
data_collator_ernie = DataCollatorWithPadding(tokenizer=tokenizer_ernie)
data_tokenized_ernie = data_ernie.map(encode_process_ernie, batched=True)

trainig_arguments_bert = TrainingArguments( # Init traning arguments
    output_dir="./traning_results_bert",
    learning_rate=4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.05,
)

trainig_arguments_ernie = TrainingArguments( # Init traning arguments
    output_dir="./traning_results_ernie",
    learning_rate=4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.05,
)

trainer_bert = Trainer( # Setup the trainer for BERT
    model=model_bert,
    args=trainig_arguments_bert,
    train_dataset=data_tokenized_bert["train"],
    eval_dataset=data_tokenized_bert["test"],
    tokenizer=tokenizer_bert,
    data_collator=data_collator_bert
)

trainer_ernie = Trainer( # Setup the trainer for ERNIE
    model=model_ernie,
    args=trainig_arguments_ernie,
    train_dataset=data_collator_ernie["train"],
    eval_dataset=data_collator_ernie["test"],
    tokenizer=tokenizer_ernie,
    data_collator=data_collator_ernie    
)

trainer_bert.train()

print("Done with training the BERT model")

trainer_ernie.train()

print("Done with training the ERNIE model")


