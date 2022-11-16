import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import ErnieModel, AutoTokenizer, AutoModel, BertTokenizer, ErnieForMaskedLM
from pytorch_pretrained_bert import (
    BertAdam, BertConfig,
    BertForSequenceClassification, BertModel,
    BertForMaskedLM
)

# TODO: Set up basic BERT model 
# TODO: Set up the XLM model or the Ernie model
# TODO: Plot both results with similar runs and compare

# Documentation used: huggingface.co/docs/transformers

# ------------- ERNIE Pytorch -----------------

tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-2.0-large-en")
model = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-2.0-large-en")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
decoded = tokenizer.decode(predicted_token_id)
print("Decoded guess:", decoded)

print("Done with ERNIE")

# ------------- BERT Pytorch -----------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

logits = None
with torch.no_grad():
    logits = model(**inputs)

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
decoded = tokenizer.decode(predicted_token_id)
print("Decoded guess:", decoded)