import transformers as tf
from datasets import load
import torch
from pynvml import *
import torch

torch.cuda.empty_cache() # Empty gpu cache
torch.cuda.set_per_process_memory_fraction(0.99)

def print_gpu_usage():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# ------------- Helper Functions -----------------

def encode_process_bert(input):
    return tokenizer_bert(input["text"], truncation=True)

# ------------- BERT Pytorch -----------------

tokenizer_bert = tf.AutoTokenizer.from_pretrained("bert-base-uncased", num_labels=2, padding=True, truncation=True, max_length=200)
model_bert = tf.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to('cuda')

# ------------- Datasets -----------------

data = load.load_dataset("imdb") # Load the dataset 

data_bert = data.map(encode_process_bert, batched=True) # Process for BERT model
data_collator_bert = tf.DataCollatorWithPadding(tokenizer=tokenizer_bert) # Do data collation with the BERT tokenizer
data_tokenized_bert = data_bert.map(encode_process_bert, batched=True) # Tokenize data with BERT tokenizer

trainig_arguments_bert = tf.TrainingArguments( # Init traning arguments
    output_dir="./bert_text_class",
    learning_rate=4e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1, # 6250 global steps
    weight_decay=0.05,
    gradient_checkpointing=True,
    fp16=True,
    gradient_accumulation_steps=4,
    optim="adafactor",
)

trainer_bert = tf.Trainer( # Setup the trainer for BERT
    model=model_bert,
    args=trainig_arguments_bert,
    train_dataset=data_tokenized_bert["train"],
    eval_dataset=data_tokenized_bert["test"],
    tokenizer=tokenizer_bert,
    data_collator=data_collator_bert
)

torch.cuda.empty_cache() # Empty gpu cache

trainer_bert.train()

print("Done with training the BERT model")

eval_bert = trainer_bert.evaluate()
print("--- BERT Eval ---")
print("\n", eval_bert)


