import transformers as tf
import tensorflow as flow
from datasets import load
import numpy as np
from pynvml import *
import torch

torch.cuda.empty_cache() # Empty gpu cache
torch.cuda.set_per_process_memory_fraction(0.9999)

def encode_process_ernie(input):
    return tokenizer_ernie(input["text"], truncation=True)

# ------------- ERNIE Pytorch -----------------

def print_gpu_usage():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

tokenizer_ernie = tf.AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en", num_labels=2, max_length=3345, max_position_embeddings=3345, 
    max_split_size_mb=200)
model_ernie = tf.ErnieForSequenceClassification.from_pretrained("nghuyong/ernie-2.0-base-en", num_labels=2, max_position_embeddings=3345, ignore_mismatched_sizes=True).to('cuda')

data = load.load_dataset("imdb") # Load the dataset 

data_ernie = data.map(encode_process_ernie, batched=True) # Same as above, only for the ernie tokenizer
data_collator_ernie = tf.DataCollatorWithPadding(tokenizer=tokenizer_ernie)
data_tokenized_ernie = data_ernie.map(encode_process_ernie, batched=True)
# data_ids = tokenizer_ernie.convert_tokens_to_ids(data_tokenized_ernie)
# print(flow.constant([data_ids]))

trainig_arguments_ernie = tf.TrainingArguments( # Init traning arguments
    output_dir="./ernie_text_class",
    evaluation_strategy = "epoch",
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

torch.cuda.empty_cache() # Empty gpu cache

trainer_ernie = tf.Trainer( # Setup the trainer for ERNIE
    model=model_ernie,
    args=trainig_arguments_ernie,
    train_dataset=data_tokenized_ernie["train"],
    eval_dataset=data_tokenized_ernie["test"],
    tokenizer=tokenizer_ernie,
    data_collator=data_collator_ernie, 
)

torch.cuda.empty_cache() # Empty gpu cache

trainer_ernie.train()

print("Done with training the ERNIE model")

eval_ernie = trainer_ernie.evaluate()
print("--- ERNIE Eval ---")
print("\n", eval_ernie)