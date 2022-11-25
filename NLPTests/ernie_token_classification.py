import transformers as tf
from datasets import load
import torch


# ------------- ERNIE Pytorch -----------------

data = load.load_dataset("wnut_17")

tokenizer_ernie = tf.AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")
model_ernie = tf.ErnieForSequenceClassification.from_pretrained("nghuyong/ernie-2.0-base-en")

def tokenize_input(input):
    tokenized = tokenizer_ernie(input['tokens'], truncation=True, is_split_into_words=True)
    token_labels = []
    for index, token in enumerate(input["ner_tags"]):
        prev_map_id = -1
        ids = []
        map_ids = tokenized.word_ids(batch_index=index)
        for id in map_ids:
            if id != prev_map_id:
                ids.append(token[id])
            else:
                ids.append(-100)
            prev_map_id = id
        token_labels.append(ids)
    tokenized["labels"] = token_labels
    return tokenized

data_collator_ernie = tf.DataCollatorWithPadding(tokenizer=tokenizer_ernie) 
data_tokenized_ernie = data.map(tokenize_input, batched=True) 

trainig_arguments_ernie = tf.TrainingArguments( # Init traning arguments
    output_dir="./ernie_token_class",
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

trainer_ernie = tf.Trainer( # Setup the trainer for BERT
    model=model_ernie,
    args=trainig_arguments_ernie,
    train_dataset=data_tokenized_ernie["train"],
    eval_dataset=data_tokenized_ernie["test"],
    tokenizer=tokenizer_ernie,
    data_collator=data_collator_ernie
)

torch.cuda.empty_cache() # Empty gpu cache

trainer_ernie.train()

print("Done with training the BERT model")

eval_ernie = trainer_ernie.evaluate()
print("--- Ernie Eval ---")
print("\n", eval_ernie)