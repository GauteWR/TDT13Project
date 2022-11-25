import transformers as tf
from datasets import load
import torch


# ------------- ERNIE Pytorch -----------------

data = load.load_dataset("conll2003")

tokenizer_ernie = tf.AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en", num_labels=14)
model_ernie = tf.ErnieForTokenClassification.from_pretrained("nghuyong/ernie-2.0-base-en", num_labels=14)

def tokenize_input(input):
    tokens = []
    tokenized_inputs = tokenizer_ernie(input["tokens"], truncation=True, padding=True, is_split_into_words=True)
    for index, label in enumerate(input[f"ner_tags"]):
        previous_id = None
        token_ids = []
        word_ids = tokenized_inputs.word_ids(batch_index=index)
        for id in word_ids:
            if id is None:
                token_ids.append(-100)
            elif id != previous_id:
                token_ids.append(label[id])
            else:
                token_ids.append(-100)
            previous_id = id
        tokens.append(token_ids)
    tokenized_inputs["labels"] = tokens
    return tokenized_inputs

data_collator_ernie = tf.DataCollatorForTokenClassification(tokenizer=tokenizer_ernie) 
data_tokenized_ernie = data.map(tokenize_input, batched=True) 

trainig_arguments_ernie = tf.TrainingArguments( # Init traning arguments
    output_dir="./ernie_token_class",
    learning_rate=4e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=10, 
    weight_decay=0.05,
    gradient_checkpointing=True,
    fp16=True,
    gradient_accumulation_steps=4,
    optim="adafactor",
)

trainer_ernie = tf.Trainer( # Setup the trainer for ERNIE
    model=model_ernie,
    args=trainig_arguments_ernie,
    train_dataset=data_tokenized_ernie["train"],
    eval_dataset=data_tokenized_ernie["test"],
    tokenizer=tokenizer_ernie,
    data_collator=data_collator_ernie
)

torch.cuda.empty_cache() # Empty gpu cache

trainer_ernie.train()

print("Done with training the Ernie model")

eval_ernie = trainer_ernie.evaluate()
print("--- Ernie Eval ---")
print("\n", eval_ernie)