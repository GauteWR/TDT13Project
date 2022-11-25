import transformers as tf
from datasets import load
import torch

# ------------- Dataset -----------------

data = load.load_dataset("conll2003")

print("Dataset loaded")

# ------------- BERT Pytorch -----------------

tokenizer_bert = tf.AutoTokenizer.from_pretrained("bert-base-uncased", num_labels=14)
print("Tokenizer loaded")
model_bert = tf.BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=14)
print("Model loaded")

def tokenize_input(input):
    tokens = []
    tokenized_inputs = tokenizer_bert(input["tokens"], truncation=True, padding=True, is_split_into_words=True)
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


data_tokenized_bert = data.map(tokenize_input, batched=True) # Tokenize data with BERT tokenizer
print("Tokenized")
data_collator_bert = tf.DataCollatorForTokenClassification(tokenizer=tokenizer_bert) # Do data collation with the BERT tokenizer
print("Collecter done")

trainig_arguments_bert = tf.TrainingArguments( # Init traning arguments
    output_dir="./bert_token_class",
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