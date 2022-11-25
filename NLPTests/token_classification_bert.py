import transformers as tf
from datasets import load
import torch

# ------------- Dataset -----------------

data = load.load_dataset("wnut_17")

# ------------- BERT Pytorch -----------------

tokenizer_bert = tf.AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = tf.BertForSequenceClassification.from_pretrained("bert-base-uncased")

def tokenize_input(input):
    tokenized = tokenizer_bert(input['tokens'], truncation=True, is_split_into_words=True)
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

data_collator_bert = tf.DataCollatorWithPadding(tokenizer=tokenizer_bert) # Do data collation with the BERT tokenizer
data_tokenized_bert = data.map(tokenize_input, batched=True) # Tokenize data with BERT tokenizer

trainig_arguments_bert = tf.TrainingArguments( # Init traning arguments
    output_dir="./bert_token_class",
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