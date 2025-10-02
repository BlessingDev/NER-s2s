import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForTokenClassification,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import argparse

# 1. SETUP: LOAD AND PARSE THE DATA
# ------------------------------------
# We'll use Flan-T5, which is excellent for instruction-based tasks.
MODEL_CHECKPOINT = "google/flan-t5-base"
TOKEN_CLASSIFIER_CHECKPOINT_DIR = "/workspace/model_dir/flan-t5-base/binary-ner-fp32-mixed-2/final_model"
TRAIN_DATASET_PATH = "/workspace/datas/few-nerd/supervised/train.preprocessed.csv"
VAL_DATASET_PATH = "/workspace/datas/few-nerd/supervised/dev.preprocessed.csv"
model_name = MODEL_CHECKPOINT.split("/")[-1]

# Create a Hugging Face Dataset
train_dataset = Dataset.from_csv(TRAIN_DATASET_PATH)
val_dataset = Dataset.from_csv(VAL_DATASET_PATH)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
if "flan-t5" in model_name:
    new_words = ['{', '}']
    tokenizer.add_tokens(new_words)

fine_tuned_classifier = T5ForTokenClassification.from_pretrained(TOKEN_CLASSIFIER_CHECKPOINT_DIR)


# 2. PREPROCESS THE DATA
# ------------------------------------
# We frame the task with a prefix to guide the model.
PREFIX = "Extract named entities as a Json format. \nNow, given the sentence: "
SURFIX = "\n JSON result: "
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

def preprocess_function(examples):
    # Prepare inputs with the prefix
    inputs = [PREFIX + doc + SURFIX for doc in examples["Sentence"]]
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, truncation=True, padding="max_length")

    # Tokenize the target NER JSON strings
    # The `text_target` is the NER string itself.
    labels = tokenizer(text_target=examples["NER"], max_length=tokenizer.model_max_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing to our datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)


# 3. FINE-TUNE THE MODEL
# ------------------------------------
# Load the pre-trained T5 model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
if "flan-t5" in model_name:
    model.resize_token_embeddings(len(tokenizer))

# Data collator will dynamically pad inputs and labels
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

save_dir = f"/workspace/model_dir/{model_name}/ner-json-gen-mixed-encoder"

# Extract the state dictionary from the fine-tuned classifier
fine_tuned_state_dict = fine_tuned_classifier.state_dict()

# Create a new dictionary to hold only the encoder weights
encoder_weights = {}
for key, value in fine_tuned_state_dict.items():
    if "encoder" in key:
        t5_key = key.replace("transformer.", "")
        encoder_weights[t5_key] = value

model.load_state_dict(encoder_weights, strict=False)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    do_eval=True,
    learning_rate=1e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20, # Increase epochs for small datasets
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Use mixed precision if a GPU is available
    push_to_hub=False,
)

# Create the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training! 🚀
trainer.train()

# Save the final model
trainer.save_model(f"{save_dir}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=MODEL_CHECKPOINT
    )
    