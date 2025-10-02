import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import (
    SFTTrainer,
    SFTConfig
)

# 1. SETUP: LOAD AND PARSE THE DATA
# ------------------------------------
# We'll use Flan-T5, which is excellent for instruction-based tasks.
MODEL_CHECKPOINT = "google/gemma-3-270m-it"
TRAIN_DATASET_PATH = "/workspace/datas/few-nerd/supervised/train.preprocessed.csv"
VAL_DATASET_PATH = "/workspace/datas/few-nerd/supervised/dev.preprocessed.csv"
model_name = MODEL_CHECKPOINT.split("/")[-1]

# Create a Hugging Face Dataset
train_dataset = Dataset.from_csv(TRAIN_DATASET_PATH)
val_dataset = Dataset.from_csv(VAL_DATASET_PATH)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


# 2. PREPROCESS THE DATA
# ------------------------------------
# We frame the task with a prefix to guide the model.
PREFIX = "Extract named entities as a Json format. \nNow, given the sentence: "
SURFIX = "\n JSON result: "
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

def preprocess_function(sample):
    # Prepare inputs with the prefix
    return {
      "messages": [
          {"role": "user", "content": PREFIX + sample["Sentence"]},
          {"role": "assistant", "content": sample["NER"]}
      ]
    }

# Apply the preprocessing to our datasets
tokenized_train_dataset = train_dataset.map(preprocess_function)
tokenized_val_dataset = val_dataset.map(preprocess_function)


# 3. FINE-TUNE THE MODEL
# ------------------------------------
# Load the pre-trained T5 model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CHECKPOINT, 
    device_map="auto",
    dtype=torch.float32,
    attn_implementation='eager',
)

save_dir = f"/workspace/model_dir/{model_name}/ner-json-gen-mixed"



# Define training arguments
training_args = SFTConfig(
    output_dir=save_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    do_eval=True,
    packing=False,
    logging_steps=100, 
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=10,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20, # Increase epochs for small datasets
    bf16=True, # Use mixed precision if a GPU is available
    push_to_hub=False,
    report_to="tensorboard"
)

# Create the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    processing_class=tokenizer,
)

# Start training! 🚀
trainer.train()

# Save the final model
trainer.save_model(f"{save_dir}/final_model")