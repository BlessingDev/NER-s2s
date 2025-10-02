import torch
import numpy as np
from custom_dataset import FewNerdBinaryDataset
from transformers import (
    AutoTokenizer,
    T5ForTokenClassification, # This class uses the T5 encoder with a token classification head
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from accelerate import infer_auto_device_map

# 1. SETUP: CHOOSE MODEL AND DATASET
# ------------------------------------
# We use flan-t5-small for a runnable example. For better performance,
# consider 'google/flan-t5-base' or 'google/flan-t5-large'.
MODEL_CHECKPOINT = "/workspace/model_dir/flan-t5-large/binary-ner/checkpoint-1000" # 임시 경로
TEST_DATASET_PATH = "/workspace/datas/few-nerd/supervised/test.binary.flan_t5_xl.csv"


# 2. LOAD DATASET AND TOKENIZER
# ------------------------------------
# Load the CoNLL-2003 dataset for Named Entity Recognition (NER)
test_datasets = FewNerdBinaryDataset(TEST_DATASET_PATH)
label_names = [0, 1, -100]

# Load the tokenizer for Flan-T5
# We must use use_fast=True to get the word_ids() mapping.
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)


# 4. SETUP THE TRAINER
# ------------------------------------
# Data collator handles dynamic padding for batches
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Function to compute metrics during evaluation
def compute_metrics(p):
    predictions, labels = p
    # Get the most likely prediction (argmax)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored indices (the -100 labels)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recalls = []
    precisions = []
    accuracies = []
    for i in range(len(predictions)):
        cur_pred = np.array(true_predictions[i])
        cur_label = np.array(true_labels[i])
        TP = np.sum((cur_pred == cur_label) & (cur_label == 1))
        TN = np.sum((cur_pred == cur_label) & (cur_label == 0))
        FP = np.sum((cur_pred != cur_label) & (cur_pred == 1))
        FN = np.sum((cur_pred != cur_label) & (cur_label == 1))
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0
        accuracies.append(accuracy)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    results = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "accuracy": np.mean(accuracies)
    }
    results["f1"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"]) if (results["precision"] + results["recall"]) > 0 else 0
    
    return results

# Create id2label and label2id mappings for the model
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

# Load the model for token classification
# THIS IS THE KEY STEP: T5ForTokenClassification uses the T5 encoder ONLY.
# The decoder is not used.
model = T5ForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)

device_map = infer_auto_device_map(model, max_memory={0: "2GiB", 1: "24GiB", 2: "24GiB", 3: "24GiB"})

training_args = TrainingArguments(
    output_dir="/workspace/model_dir/flan-t5-large/binary-ner",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    do_eval=True,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False, # Set to True if you want to upload your model
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 6. EVALUATE THE MODEL
# ------------------------------------
print("Evaluating the fine-tuned model...")
eval_results = trainer.evaluate(eval_dataset=test_datasets)
print(f"Final Evaluation Results:\n{eval_results}")