import torch
import numpy as np
from custom_dataset import FewNerdBinaryDataset
from transformers import (
    AutoTokenizer,
    T5ForTokenClassification,
    T5GemmaForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 1. SETUP: CHOOSE MODEL AND DATASET
# ------------------------------------
# We use flan-t5-small for a runnable example. For better performance,
# consider 'google/flan-t5-base' or 'google/flan-t5-large'.


# 2. LOAD DATASET AND TOKENIZER
# ------------------------------------
label_names = [0, 1, -100]

# Load the tokenizer for Flan-T5
# We must use use_fast=True to get the word_ids() mapping.


# 4. SETUP THE TRAINER
# ------------------------------------
# Data collator handles dynamic padding for batches

# Function to compute metrics during evaluation



def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model_name = args.model_checkpoint.split("/")[-1]
    
    train_datasets = FewNerdBinaryDataset(args.train_file, tokenizer)
    val_datasets = FewNerdBinaryDataset(args.validation_file, tokenizer)

    if "flan-t5" in model_name:
        new_words = ['{', '}']
        tokenizer.add_tokens(new_words)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Create id2label and label2id mappings for the model
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    # Load the model for token classification
    # THIS IS THE KEY STEP: T5ForTokenClassification uses the T5 encoder ONLY.
    # The decoder is not used.
    
    model = None
    if "flan-t5" in model_name:
        model = T5ForTokenClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            is_encoder_decoder=False
        )
        model.resize_token_embeddings(len(tokenizer))
    elif "t5gemma" in model_name:
        model = T5GemmaForTokenClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            device_map="auto",
            is_encoder_decoder=False,
            dropout_rate=args.dropout_rate,
            use_cache=False,
        )

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
        
        torch.cuda.empty_cache()
        return results

    
    bf16_precision = torch.cuda.is_available() and model.dtype == torch.float32
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        do_eval=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 2},
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=bf16_precision, # Use mixed precision if a GPU is available
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        save_total_limit=3,
        report_to="tensorboard",
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )


    # 5. TRAIN THE MODEL
    # ------------------------------------
    print("Starting training on the encoder...")

    trainer.train()

    # Save the final model
    trainer.save_model(f"{args.output_dir}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary NER model")
    parser.add_argument("--train_file", type=str, default="/workspace/datas/few-nerd/supervised/train.binary.csv", help="Path to the training file")
    parser.add_argument("--validation_file", type=str, default="/workspace/datas/few-nerd/supervised/dev.binary.csv", help="Path to the validation file")
    parser.add_argument("--model_checkpoint", type=str, default="google/flan-t5-base", help="Model name or path")
    
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/model_dir/flan-t5-base/binary-ner-fp32-mixed", help="Path to the output directory"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Dropout rate for the model"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    
    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "google/t5gemma-b-b-ul2-it",
        "--train_file", "/workspace/datas/few-nerd/supervised/test.binary.csv",
        "--output_dir", "/workspace/model_dir/t5gemma-b-b-ul2-it/test"
    ])'''

    print(args)

    main(args)