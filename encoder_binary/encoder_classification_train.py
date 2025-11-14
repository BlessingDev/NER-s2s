import torch
import numpy as np
from custom_dataset import TokenClassificationDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import average_precision_score
import argparse
import json
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 1. SETUP: CHOOSE MODEL AND DATASET
# ------------------------------------
# We use flan-t5-small for a runnable example. For better performance,
# consider 'google/flan-t5-base' or 'google/flan-t5-large'.


# 2. LOAD DATASET AND TOKENIZER
# ------------------------------------

# Load the tokenizer for Flan-T5
# We must use use_fast=True to get the word_ids() mapping.


# 4. SETUP THE TRAINER
# ------------------------------------
# Data collator handles dynamic padding for batches

# Function to compute metrics during evaluation



def main(args):
    model_name = args.model_checkpoint.split("/")[-1]
    
    entity_types_dict = []
    with open(args.entity_types_file, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
        entity_types_dict = json_data
    
    label_names = []
    if args.dataset_name == "binary":
        label_names = [1]
    elif args.dataset_name == "switch":
        label_names = [0, 1]
    else:
        label_names = entity_types_dict[args.dataset_name].copy()
        label_names.insert(0, 0)  # For non-entity tokens

    
    # Create id2label and label2id mappings for the model
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    # Load the model for token classification
    # THIS IS THE KEY STEP: T5ForTokenClassification uses the T5 encoder ONLY.
    # The decoder is not used.
    
    torch.manual_seed(args.seed)
    
    model = None
    if "flan-t5" in model_name:
        if args.custom_model:
            from model_code.t5_tokenclassification import T5ForTokenClassification
        else:
            from transformers import T5ForTokenClassification
        
        # class_weight1 = [0.5, 1.0]
        # class_weight2 = [1.0, 2.0]
        # class_weight3 = [1.0, 1.2]
        
        model = T5ForTokenClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            classifier_dropout=args.dropout_rate,
            smoothing_value=args.label_smoothing,
            is_encoder_decoder=False,
            dynamic_class_weights=args.dynamic_class_weights
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

        if args.custom_model:
            model.classifier.weight.data.normal_(mean=0.0, std=0.2)
            model.classifier.bias.data.zero_()
            model.transform1.weight.data.normal_(mean=0.0, std=0.2)
            model.transform1.bias.data.zero_()
    elif "t5gemma" in model_name:
        if args.custom_model:
            from model_code.t5_tokenclassification import T5GemmaForTokenClassification
        else:
            from transformers import T5GemmaForTokenClassification
        model = T5GemmaForTokenClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            is_encoder_decoder=False,
            classifier_dropout_rate=args.dropout_rate,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    elif "xlm-roberta" in model_name:
        if args.custom_model:
            from model_code.roberta import XLMRobertaForTokenClassification
        else:
            from transformers import XLMRobertaForTokenClassification

        model = XLMRobertaForTokenClassification.from_pretrained(
            args.model_checkpoint,
            classifier_dropout=args.dropout_rate,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id
        )
        
        if args.custom_model:
            model.classifier.weight.data.normal_(mean=0.0, std=0.1)
            model.classifier.bias.data.zero_()
            model.transform1.weight.data.normal_(mean=0.0, std=0.1)
            model.transform1.bias.data.zero_()
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    elif "roberta" in model_name:
        if args.custom_model:
            from model_code.roberta import RobertaForTokenClassification
        else:
            from transformers import RobertaForTokenClassification
        
        model = RobertaForTokenClassification.from_pretrained(
            args.model_checkpoint,
            classifier_dropout=args.dropout_rate,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id
        )
        
        if args.custom_model:
            model.classifier.weight.data.normal_(mean=0.0, std=0.1)
            model.classifier.bias.data.zero_()
            model.transform1.weight.data.normal_(mean=0.0, std=0.1)
            model.transform1.bias.data.zero_()
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    
    tokenizer.model_max_length = 1024

    print("Data Loading...")
    train_datasets = TokenClassificationDataset(args.train_file, tokenizer, label_names)
    val_datasets = TokenClassificationDataset(args.validation_file, tokenizer, label_names)
    print("Data Loaded.")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    def compute_metrics(p):
        decision_threshold = 0.5
        predictions, labels = p
        # Get the most likely prediction (argmax)
        
        label_len = len(label_names)
        if label_len == 1:
            discrete_predictions = (predictions[:, :, -1] >= decision_threshold).astype(int)

            extra_label_names = [0, 1]
        else:
            discrete_predictions = np.argmax(predictions, axis=2)
            
            extra_label_names = label_names
        
        # Remove ignored indices (the -100 labels)
        true_discrete_predictions = [
            [extra_label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(discrete_predictions, labels)
        ]
        true_labels = [
            [extra_label_names[l] for l in label if l != -100]
            for label in labels
        ]
        
        
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        for i in range(len(discrete_predictions)):
            cur_pred = np.array(true_discrete_predictions[i])
            cur_label = np.array(true_labels[i])
            TP = np.sum((cur_pred == cur_label) & (cur_label != 0))
            TN = np.sum((cur_pred == cur_label) & (cur_label == 0))
            FP = np.sum((cur_pred != cur_label) & (cur_pred != 0))
            FN = np.sum((cur_pred != cur_label) & (cur_label != 0))

            total_tp += TP
            total_tn += TN
            total_fp += FP
            total_fn += FN

        
        total = total_tp + total_tn + total_fp + total_fn
        accuracy = (total_tp + total_tn) / total if total > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        tnr = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
        results = {
            "precision": precision,
            "recall": recall,
            "tnr": tnr,
            "accuracy": accuracy
        }
        
        if label_len == 2:
            predictions = predictions[:, :, 1]
            flat_labels = labels.flatten()
            flat_predictions = predictions.flatten()
            
            valid_indices = flat_labels != -100
            flat_labels = flat_labels[valid_indices]
            flat_predictions = flat_predictions[valid_indices]
            
            auprc = average_precision_score(flat_labels, flat_predictions)
            results["auprc"] = auprc
        
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
        lr_scheduler_kwargs={"num_cycles": args.num_cycles},
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=bf16_precision, # Use mixed precision if a GPU is available
        load_best_model_at_end=True,
        metric_for_best_model="eval_auprc",
        save_total_limit=3,
        logging_steps=args.logging_steps,
        seed=args.seed,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )


    # 5. TRAIN THE MODEL
    # ------------------------------------
    print("Starting training on the encoder...")

    trainer.train()

    # Save the final model
    trainer.save_model(f"{args.output_dir}/final_model")
    print(f"model saved to {args.output_dir}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary NER model")
    parser.add_argument("--train_file", type=str, default="/workspace/datas/few-nerd/supervised/train.binary.csv", help="Path to the training file")
    parser.add_argument("--validation_file", type=str, default="/workspace/datas/few-nerd/supervised/dev.binary.csv", help="Path to the validation file")
    parser.add_argument("--model_checkpoint", type=str, default="google/flan-t5-base", help="Model name or path")
    parser.add_argument(
        "--entity_types_file", 
        type=str, 
        default="/workspace/datas/entity_types.json", 
        help="Path to the entity types file"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/model_dir/flan-t5-base/binary-ner-fp32-mixed", help="Path to the output directory"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="binary",
        help="Name of the dataset"
    )
    
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Dropout rate for the model"
    )
    parser.add_argument(
        "--dynamic_class_weights",
        action="store_true",
        help="Whether to use dynamic class weights during training"
    )
    parser.add_argument(
        "--custom_model",
        action="store_true"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing value for CrossEntropyLoss"
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
        "--num_cycles",
        type=int,
        default=5,
        help="Number of cycles for cosine learning rate scheduler"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of evaluation steps with no improvement after which training will be stopped"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    '''args = parser.parse_args([
        "--model_checkpoint", "google/t5gemma-b-b-ul2",
        "--train_file", "/workspace/datas/wnut17/train.switch.csv",
        "--validation_file", "/workspace/datas/wnut17/dev.switch.csv",
        "--dropout_rate", "0.1",
        "--dataset_name", "switch",
        "--batch_size", "256",
        "--output_dir", "/workspace/model_dir/test",
        "--custom_model"
    ])'''

    print(args)

    main(args)