import torch
import numpy as np
from scipy.special import expit, softmax
from torch.utils.data import DataLoader
from custom_dataset import TokenClassificationTestDataset
from transformers import (
    AutoTokenizer,
    T5GemmaForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    tokenizer.model_max_length = 1024


    # Load the model for token classification
    # THIS IS THE KEY STEP: T5ForTokenClassification uses the T5 encoder ONLY.
    # The decoder is not used.
    
    model = None
    if "flan-t5" in args.model_name:
        if args.custom_model:
            from model_code.t5_tokenclassification import T5ForTokenClassification
        else:
            from transformers import T5ForTokenClassification
        
        model = T5ForTokenClassification.from_pretrained(
            args.model_checkpoint,
            classifier_dropout=0.0,
            is_encoder_decoder=False,
        )
    elif "t5gemma" in args.model_name:
        model = T5GemmaForTokenClassification.from_pretrained(
            args.model_checkpoint,
            device_map="auto",
            is_encoder_decoder=False,
            use_cache=False,
        )
    elif "xlm-roberta" in args.model_name:
        if args.custom_model:
            from model_code.roberta import XLMRobertaForTokenClassification
        else:
            from transformers import XLMRobertaForTokenClassification

        model = XLMRobertaForTokenClassification.from_pretrained(
            args.model_checkpoint,
            classifier_dropout=0.0
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    elif "roberta" in args.model_name:
        if args.custom_model:
            from model_code.roberta import RobertaForTokenClassification
        else:
            from transformers import RobertaForTokenClassification
        
        model = RobertaForTokenClassification.from_pretrained(
            args.model_checkpoint,
            classifier_dropout=args.dropout_rate,
            num_labels=0.0
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    label_names = []
    for idx in range(len(model.config.id2label)):
        label_names.append(model.config.id2label[idx])
    
    test_dataset = TokenClassificationTestDataset(args.test_file, tokenizer, label_names)
    
    #print(test_dataset.get_label_ratio())

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    decision_threshold = args.decision_threshold
    def compute_metrics(p):
        predictions, labels = p
        # Get the most likely prediction (argmax)

        if len(label_names) == 1:
            predictions[:, :, 0] = expit(predictions[:, :, 0])
            predictions = (predictions[:, :, 0] >= decision_threshold).astype(int)
            
            
            extra_label_names = [0, 1]
        elif len(label_names) == 2:
            predictions = softmax(predictions, axis=-1)
            predictions = (predictions[:, :, 1] >= decision_threshold).astype(int)
            extra_label_names = label_names
        else:
            predictions = np.argmax(predictions, axis=2)

            extra_label_names = label_names

        # Remove ignored indices (the -100 labels)
        true_predictions = list()
        for prediction, label in zip(predictions, labels):
            prediction = prediction[:len(label)]
            true_predictions.append([extra_label_names[p] for p, l in zip(prediction, label) if l != -100])
        
        true_labels = [
            [extra_label_names[l] for l in label if l != -100]
            for label in labels
        ]
        
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        for i in range(len(predictions)):
            cur_pred = np.array(true_predictions[i])
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
            "true_negative_rate": tnr,
            "accuracy": accuracy
        }
        results["f1"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"]) if (results["precision"] + results["recall"]) > 0 else 0
        
        torch.cuda.empty_cache()
        return results

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.prediction_output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=False,
        do_predict=True,
        report_to="none", # Disable logging to wandb/tensorboard
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator, shuffle=False)
    print("Starting inference on the test set...")
    #predictions = trainer.predict(test_loader)
    with torch.no_grad():
        predictions = trainer.prediction_loop(test_loader, description="Prediction")
    
    max_len = predictions.predictions.shape[1] # 전체 시퀀스 중 가장 긴 길이
    label_array = np.ones((predictions.predictions.shape[0], max_len), dtype=np.int32) * -100
    for i, l in enumerate(test_dataset.labels):
        cur_len = len(l)
        label_array[i, :cur_len] = l
        
    metrics = compute_metrics((predictions.predictions, label_array))

    # 6. SAVE AND PRINT RESULTS
    print("\n--- Test Set Metrics ---")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"True Negative Rate: {metrics['true_negative_rate']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    # Create the output directory if it doesn't exist
    os.makedirs(args.prediction_output_dir, exist_ok=True)
    
    # Save metrics to a JSON file
    metrics_output_path = os.path.join(args.prediction_output_dir, "test_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nTest metrics saved to: {metrics_output_path}")

    # Save raw predictions (logits) and labels
    preds_output_path = os.path.join(args.prediction_output_dir, "test_predictions.npy")
    np.save(preds_output_path, predictions.predictions)
    print(f"Raw predictions (logits) saved to: {preds_output_path}")
    
    labels_output_path = os.path.join(args.prediction_output_dir, "test_labels.npy")
    np.save(labels_output_path, label_array)
    print(f"Test labels saved to: {labels_output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary NER model")
    parser.add_argument("--test_file", type=str, default="/workspace/datas/few-nerd/supervised/train.binary.csv", help="Path to the training file")
    parser.add_argument("--model_checkpoint", type=str, default="google/flan-t5-base", help="Model name or path")
    parser.add_argument(
        "--model_name",
        default="flan-t5",
        type=str,
        help="Model name: flan-t5 or t5gemma"
    )
    parser.add_argument(
        "--entity_types_file", 
        type=str, 
        default="/workspace/datas/entity_types.json", 
        help="Path to the entity types file"
    )
    
    parser.add_argument(
        "--prediction_output_dir", type=str, default="/workspace/datas/encoder/test", help="Path to the output directory"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification"
    )
    parser.add_argument(
        "--custom_model",
        action="store_true"
    )
    
    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "/workspace/model_dir/classification/flan-t5-base/conll2003/encoder-switch-ner-custom-drop20-cycle10-lr2e-4-cosine_restart/final_model",
        "--test_file", "/workspace/datas/mit_restaurant/test.switch.csv",
        "--custom_model",
        "--batch_size", "256",
        "--prediction_output_dir", "/workspace/datas/encoder/test"
    ])'''

    print(args)

    main(args)