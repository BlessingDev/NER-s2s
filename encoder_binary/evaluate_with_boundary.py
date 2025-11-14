import torch
import numpy as np
from scipy.special import expit, softmax
from torch.utils.data import DataLoader
from custom_dataset import TokenClassificationDataset
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


    # Load the model for token classification
    # THIS IS THE KEY STEP: T5ForTokenClassification uses the T5 encoder ONLY.
    # The decoder is not used.
    

    predictions = np.load(os.path.join(args.prediction_dir, "test_predictions.npy"))
    labels = np.load(os.path.join(args.prediction_dir, "test_labels.npy"))

    label_names = range(predictions.shape[2])
    
    
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
    
    metrics = compute_metrics((predictions, labels))

    # 6. SAVE AND PRINT RESULTS
    print("\n--- Test Set Metrics ---")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"True Negative Rate: {metrics['true_negative_rate']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    
    # Save metrics to a JSON file
    metrics_output_path = os.path.join(args.prediction_dir, f"test_metrics_dc{int(args.decision_threshold * 100)}.json")
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nTest metrics saved to: {metrics_output_path}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary NER model")

    parser.add_argument(
        "--entity_types_file", 
        type=str, 
        default="/workspace/datas/entity_types.json", 
        help="Path to the entity types file"
    )
    
    parser.add_argument(
        "--prediction_dir", type=str, default="/workspace/datas/encoder/test", help="Path to the output directory"
    )
    
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification"
    )
    
    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "/workspace/model_dir/classification/flan-t5-base/conll2003/encoder-switch-ner-custom-drop20-cycle10-lr2e-4-cosine_restart/final_model",
        "--test_file", "/workspace/datas/conll2003/testb.switch.csv",
        "--custom_model",
        "--batch_size", "256",
        "--prediction_output_dir", "/workspace/datas/encoder/test"
    ])'''

    print(args)

    main(args)