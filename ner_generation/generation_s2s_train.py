import pandas as pd
import torch
import torch.optim.adamw as adamw
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5ForTokenClassification,
    T5GemmaForTokenClassification,
    EarlyStoppingCallback
)

import json
import argparse
import os

# 1. SETUP: LOAD AND PARSE THE DATA
# ------------------------------------
# We'll use Flan-T5, which is excellent for instruction-based tasks.
MODEL_CHECKPOINT = "google/flan-t5-base"
TRAIN_DATASET_PATH = "/workspace/datas/few-nerd/supervised/train.preprocessed.big.csv"
VAL_DATASET_PATH = "/workspace/datas/few-nerd/supervised/dev.preprocessed.big.csv"


def load_encoder_weight(args, model):
    # Load the encoder weights from the specified checkpoint
    if args.encoder_weight is not None:
        fine_tuned_classifier = None
        if "flan-t5" in args.model_checkpoint:
            fine_tuned_classifier = T5ForTokenClassification.from_pretrained(args.encoder_weight)
        elif "t5gemma" in args.model_checkpoint:
            fine_tuned_classifier = T5GemmaForTokenClassification.from_pretrained(args.encoder_weight)
        
        fine_tuned_state_dict = fine_tuned_classifier.state_dict()
        
        encoder_weights = {}
        for key, value in fine_tuned_state_dict.items():
            if "encoder" in key:
                t5_key = key
                if "flan-t5" in args.model_checkpoint:
                    t5_key = key.replace("transformer.", "")
                encoder_weights[t5_key] = value

        model.load_state_dict(encoder_weights, strict=False)
        
    else:
        return None

def main(args):
    train_dataset = Dataset.from_csv(args.train_file)
    val_dataset = Dataset.from_csv(args.validation_file)


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.model_max_length = 1024
    model_name = args.model_checkpoint.split("/")[-1]

    if "flan-t5" in model_name:
        new_words = ['{', '}']
        tokenizer.add_tokens(new_words)

    # load entity types
    entity_types_dict = []
    with open(args.entity_types_file, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
        entity_types_dict = json_data
    

    # 2. PREPROCESS THE DATA
    # ------------------------------------
    # We frame the task with a prefix to guide the model.
    #PREFIX = "Extract named entities as a Json format. Select entity type from the given list.\nEntity Types: {entity_types}\nSentence: ".format(entity_types=", ".join(entity_types))
    PREFIX = "Extract named entities as a Json format. Select entity type from the given list.\nEntity Types: {entity_types}\nSentence: "
    SURFIX = "\n JSON result: "
    #PREFIX_SIM = "Entity Types: {entity_types}\n".format(entity_types=", ".join(entity_types))

    def preprocess_function(examples):
        # Prepare inputs with the prefix
        inputs = []
        for sentence, type  in zip(examples["Sentence"], examples["types"]):
            entity_list = []
            if args.dataset_name == "mix":
                entity_list = entity_types_dict[type]
            else:
                entity_list = entity_types_dict[args.dataset_name]
            inputs.append(PREFIX.format(entity_types=", ".join(entity_list)) + sentence + '\n')

        #inputs = [PREFIX_SIM + doc + '\n' for doc in examples["Sentence"]]
        model_inputs = tokenizer(inputs, truncation=True)

        # Tokenize the target NER JSON strings
        # The `text_target` is the NER string itself.
        labels = tokenizer(text_target=examples["NER"], truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply the preprocessing to our datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)


    # 3. FINE-TUNE THE MODEL
    # ------------------------------------
    # Load the pre-trained T5 model
    if "t5gemma" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint, attn_implementation='eager', device_map="auto", dropout_rate=args.dropout_rate, dtype=torch.bfloat16, use_cache=False)
    elif "flan-t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint, device_map="auto")
        model.resize_token_embeddings(len(tokenizer))

    load_encoder_weight(args, model)

    # Data collator will dynamically pad inputs and labels
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    bf16_precision = torch.cuda.is_available() and model.dtype == torch.float32
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        do_eval=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=200,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.train_epochs,
        predict_with_generate=True,
        bf16=bf16_precision, # Use mixed precision if a GPU is available
        gradient_checkpointing=True,
        push_to_hub=False,
        report_to="tensorboard",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=3
    )
    
    
    # Create the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Start training! 🚀
    trainer.train()

    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=TRAIN_DATASET_PATH)
    parser.add_argument("--validation_file", type=str, default=VAL_DATASET_PATH)
    parser.add_argument("--model_checkpoint", type=str, default=MODEL_CHECKPOINT)
    
    parser.add_argument(
        "--output_dir", type=str, required=True
    )
    parser.add_argument(
        "--encoder_weight", type=str, default=None
    )
    parser.add_argument(
        "--train_epochs", type=int, default=10
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--entity_types_file", 
        type=str, 
        default="/workspace/datas/entity_types.json", 
        help="Path to the entity types file"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="fewnerd_big",
        help="Name of the dataset used for specifying entity types"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    
    args = parser.parse_args()
    '''args = parser.parse_args(
        [
            "--model_checkpoint", "google/t5gemma-b-b-prefixlm-it",
            "--output_dir", "/workspace/model_dir/test",
            "--train_file", "/workspace/datas/few-nerd/supervised/train.preprocessed.mix.csv",
            "--validation_file", "/workspace/datas/few-nerd/supervised/dev.preprocessed.mix.csv",
            "--dataset_name", "mix",
        ]
    )'''
    
    print(args)
    main(args)
