import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)
from trl import (
    SFTTrainer,
    SFTConfig
)
import argparse
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# 1. SETUP: LOAD AND PARSE THE DATA
# ------------------------------------
# We'll use Flan-T5, which is excellent for instruction-based tasks.
MODEL_CHECKPOINT = "google/gemma-3-270m-it"
TRAIN_DATASET_PATH = "/workspace/datas/few-nerd/supervised/train.preprocessed.csv"
VAL_DATASET_PATH = "/workspace/datas/few-nerd/supervised/dev.preprocessed.csv"

# We frame the task with a prefix to guide the model.
#PREFIX = "Extract named entities as a Json format. Select entity type from the given list.\nEntity Types: {entity_types}\nSentence: ".format(entity_types=", ".join(entity_types))
PREFIX = "Extract named entities as a Json format. Select entity type from the given list. // {entity_types} // {sentence} "
PREFIX_INERD = "Extract named entities as a iNERD format. Select entity type from the given list. // {entity_types} // {sentence} "
# Refer to entity types by their index from the given list.
SURFIX = "\n JSON result: "
#PREFIX_SIM = "Entity Types: {entity_types}\n".format(entity_types=", ".join(entity_types))
MAX_INPUT_LENGTH = 1024

def main(args):
    #model_name = args.model_checkpoint.split("/")[-1]

    # load entity types
    entity_types_dict = []
    with open(args.entity_types_file, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
        entity_types_dict = json_data

    # Create a Hugging Face Dataset
    train_dataset = Dataset.from_csv(args.train_file)
    val_dataset = Dataset.from_csv(args.validation_file)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint, 
        device_map="auto",
        attn_implementation='eager',
    )

    if args.token_setting == "inerd":
        tokenizer.add_tokens(['<CT>', '<ES>', '<TCS>'])
        model.resize_token_embeddings(len(tokenizer))
    

    # 2. PREPROCESS THE DATA
    # ------------------------------------
    # We frame the task with a prefix to guide the model.
    def preprocess_function(sample):
        entity_list = []
        if args.dataset_name == "mix":
            entity_list = entity_types_dict[sample["types"]]
        elif args.dataset_name == "random":
            entity_list = sample["entity_list"].split(" ")
        elif args.dataset_name == "mix_random":
            entity_list = sample["entity_list"].split(" ")
            for idx in range(len(entity_list)):
                entity_list[idx] = f"{idx}:{entity_list[idx]}"
        else:
            entity_list = entity_types_dict[args.dataset_name]
        
        prefix = PREFIX
        if args.token_setting == "inerd":
            prefix = PREFIX_INERD
        
        
        if sample["NER"] is None:
            sample["NER"] = ""
        
        assistant_text = sample["NER"]
        if args.token_setting == "inerd":
            assistant_text = "<CT> " + sample["NER"]
        
        # Prepare inputs with the prefix
        return {
          "messages": [
              {"role": "user", "content": prefix.format(
                  entity_types=" ".join(entity_list), sentence=sample["Sentence"])},
              {"role": "assistant", "content": assistant_text}
          ]
        }

    # Apply the preprocessing to our datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function)
    tokenized_val_dataset = val_dataset.map(preprocess_function)


    # 3. FINE-TUNE THE MODEL
    # ------------------------------------
    # Load the pre-trained causal LM model

    bf16_precision = torch.cuda.is_available() and model.dtype == torch.float32

    # Define training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        do_train=True,
        do_eval=True,
        packing=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 10},
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps, 
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.train_epochs, # Increase epochs for small datasets
        gradient_checkpointing=True,
        metric_for_best_model="eval_loss",
        bf16=bf16_precision, # Use mixed precision if a GPU is available
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )

    # Create the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Start training! 🚀
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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
        "--logging_steps", type=int, default=200
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--chat_template",
        action="store_true"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true"
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
        "--token_setting",
        type=str,
        default="normal",
        help="Token setting: normal, t5_json, inerd"
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
            "--model_checkpoint", "google/gemma-3-270m-it",
            "--output_dir", "/workspace/model_dir/test",
            "--train_file", "/workspace/datas/conll2003/train.inerd.csv",
            "--validation_file", "/workspace/datas/conll2003/testa.inerd.csv",
            "--dataset_name", "conll2003",
            "--token_setting", "inerd",
        ]
    )'''
    
    print(args)
    main(args)