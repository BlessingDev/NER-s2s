"""
Fine-tune a T5 model for NER generation using seq2seq approach.
Use only normal token generation objective without any other auxiliary losses.
This script uses Hugging Face's Transformers and Datasets libraries.

writer: Junho Park (Blessingdev)
"""
import pandas as pd
import torch
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
import random
from tqdm.auto import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. SETUP: LOAD AND PARSE THE DATA
# ------------------------------------
# We'll use Flan-T5, which is excellent for instruction-based tasks.
MODEL_CHECKPOINT = "google/flan-t5-base"
TRAIN_DATASET_PATH = "/workspace/datas/few-nerd/supervised/train.preprocessed.big.csv"
VAL_DATASET_PATH = "/workspace/datas/few-nerd/supervised/dev.preprocessed.big.csv"

# We frame the task with a prefix to guide the model.
#PREFIX = "Extract named entities as a Json format. Select entity type from the given list.\nEntity Types: {entity_types}\nSentence: ".format(entity_types=", ".join(entity_types))
PREFIX = "Extract named entities as a Json format. Select entity type from the given list. {line_breaker} {entity_types} {line_breaker} {sentence} {line_breaker} "
PREFIX_INERD = "List all named entities in order following iNERD format. You can select entity types from given list. {line_breaker} {entity_types} {line_breaker} {sentence}"
# Refer to entity types by their index from the given list.

LINE_BREAKER = "#/" # wnut17에서는 ~/ 사용
SURFIX = "\n JSON result: "
#PREFIX_SIM = "Entity Types: {entity_types}\n".format(entity_types=", ".join(entity_types))

def inerd2_to_json(inerd_str):
    json_obj = dict()
    if len(inerd_str.strip()) > 0:
        entity_list = inerd_str.split("<ES>")
        for entity_str in entity_list:
            if len(entity_str.strip()) == 0:
                continue
            try:
                entity_item = entity_str.strip().split("<TCS>")
                type = entity_item[1].strip()
                entity = entity_item[0].strip()
                json_obj[type] = json_obj.get(type, []) + [entity]
            except Exception:
                raise json.JSONDecodeError("Invalid iNERD format.")
    return json_obj

def load_encoder_weight(args, model):
    # Load the encoder weights from the specified checkpoint
    if args.encoder_weight is not None:
        fine_tuned_classifier = None
        if "t5gemma" in args.model_checkpoint:
            fine_tuned_classifier = T5GemmaForTokenClassification.from_pretrained(args.encoder_weight)
        elif "t5" in args.model_checkpoint:
            fine_tuned_classifier = T5ForTokenClassification.from_pretrained(args.encoder_weight)
        
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

    # load entity types
    entity_types_dict = []
    with open(args.entity_types_file, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
        entity_types_dict = json_data
    

    # 2. PREPROCESS THE DATA
    # ------------------------------------

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # 3. FINE-TUNE THE MODEL
    # ------------------------------------
    # Load the pre-trained T5 model
    if "t5gemma" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint, attn_implementation='eager', device_map="auto", dropout_rate=args.dropout_rate, dtype=torch.bfloat16, use_cache=False)
    elif "t5" in model_name:
        dtype = torch.bfloat16 if args.bf16 else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint, device_map="auto", dtype=dtype)
    
    if args.token_setting == "inerd":
        tokenizer.add_tokens(['<CT>', '<ES>', '<TCS>', LINE_BREAKER])
        model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.add_tokens(['{', '}', "~", "\\", '・', '`', "``"])
    #"ø", "Ø", "Á", "Å", "Ā", "Ä", "ā", "á", "à", "å", "ắ", "ă", "ã", "ả", "â", "ą", "ẩ", "ầ", "ẵ", "ä", "ạ", "Č", "ć", "č", "Ç", "ç", "č̣", "Đ", "đ", "ḍ", "Ĕ", "é", "ë", "ę", "ē", "ê", "ệ", "ė", "ễ", "ě", "è", "ế", "ə", "Ġ", "ġ", "ğ", "ḥ", "İ", "Í", "ı", "í", "ì", "ī", "ï", "ĩ", "ḳ", "k̂", "ķ", "Ľ", "ṁ", "Ѝ", "ñ", "ń", "ṇ", "ṅ", "Ö", "Ō", "Ó", "Ò", "ò", "ó", "ō", "ŏ", "ö", "õ", "ố", "ð", "ơ", "ộ", "ờ", "ồ", "ό", "ő", "ổ", "ṛ", "ř", "Š", "Ś", "Ş", "ś", "š", "ş", "ṣ", "Ṭ", "Ț", "ṭ", "t̄", "ť", "Ú", "ū", "ú", "ŭ", "ư", "ự", "ử", "ů", "ü", "x́", "x̄", "Ý", "ý", "ÿ", "ỹ", "ŷ", "ỳ", "Ž", "Ż", "Ẓ", "ž", "ż", "ź", "α", "β", "δ", "σ", "ς", "ά", "ᾰ", "Λ", "ε", "θ", "Φ", "λ", "π", "μ", "η", "ί", "Δ", "τ", "ζ", "Ω", "ω", "ώ", "Χ", "Κ", "κ", "ι", "ρ", "γ", "υ", "ν", "ύ", "ὑ", "έ", "Þ", "д", "ь", "П", "п", "Л", "л", "и", "ц", "к", "в", "ш", "т", "з", "г", "й", "Б", "м", "Я", "я", "љ", "ł", "Ł", "‡", "₹", "रु", "¥", "$", "€", "Σ", "Æ", "œ", "æ", "ы","Α", "а", "ɒ", "В", "Β", "С", "с", "е", "Н", "Η", "н", "Ј", "ј", "о", "ο", "Р", "р", "Т", "у", "У", "×", "ҳ", "ʻ", "ʼ", "ʽ", "−", "‑", "ʾ", "ˈ", "ʿ" "´"]) # few-nerd 데이터셋 등장하는 unk 토큰
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        # Prepare inputs with the prefix
        inputs = []
        for idx in range(len(examples["Sentence"])):
            entity_list = []
            if args.dataset_name == "given":
                entity_list = examples["entity_list"][idx].split(" ")
            elif args.dataset_name == "given_index":
                entity_list = examples["entity_list"][idx].split(" ")
                for idx in range(len(entity_list)):
                    entity_list[idx] = f"{idx}:{entity_list[idx]}"
            else:
                entity_list = entity_types_dict[args.dataset_name]

            sentence = ""
            if args.token_setting == "inerd":
                sentence = PREFIX_INERD.format(entity_types=" ".join(entity_list), sentence=examples["Sentence"][idx], line_breaker=LINE_BREAKER)
            else:
                sentence = PREFIX.format(entity_types=" ".join(entity_list), sentence=examples["Sentence"][idx], line_breaker=LINE_BREAKER)
            
            if args.chat_template:
                inputs.append([{"role": "user", "content": sentence}])
            else:
                inputs.append(sentence)

        #inputs = [PREFIX_SIM + doc + '\n' for doc in examples["Sentence"]]
        if args.chat_template:
            inputs = tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_token=True)
            
        model_inputs = tokenizer(inputs, truncation=True)
        
        model_inputs["sentences"] = inputs
        
        valid_entity_types = []
        for idx in range(len(examples["Sentence"])):
            ner = examples["NER"][idx]
            if pd.isna(ner):
                valid_entity_types.append([])
            else:
                if args.token_setting == "inerd":
                    ner_json = inerd2_to_json(ner)
                else:
                    ner_json = json.loads(ner)
                valid_entity_types.append(list(ner_json.keys()))
        model_inputs["valid_entity_types"] = valid_entity_types

        # Tokenize the target NER JSON strings
        # The `text_target` is the NER string itself.
        # 빈 문자열이 NoneType으로 들어오는 문제를 해결
        ner_strings = [ner if (ner is not None) else "" for ner in examples["NER"]]
        labels = tokenizer(text_target=ner_strings, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply the preprocessing to our datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
    
    load_encoder_weight(args, model)
    
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["sentences", "valid_entity_types", "Sentence", "NER"])
    if "entity_list" in tokenized_train_dataset.column_names:
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(["entity_list"])
    
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["sentences", "valid_entity_types", "Sentence", "NER"])
    if "entity_list" in tokenized_val_dataset.column_names:
        tokenized_val_dataset = tokenized_val_dataset.remove_columns(["entity_list"])
    
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    bf16_precision = torch.cuda.is_available() and model.dtype == torch.float32
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        do_train=True,
        do_eval=True,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": args.num_cycles},
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.train_epochs,
        generation_max_length=1024,
        bf16=bf16_precision, # Use mixed precision if a GPU is available
        predict_with_generate=True,
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
        "--encoder_weight", type=str, default=None
    )
    parser.add_argument(
        "--bf16",
        action="store_true"
    )
    parser.add_argument(
        "--encoder_train_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--type_binary",
        action="store_true"
    )
    parser.add_argument(
        "--encoder_loss_alpha",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--train_epochs", type=int, default=10
    )
    parser.add_argument(
        "--num_cycles", type=int, default=10
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01
    )
    parser.add_argument(
        "--learning_rate", type=float, default=4e-5
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=50
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10
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
            "--model_checkpoint", "google/flan-t5-base",
            #"--encoder_weight", "/workspace/model_dir/flan-t5-large/encoder-classification-conll2003-ner-lr2e-5-cosine_restart/final_model",
            "--batch_size", "32",
            #"--encoder_train_file", "/workspace/datas/fewnerd/supervised/train.binary.csv",
            "--type_binary",
            "--output_dir", "/workspace/model_dir/test",
            "--train_file", "/workspace/datas/ontonotes5/train.inerd2.contrastive.csv",
            "--validation_file", "/workspace/datas/ontonotes5/dev.inerd2.random.csv",
            "--dataset_name", "given",
            "--token_setting", "inerd",
        ]
    )'''
    
    print(args)
    main(args)
