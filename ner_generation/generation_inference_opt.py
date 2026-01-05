import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    DataLoader
)
import argparse
import datetime
import json
import os

# 1. SETUP: LOAD AND PARSE THE DATA
# ------------------------------------
# We'll use Flan-T5, which is excellent for instruction-based tasks.
# MODEL_CHECKPOINT = "google/flan-t5-base"
MODEL_CHECKPOINT = "/workspace/model_dir/flan-t5-base/ner-json-gen-mixed-encoder-1/final_model"
ENTITY_TYPE_FILE = "/workspace/datas/entity_types.json"

# Your provided dataset as a string
data_string = """Sentence: The final stage in the development of the Skyfox was the production of a model with tricycle landing gear to better cater for the pilot training market .
NER_result: {'product': ['Skyfox']}
Sentence: Also worth mentioning is the ultramarathon CajaMar Tenerife Bluetrail , the highest race in Spain and second in Europe , with the participation of several countries and great international repercussions .
NER_result: {'event': ['CajaMar Tenerife Bluetrail'], 'location': ['Spain', 'Europe']}
"""

#PREFIX = "Extract named entities as a Json format. \nNow, given the sentence: "
PREFIX = "Extract named entities as a Json format. Select entity type from the given list. {line_breaker} {entity_types} {line_breaker} {sentence} {line_breaker} "
#PREFIX_INERD = "List all named entities in order using the format Select entity type from the given list. {line_breaker} {entity_types} {line_breaker} {sentence} {line_breaker} <CT> "
PREFIX_INERD = "List all named entities in order following iNERD format. You can select entity types from given list. {line_breaker} {entity_types} {line_breaker} {sentence}"
PROMPT_SIMPLE = "named entity recognition {line_breaker} {entity_types} {line_breaker} {sentence}"
PREFIX_INERD_TEMPLATE = "Extract named entities as a iNERD format. Select entity type from the given list. // {entity_types} // {sentence} "

LINE_BREAKER = "#/" # wnut17에서는 ~/ 사용
SURFIX = "\n JSON result: "
few_shot_PREFIX = f"Extract named entities as a Json format. Examples are: {data_string}\nNow, given the sentence: "

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 1024

def main(args):
    if len(args.output_file) == 0:
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        args.output_file = f"/workspace/datas/generation/inference_results_{time_str}.csv"

    entity_types_dict = []
    with open(ENTITY_TYPE_FILE, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
        entity_types_dict = json_data
    
    test_dataset = Dataset.from_csv(args.data_file)
    if args.sample_testset < 1.0:
        test_dataset = test_dataset.shuffle(seed=42).select(range(int(len(test_dataset)*args.sample_testset)))
    
    # Load the fine-tuned model and tokenizer
    
    mask_token = "<mask>"
    if args.decoder_model:
        if "gemma-3" in args.model_checkpoint:
            from model_code.inerd_modeling_gemma3 import Gemma3ForCausalLM
            saved_model = Gemma3ForCausalLM.from_pretrained(
                args.model_checkpoint,
                device_map="auto"
            )
        else:
            saved_model = AutoModelForCausalLM.from_pretrained(
                args.model_checkpoint,
                device_map="auto"
            )
    else:
        if "t5gemma" in args.model_checkpoint:
            saved_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_checkpoint,
                device_map="auto",
                attn_implementation='eager',
                dtype=torch.bfloat16
            )
        elif "t5" in args.model_checkpoint:
            from model_code.inerd_modeling_t5 import T5ForConditionalInerdGeneration
            mask_token = "<extra_id_0>"
            inerd_version = 1
            if args.token_type == "inerd2":
                inerd_version = 2
            
            saved_model = T5ForConditionalInerdGeneration.from_pretrained(
                args.model_checkpoint,
                device_map="auto",
                inerd_version=inerd_version
            )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if args.token_type == "inerd" or args.token_type == "inerd2":
        # Initialize iNERD special tokens
        if args.decoder_model:
            saved_model.initialize_inerd(
                line_break_indicator=tokenizer.encode(LINE_BREAKER, add_special_tokens=False),
                start_of_turn_id=tokenizer.convert_tokens_to_ids("<start_of_turn>"),
                end_of_turn_id=tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                ct_token_id=tokenizer.convert_tokens_to_ids("<CT>"),
                space_token_id=tokenizer.encode(" ", add_special_tokens=False)[0],
                es_token_id=tokenizer.convert_tokens_to_ids("<ES>"),
                tcs_token_id=tokenizer.convert_tokens_to_ids("<TCS>")
            )
        else:
            saved_model.initialize_inerd(
                line_break_indicator=tokenizer.encode(LINE_BREAKER, add_special_tokens=False),
                ct_token_id=tokenizer.convert_tokens_to_ids(mask_token),
                space_token_id=tokenizer.encode(" ", add_special_tokens=False),
                es_token_id=tokenizer.convert_tokens_to_ids("<ES>"),
                tcs_token_id=tokenizer.convert_tokens_to_ids("<TCS>")
            )
    
    
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
            if args.prompt_setting == "inerd":
                sentence = PREFIX_INERD.format(entity_types=" ".join(entity_list), sentence=examples["Sentence"][idx], line_breaker=LINE_BREAKER)
            elif args.prompt_setting == "simple":
                sentence = PROMPT_SIMPLE.format(entity_types=" ".join(entity_list), sentence=examples["Sentence"][idx], line_breaker=LINE_BREAKER)
            elif args.prompt_setting == "json":
                sentence = PREFIX.format(entity_types=" ".join(entity_list), sentence=examples["Sentence"][idx], line_breaker=LINE_BREAKER)
            
            inputs.append(sentence)
            
        model_inputs = tokenizer(inputs, truncation=True)
        
        model_inputs["sentences"] = inputs

        # Tokenize the target NER JSON strings
        # The `text_target` is the NER string itself.
        # 빈 문자열이 NoneType으로 들어오는 문제를 해결
        ner_strings = [ner if (ner is not None) else "" for ner in examples["NER"]]
        labels = tokenizer(text_target=ner_strings, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=saved_model)
    
    training_args = TrainingArguments(
        output_dir=args.prediction_output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=False,
        do_predict=True,
        report_to="none", # Disable logging to wandb/tensorboard
    )
    trainer = Trainer(
        model=saved_model,
        args=training_args,
        processing_class=tokenizer
    )
    test_loader = DataLoader(tokenized_test_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator, shuffle=False)
    with torch.no_grad():
        predictions = trainer.prediction_loop(test_loader, description="Prediction")
    print("--- Inference Results ---")

    # You can easily display the results using a pandas DataFrame
    df = pd.DataFrame(results_dataset)

    output_file = os.path.join(args.prediction_output_dir, args.output_file)
    df.to_csv(output_file, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Generation Inference")
    
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--decoder_model",
        action="store_true"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true"
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="/workspace/datas/few-nerd/supervised/test.preprocessed.big.csv"
    )
    parser.add_argument(
        "--dataset_name",
        type=str, 
        default="fewnerd_big"
    )
    parser.add_argument(
        "--prediction_output_dir",
        type=str,
        default="/workspace/datas/generated/ner_generation_inference"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.csv"
    )
    parser.add_argument(
        "--sample_testset",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default="regular"
    )
    parser.add_argument(
        "--prompt_setting",
        type=str,
        default="simple"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3"
    )
    

    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "/workspace/model_dir/generation/t5-efficient-base/first/ner-inerd2_conll2003-conll2003_simple-fp32-w1e3-lr1e4-seed_42/final_model",
        #"--decoder_model",
        #"--pipeline",
        "--batch_size", "32",
        "--dataset_name", "conll2003",
        "--data_file", "/workspace/datas/conll2003/testb.inerd2.csv",
        "--prompt_setting", "simple",
        "--token_type", "inerd2"
    ])'''

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(args)
    
    main(args)