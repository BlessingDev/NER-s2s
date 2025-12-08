import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)
import argparse
import datetime
import json

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
PREFIX_INERD_TEMPLATE = "Extract named entities as a iNERD format. Select entity type from the given list. // {entity_types} // {sentence} "
LINE_BREAK_INDICATOR = "#/"
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
            from model_code.inerd_modeling_t5 import T5ForConditionalGeneration
            inerd_version = 1
            if args.prompt_type == "inerd2":
                inerd_version = 2
            
            saved_model = T5ForConditionalGeneration.from_pretrained(
                args.model_checkpoint,
                device_map="auto",
                inerd_version=inerd_version
            )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if args.prompt_type == "inerd" or args.prompt_type == "inerd2":
        # Initialize iNERD special tokens
        if args.decoder_model:
            saved_model.initialize_inerd(
                line_break_indicator=tokenizer.encode(LINE_BREAK_INDICATOR, add_special_tokens=False),
                start_of_turn_id=tokenizer.convert_tokens_to_ids("<start_of_turn>"),
                end_of_turn_id=tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                ct_token_id=tokenizer.convert_tokens_to_ids("<CT>"),
                space_token_id=tokenizer.encode(" ", add_special_tokens=False)[0],
                es_token_id=tokenizer.convert_tokens_to_ids("<ES>"),
                tcs_token_id=tokenizer.convert_tokens_to_ids("<TCS>")
            )
        else:
            saved_model.initialize_inerd(
                line_break_indicator=tokenizer.encode(LINE_BREAK_INDICATOR, add_special_tokens=False),
                ct_token_id=tokenizer.convert_tokens_to_ids("<CT>"),
                space_token_id=tokenizer.encode(" ", add_special_tokens=False),
                es_token_id=tokenizer.convert_tokens_to_ids("<ES>"),
                tcs_token_id=tokenizer.convert_tokens_to_ids("<TCS>")
            )
    
    def generate_predictions_s2s(batch):
        """Generates NER JSON for a batch of sentences."""
        # Prepare inputs with the prefix
        inputs_with_prefix = []
        for idx in range(len(batch["Sentence"])):
            entity_list = []
            if args.dataset_name == "mix":
                entity_list = entity_types_dict[batch["types"][idx]]
            elif args.dataset_name == "random":
                entity_list = batch["entity_list"][idx].split(" ")
                for idx in range(len(entity_list)):
                    entity_list[idx] = f"{idx}:{entity_list[idx]}"
            else:
                entity_list = entity_types_dict[args.dataset_name]
            
            if args.prompt_type == "inerd" or args.prompt_type == "inerd2":
                inputs_with_prefix.append(PREFIX_INERD.format(entity_types=" ".join(entity_list), sentence=batch["Sentence"][idx], line_breaker=LINE_BREAK_INDICATOR))
            else:
                inputs_with_prefix.append(PREFIX.format(entity_types=" ".join(entity_list), sentence=batch["Sentence"][idx], line_breaker=LINE_BREAK_INDICATOR))
        #inputs_with_prefix = [PREFIX + sentence + "\n" for sentence in batch["Sentence"]]
        
        # Tokenize the entire batch
        inputs = tokenizer(
            inputs_with_prefix, 
            padding=True,
            truncation=True, 
            max_length=MAX_INPUT_LENGTH, 
            return_tensors="pt"
        )
        
        # Move tokenized inputs to the same device as the model
        inputs = {k: v.to(saved_model.device) for k, v in inputs.items()}

        saved_model.initialize_inerd_batch(inputs["input_ids"])
        # Generate outputs
        output_sequences = saved_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_TARGET_LENGTH
        )
        
        # Decode the generated sequences
        predictions = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        # Return the predictions as a new column
        return {"generated_ner": predictions}
    
    pipe = None
    if args.pipeline and not args.decoder_model:
        # For seq2seq models
        pipe = pipeline("text2text-generation", model=saved_model, tokenizer=tokenizer, batch_size=args.batch_size)
    elif args.decoder_model:
        pipe = pipeline("text-generation", model=saved_model, tokenizer=tokenizer, batch_size=args.batch_size)
        pipe.tokenizer.padding_side = "left"  # For causal
        pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
        
    def generate_prediction_pipeline(batch):
        REGULAR_TERM = "Generate only one JSON result without any additional text. \n"
        inputs_with_prefix = []
        for idx in range(len(batch["Sentence"])):
            entity_list = []
            if args.dataset_name == "mix":
                entity_list = entity_types_dict[batch["types"][idx]]
            elif args.dataset_name == "random":
                entity_list = batch["entity_list"][idx].split(" ")
                for idx in range(len(entity_list)):
                    entity_list[idx] = f"{idx}:{entity_list[idx]}"
            else:
                entity_list = entity_types_dict[args.dataset_name]
                
            if args.prompt_type == "inerd" or args.prompt_type == "inerd2":
                user_message = PREFIX_INERD_TEMPLATE.format(entity_types=" ".join(entity_list), sentence=batch["Sentence"][idx])
            else:
                user_message = PREFIX.format(entity_types=", ".join(entity_list)) + batch["Sentence"][idx]
            
            if args.zero_shot:
                user_message += REGULAR_TERM
            
            inputs_with_prefix.append([{"role": "user", "content": user_message}])

        prompts = pipe.tokenizer.apply_chat_template(inputs_with_prefix, tokenize=False, add_generation_prompt=True)
        
        if args.prompt_type == "inerd" or args.prompt_type == "inerd2":
            #prompts = [prompt + "<CT> " for prompt in prompts]
            
            saved_model.initialize_inerd_batch(pipe.tokenizer(prompts, padding=True, return_tensors="pt").input_ids)
        
        outputs = pipe(prompts, max_new_tokens=MAX_TARGET_LENGTH)
        
    
        predictions = None
        if args.decoder_model:
            predictions = [o[0]["generated_text"][len(prompts[i]):] for i, o in enumerate(outputs)]
            
            # <CT> 제거하기
            if args.prompt_type == "inerd" or args.prompt_type == "inerd2":
                for i in range(len(predictions)):
                    predictions[i] = predictions[i][4:].strip()
        else:
            predictions = [o["generated_text"] for o in outputs]

        return {"generated_ner": predictions}
    
    generate_prediction_func = None
    if args.pipeline or args.decoder_model:
        generate_prediction_func = generate_prediction_pipeline
    else:
        generate_prediction_func = generate_predictions_s2s

    results_dataset = test_dataset.map(
        generate_prediction_func, 
        batched=True, 
        batch_size=args.batch_size # Adjust batch size based on your GPU memory
    )
    
    print("--- Inference Results ---")

    # You can easily display the results using a pandas DataFrame
    df = pd.DataFrame(results_dataset)

    df.to_csv(args.output_file, index=False)
    

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
        "--output_file",
        type=str,
        default="/workspace/datas/generated/inference_results.csv"
    )
    parser.add_argument(
        "--sample_testset",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="regular"
    )
    

    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "/workspace/model_dir/generation/flan-t5-base/ner-inerd2_given-conll2003-curriculum2-fp32-w1e3-lr1e4-seed_42-staretegy2/final_model",
        #"--decoder_model",
        #"--pipeline",
        "--batch_size", "32",
        "--dataset_name", "conll2003",
        "--data_file", "/workspace/datas/conll2003/testb.inerd2.csv",
        "--prompt_type", "inerd2"
    ])'''

    print(args)
    
    main(args)