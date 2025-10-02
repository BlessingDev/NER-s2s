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

def main(args):
    if len(args.output_file) == 0:
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        args.output_file = f"/workspace/datas/generation/inference_results_{time_str}.csv"

    entity_types_dict = []
    with open(ENTITY_TYPE_FILE, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
        entity_types_dict = json_data
    
    test_dataset = Dataset.from_csv(args.data_file)
    #PREFIX = "Extract named entities as a Json format. \nNow, given the sentence: "
    PREFIX = "Extract named entities as a Json format. Select entity type from the given list.\nEntity Types: {entity_types}\nSentence: ".format(entity_types=", ".join(entity_types_dict[args.dataset_name]))
    SURFIX = "\n JSON result: "
    few_shot_PREFIX = f"Extract named entities as a Json format. Examples are: {data_string}\nNow, given the sentence: "

    MAX_INPUT_LENGTH = 1024
    MAX_TARGET_LENGTH = 512
    
    # Load the fine-tuned model and tokenizer
    
    if args.decoder_model:
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
        else:
            saved_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_checkpoint,
                device_map="auto"
            )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    
    def generate_predictions_s2s(batch):
        """Generates NER JSON for a batch of sentences."""
        # Prepare inputs with the prefix
        inputs_with_prefix = [PREFIX + sentence + "\n" for sentence in batch["Sentence"]]
        
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
    if args.decoder_model:
        pipe = pipeline("text-generation", model=saved_model, tokenizer=tokenizer, batch_size=args.batch_size)
        
    def generate_prediction_causal(batch):
        inputs_with_prefix = [[{"role": "user", "content": PREFIX + sentence}] for sentence in batch["Sentence"]]

        prompts = pipe.tokenizer.apply_chat_template(inputs_with_prefix, tokenize=False, add_generation_prompt=True)
        
        outputs = pipe(prompts, max_new_tokens=MAX_TARGET_LENGTH, disable_compile=True)

        predictions = [o[0]["generated_text"][len(prompts[i]):] for i, o in enumerate(outputs)]

        return {"generated_ner": predictions}

    generate_prediction_func = None
    if args.decoder_model:
        generate_prediction_func = generate_prediction_causal
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
        default=""
    )
    

    args = parser.parse_args()
    '''args = parser.parse_args([
        "--model_checkpoint", "google/t5gemma-2b-2b-prefixlm-it",
        "--batch_size", "32",
        "--dataset_name", "fewnerd_small",
        "--data_file", "/workspace/datas/few-nerd/supervised/test.preprocessed.small.csv"
    ])'''
    
    main(args)