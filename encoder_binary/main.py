import binary_preprocess
import pandas as pd
from transformers import AutoTokenizer

def main():
    print("encoder_binary_ner")

    file_path = "/workspace/datas/few-nerd/supervised/test.binary.csv"
    df = pd.read_csv(file_path)
    
    model_checkpoint = "google/flan-t5-xl"
    #model_checkpoint = "google/t5gemma-ml-ml-ul2"
    # 둘 중 어느 모델이 더 좋을지 비교해보자.
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    result_df = binary_preprocess.preprocess_binary_to_tokenization(df, tokenizer=tokenizer)

    result_df.to_csv("/workspace/datas/few-nerd/supervised/test.binary.flan_t5_xl.csv", index=False)
    #df = binary_preprocess.proprocess_ner_to_binary("/workspace/datas/few-nerd/supervised/dev.txt")

    #df.to_csv("/workspace/datas/few-nerd/supervised/dev.binary.csv", index=False)

if __name__ == "__main__":
    main()