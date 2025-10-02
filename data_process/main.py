import conll_preprocess
import json
import pathlib
import pandas as pd

def save_entity_types(entity_file_path, dataset_name, entity_types):
    entity_file = pathlib.Path(entity_file_path)
    entity_file.parent.mkdir(parents=True, exist_ok=True)
    
    if entity_file.exists():
        print(f"{entity_file_path} already exists. Loading existing entity types.")
        existing_types = None
        with open(entity_file_path, 'r', encoding='utf-8') as f:
            existing_types = json.loads(f.read())
        
        existing_types[dataset_name] = entity_types
        with open(entity_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(existing_types, ensure_ascii=False))
    else:
        print(f"{entity_file_path} does not exist. Creating new entity types file.")
        with open(entity_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({dataset_name: entity_types}, ensure_ascii=False))

def mix_big_and_small(data_directory, split, ratio=0.5):
    big_df = pd.read_csv(f"{data_directory}/few-nerd/supervised/{split}.preprocessed.big.csv")
    small_df = pd.read_csv(f"{data_directory}/few-nerd/supervised/{split}.preprocessed.small.csv")

    # Sample from the small dataframe
    small_sample = small_df.sample(frac=ratio)
    small_sample.loc[:, "types"] = "fewnerd_small"
    
    small_index = small_sample.index
    big_sample = big_df[~big_df.index.isin(small_index)]
    big_sample.loc[:, "types"] = "fewnerd_big"

    combined_df = pd.concat([big_sample, small_sample], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle the combined dataframe
    return combined_df

def main():
    print("Recon Bridging the Gap Module")
    
    split = "test"

    #df, big_ner_types, small_ner_types = conll_preprocess.preprocess_fewnerd_conll(f"/workspace/datas/few-nerd/supervised/{split}.txt", focus_point="small")
    df, ner_types = conll_preprocess.preprocess_MIT_conll(f"/workspace/datas/mit restaurant/restauranttest.bio.txt")
    
    #df = mix_big_and_small("/workspace/datas", split, ratio=0.5)
    
    df.to_csv(f"/workspace/datas/mit restaurant/{split}.preprocessed.csv", index=False)
    
    # entity type 저장
    save_entity_types(f"/workspace/datas/entity_types.json", "mit_restaurant", ner_types)

if __name__ == "__main__":
    main()