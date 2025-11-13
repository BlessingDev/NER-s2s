import conll_preprocess
import classification_preprocess
import preprocess_utils
import json
import pathlib
import pandas as pd
import time

def main():
    print("Recon Bridging the Gap Module")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    
    split = "train"
    dataset_path = "fewnerd/supervised"

    #df, big_ner_types, small_ner_types = conll_preprocess.preprocess_fewnerd_conll(f"/workspace/datas/few-nerd/supervised/{split}.txt", focus_point="small")
    #df, ner_types = conll_preprocess.preprocess_simple_conll(f"/workspace/datas/{dataset_path}/{split}.txt")
    #df, ner_types = conll_preprocess.preprocess_conll_data(f"/workspace/datas/{dataset_path}/eng.{split}")
    #df = classification_preprocess.preprocess_conll_to_binary(f"/workspace/datas/{dataset_path}/eng.{split}")
    #df = classification_preprocess.preprocess_conll_to_classification(f"/workspace/datas/{dataset_path}/eng.{split}")
    #df = classification_preprocess.mit_ner_to_classification(f"/workspace/datas/{dataset_path}/restaurant{split}.bio.txt")
    #df = classification_preprocess.preprocess_sim_conll_to_classification(f"/workspace/datas/{dataset_path}/{split}.txt", is_bio=True)
    df = classification_preprocess.preprocess_classification_to_switch(f"/workspace/datas/{dataset_path}/{split}.classification.csv", "fewnerd_small", type_ratio=0.1, line_identifier="#/")
    
    #df = preprocess_utils.add_types_column_to_dataset(f"/workspace/datas/{dataset_path}/{split}.preprocessed.csv", "conll2003")
    #df = preprocess_utils.use_entity_list_for_entity_types_full(f"/workspace/datas/{dataset_path}/{split}.preprocessed.csv", "/workspace/datas/entity_types.json", min_length_ratio=0.5, max_length_ratio=1.0)
    #df = conll_preprocess.json_dataset_to_inerd(f"/workspace/datas/{dataset_path}/{split}.json.random.csv", json_column="NER")
    
    df.to_csv(f"/workspace/datas/{dataset_path}/{split}.switch.csv", index=False)
    
    #preprocess_utils.split_train_dev(f"/workspace/datas/{dataset_path}/train.classification.csv", f"/workspace/datas/{dataset_path}/dev.classification.csv", dev_ratio=0.2)

    #preprocess_utils.convert_ner_json_to_simplified_format(f"/workspace/datas/{dataset_path}/{split}.preprocessed.random.csv", f"/workspace/datas/{dataset_path}/{split}.preprocessed.random.sim.csv")

    # entity type 저장
    #preprocess_utils.save_entity_types(f"/workspace/datas/entity_types.json", "jnlpba", ner_types)

if __name__ == "__main__":
    main()