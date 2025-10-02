
model_checkpoint="/workspace/model_dir/flan-t5-base/ner-json-mix_fp32_encoder_w1e3/final_model"
model_name="flan-t5-base"
dataset_names=("fewnerd_small")
dataset_paths=("/workspace/datas/few-nerd/supervised/test.preprocessed.small.csv")

index=0
while [ $index -lt 1 ]
do
    python /workspace/ner_generation/generation_inference.py \
        --model_checkpoint ${model_checkpoint} \
        --output_file /workspace/datas/generation/${model_name}_encoder_fewnerd-tuned_${dataset_names[$index]}.csv \
        --batch_size 128 \
        --dataset_name ${dataset_names[$index]} \
        --data_file ${dataset_paths[$index]}
    
    index=`expr $index + 1`
done
