model_checkpoint_name="ner-inerd_ace05-fp32-w1e3-lr8e5"
model_name="gemma-3-270m-it"
model_checkpoint_path="/workspace/model_dir/generation/$model_name/$model_checkpoint_name/final_model"
dataset_names=("ace05")
dataset_paths=("/workspace/datas/ace05/test.json.csv")

index=0
while [ $index -lt 1 ]
do
    python /workspace/ner_generation/generation_inference.py \
        --model_checkpoint ${model_checkpoint_path} \
        --output_file /workspace/datas/generated/${model_name}-${model_checkpoint_name}-tuned-${dataset_names[$index]}.csv \
        --batch_size 32 \
        --decoder_model \
        --pipeline \
        --dataset_name ${dataset_names[$index]} \
        --data_file ${dataset_paths[$index]} \
        --prompt_type inerd
    
    index=`expr $index + 1`
done
