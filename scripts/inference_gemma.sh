
python /workspace/ner_generation/generation_inference.py \
    --model_checkpoint /workspace/model_dir/gemma-3-270m-it/ner-json-gen-mixed/final_model \
    --output_file /workspace/datas/generation/gemma-3-270m-it_tuned.csv \
    --batch_size 128 \
    --decoder_model