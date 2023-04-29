CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --default_root_dir blip2-opt-2.7b \
  --max_epochs 5 --batch_size 2 \
  --lr 1e-5 --warmup_ratio 0.05 \
  --num_workers 8 --devices 2 \
  --dataset NICE --dropout 0.3 \
  \
  --model_card Salesforce/blip2-opt-2.7b \
  --lm_card google/flan-t5-base
  

  #Salesforce/blip2-opt-2.7b
  #Salesforce/blip2-flan-t5-xl
  #Salesforce/blip2-flan-t5-xxl
  #Salesforce/blip2-opt-6.7b
  #Salesforce/blip2-opt-6.7b-coco
  #Salesforce/blip2-opt-2.7b-coco
  #Salesforce/blip2-flan-t5-xl-coco

  #google/flan-t5-small
  #google/flan-t5-base
  #google/flan-t5-large
