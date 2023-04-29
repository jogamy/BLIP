CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --default_root_dir blip2-flan-t5-xl-coco \
  --max_epochs 5 --batch_size 4 \
  --lr 1e-5 --warmup_ratio 0.05 \
  --num_workers 8 --devices 2 \
  --dataset NICE --dropout 0.3 \
  \
  --enc_tok Salesforce/blip2-flan-t5-xl-coco \
  --dec_tok Salesforce/blip2-flan-t5-xl-coco \
  --model_card Salesforce/blip2-flan-t5-xl-coco
  