rootpath=/home/nikki/data/AVS_data
testCollection=v3c1
query_sigmoid_threshold=0.99

#logger_name=/home/nikki/data/AVS_data/tgif-msrvtt10k-VATEX/Improved_ITV_trained_models/run_Improved-ITV_tgif-msrvtt10k-VATEX_on_mean_slowfast+mean_swintrans_r152_101+C32B2bindFeature_lr02
logger_name=/home/nikki/data/AVS_data/tgif-msrvtt10k-VATEX/ICMR2024/tv2016train/Improved_ITV_word_only_dp_0.2_measure_cosine_lambda_0.2/BLIP2_txtFeature+CLIP_ViT-B_32_txtFeature+imagebind_txtfeature/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os+CLIP_ViT-B_32_vidFeature+BLIP2_vidFeature+imagebind_vidfeature_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048_decoder_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0002_decay_0.99_grad_clip_2.0_val_metric_recall/run_improvedITV_concept_phrase_cfre20_4090

checkpoint_name=model_best.pth.match.tar

overwrite=0
query_sets=tv19.avs.txt,tv20.avs.txt,tv21.avs.txt
query_num_all=70
gpu=0


echo "CUDA_VISIBLE_DEVICES=$gpu python predictor.py $testCollection  --checkpoint_name $checkpoint_name --query_num_all $query_num_all --query_sigmoid_threshold $query_sigmoid_threshold --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --query_sets $query_sets"
CUDA_VISIBLE_DEVICES=$gpu python predictor.py  $testCollection  --checkpoint_name $checkpoint_name --query_num_all $query_num_all --query_sigmoid_threshold $query_sigmoid_threshold --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --query_sets $query_sets

