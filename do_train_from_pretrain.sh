trainCollection=tgif-msrvtt10k-VATEX
valCollection=tv2016train
testCollection=iacc.3
n_caption=2
rootpath=/home/nikki/data/AVS_data
visual_feature=pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os+CLIP_ViT-B_32_vidFeature+BLIP2_vidFeature+imagebind_vidfeature
motion_feature=mean_slowfast+mean_swintrans
textual_feature=BLIP2_txtFeature+CLIP_ViT-B_32_txtFeature+imagebind_txtfeature

lr=0.0002
overwrite=1
epoch=100
direction=all
cost_style=sum
lambda=0.2
ul_alpha=0.01
decoder_layers=0-2048
classification_loss_type=favorBCEloss
concept_fre_threshold=20
concept_bank=concept_phrase


##tgif-msrvtt10k-VATEX


resume_path=/home/nikki/data/AVS_data/WebVid/tgif-msrvtt10k-VATEX/ICMR2024/WebVid_val/Improved_ITV_pretrain_word_only_dp_0.2_measure_cosine_lambda_0.2/BLIP2_txtFeature+CLIP_ViT-B_32_txtFeature+imagebind_txtfeature/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os+CLIP_ViT-B_32_vidFeature+BLIP2_vidFeature+imagebind_vidfeature_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048_decoder_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0002_decay_0.99_grad_clip_2.0_val_metric_recall/run_ImproveITV_pretrainWebVid_tgif-msrvtt10k-VATEXconcept_phrase_20
postfix=run_improvedITVfromPretrainModel_${concept_bank}_cfre${concept_fre_threshold}_4090

echo "CUDA_VISIBLE_DEVICES=$gpu python train.py $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	 --resume --resume_path $resume_path --concept_bank ${concept_bank} --textual_feature $textual_feature --with_textual_mapping --concept_fre_threshold $concept_fre_threshold --unlikelihood  --decoder_layers $decoder_layers --motion_feature $motion_feature --multiclass_loss_lamda ${lambda}  --ul_alpha ${ul_alpha} --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out"

CUDA_VISIBLE_DEVICES=$gpu python train.py  $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	 --resume --resume_path $resume_path --concept_bank ${concept_bank} --textual_feature $textual_feature --with_textual_mapping --concept_fre_threshold $concept_fre_threshold  --unlikelihood  --decoder_layers $decoder_layers --motion_feature $motion_feature --multiclass_loss_lamda ${lambda} --ul_alpha ${ul_alpha}  --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out
