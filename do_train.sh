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
postfix=run_improvedITV_${concept_bank}_cfre${concept_fre_threshold}_4090
echo "CUDA_VISIBLE_DEVICES=$gpu python train.py $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	 --concept_bank ${concept_bank} --textual_feature $textual_feature --with_textual_mapping --concept_fre_threshold $concept_fre_threshold --unlikelihood  --decoder_layers $decoder_layers --motion_feature $motion_feature --multiclass_loss_lamda ${lambda}  --ul_alpha ${ul_alpha} --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out"

CUDA_VISIBLE_DEVICES=$gpu python train.py  $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	 --concept_bank ${concept_bank} --textual_feature $textual_feature --with_textual_mapping --concept_fre_threshold $concept_fre_threshold  --unlikelihood  --decoder_layers $decoder_layers --motion_feature $motion_feature --multiclass_loss_lamda ${lambda} --ul_alpha ${ul_alpha}  --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out
