rootpath=/home/nikki/data/AVS_data
testCollection=iacc.3
query_sigmoid_threshold=0.99

logger_name=/home/nikki/data/AVS_data/tgif-msrvtt10k-VATEX/Improved_ITV_trained_models/run_Improved-ITV_tgif-msrvtt10k-VATEX_on_mean_slowfast+mean_swintrans_r152_101+C32B2bindFeature_lr02

checkpoint_name=model_best.pth.match.tar
#checkpoint_name=model_best.pth.class.tar

overwrite=0
query_sets=tv16.avs.txt,tv17.avs.txt,tv18.avs.txt
query_num_all=90
gpu=0


echo "CUDA_VISIBLE_DEVICES=$gpu python predictor.py  $testCollection  --checkpoint_name $checkpoint_name --query_num_all $query_num_all --query_sigmoid_threshold $query_sigmoid_threshold --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --query_sets $query_sets"
CUDA_VISIBLE_DEVICES=$gpu python predictor.py  $testCollection  --checkpoint_name $checkpoint_name --query_num_all $query_num_all --query_sigmoid_threshold $query_sigmoid_threshold --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --query_sets $query_sets
