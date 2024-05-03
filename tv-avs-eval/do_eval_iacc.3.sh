rootpath=/home/nikki/data/AVS_data
testCollection=iacc.3
overwrite=1
topk=1000

for topic_set in {tv16,tv17,tv18}; do

    for theta in {0_0,0_3,0_5,1_0}
        do
	    score_file=/home/nikki/data/AVS_data/iacc.3/results/${topic_set}.avs.txt/tgif-msrvtt10k-VATEX/tv2016train/Improved_ITV_trained_models/run_Improved-ITV_iacc.3_on_mean_slowfast+mean_swintrans_r152_101+C32B2bindFeature_lr02/model_best.pth.match.tar/id.sent.sim.0.99.combinedDecodedConcept_theta${theta}_score

            bash do_txt2xml.sh $rootpath $testCollection $score_file $topic_set $topk $overwrite
            echo python trec_eval.py ${score_file}.xml --topk $topk --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite
            python trec_eval.py ${score_file}.xml --topk $topk --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite

    done
done
