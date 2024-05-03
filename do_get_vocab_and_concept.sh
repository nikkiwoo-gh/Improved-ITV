collection=$1
rootpath=/vireo00/nikki/AVS_data
threshold=20
overwrite=1

##step 1: obtain concept and contrary relations

pwd
echo "python build_concept_phrase.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite"
python build_concept_phrase.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite


#echo "python detect_contrary_relation_wordnet.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite"
#python detect_contrary_relation_wordnet.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite

#echo "python readContractPairs.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite"
#python readContractPairs.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite

