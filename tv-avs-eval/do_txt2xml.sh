etime=1.0

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 rootpath testCollection score_file edition"
    exit
fi

rootpath=$1
test_collection=$2
score_file=$3
edition=$4
topk=$5
overwrite=$6

echo python txt2xml.py $test_collection $score_file --edition $edition --topk $topk --priority 1 --etime $etime --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite

python txt2xml.py $test_collection $score_file --edition $edition --topk $topk --priority 1 --etime $etime --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite


