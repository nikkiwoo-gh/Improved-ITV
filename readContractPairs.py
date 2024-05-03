import os
import argparse
import sys
from util.utils import checkToSkip,Progbar
from basic.constant import ROOT_PATH

def main(option):


    contract_file = os.path.join(option.rootpath, opt.trainCollection, 'TextData', 'concept_phrase',
                                option.concept_bank+'_frequency_count_gt'+str(option.threshold)+'.txt.contradict')


    savename = contract_file+'.contradict_pairs'



    if checkToSkip(savename, option.overwrite):
        sys.exit(0)

    with open(contract_file,'r',encoding='utf-8') as reader:
        in_lines = reader.readlines()

    pbar = Progbar(len(in_lines))
    out_lines = []
    num = 0
    for in_line in in_lines:
        in_line_split = in_line.split('<->')
        if len(in_line_split)==2:
            concept_i =in_line_split[0]
            antonyms_i = in_line_split[1]
            out_lines.append(in_line)
        pbar.add(1)
        num = num+1
    with open(savename,'w',encoding='utf-8') as writer:
        writer.writelines(out_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,help='path to datasets')
    parser.add_argument('trainCollection', type=str, help='train collection')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed file. (default: 0)')
    parser.add_argument('--concept_bank', type=str, default='concept_phrase', help='concept_bank filename')
    parser.add_argument('--threshold', type=int, default=20, help='concept frequence threshold')

    opt = parser.parse_args()


    main(opt)
