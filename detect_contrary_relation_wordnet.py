import argparse
from nltk.corpus import wordnet
from util.util import Progbar, checkToSkip
import sys
import os
from nltk import pos_tag
from basic.constant import ROOT_PATH



def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('I'):
        return wordnet.PRO
    else:
        return None # for easy if-statement

def getAntonyms_wordnet(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return antonyms

def getAntonyms_Stanford_ACL2018_wordnet(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.extend(l._synset._lemma_names)
            if l.antonyms():
                antonyms.extend(l.antonyms()[0]._synset._lemma_names)

    return antonyms

def main(option):

    # set concept list for multi-label classification
    wordfile = os.path.join(option.rootpath, option.collection, 'TextData', 'concept_phrase',
                                'concept_phrase_frequency_count_gt%s_re.txt' % option.threshold)
    savename = wordfile+'.contradict'

    if checkToSkip(savename, option.overwrite):
        sys.exit(0)

    with open(wordfile,'r',encoding='utf-8') as reader:
        in_lines = reader.readlines()


    ##parse concept phase to specify format, e.g., bold man -> bold_man
    concepts = []
    for in_line in in_lines:
        concept = '_'.join(in_line.split()[0:-1])
        concepts.append(concept)


    pbar = Progbar(len(in_lines))
    out_lines = []
    num = 0
    for concept in concepts:
        out_line = concept.strip()
        synset = wordnet.synsets(concept)
        antonyms= []
        if len(synset) == 0:
            if len(concept.split('_'))<3:
                split_words = concept.split('_')
                tagged = pos_tag(split_words)
                for i,iconcept in enumerate(split_words):
                    tag = tagged[i][1]
                    if iconcept in concepts:
                        antonym = getAntonyms_wordnet(iconcept)
                        if len(antonym) > 0:
                            antonyms = antonyms + antonym
        else:
            # antonyms = getAntonyms_wordnet(concept)
            antonyms = getAntonyms_Stanford_ACL2018_wordnet(concept)
        antonyms = set(antonyms)
        antonyms = list(antonyms)
        antonyms_list = []
        for antonym in antonyms:
            if antonym in concepts:
                antonyms_list.append(antonym)
        if len(antonyms_list) > 0:
            out_line = out_line + '<->' + ','.join(list(antonyms_list))

        out_lines.append(out_line+'\n')
        pbar.add(1)
        num = num+1
    with open(savename,'w',encoding='utf-8') as writer:
        writer.writelines(out_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trainCollection', type=str, help='train collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,help='path to datasets')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed file. (default: 0)')
    parser.add_argument('--concept_bank', type=str, default='concept_word', help='concept_bank filename')
    parser.add_argument('--threshold', type=int, default=5, help='concept frequence threshold')

    opt = parser.parse_args()


    main(opt)

