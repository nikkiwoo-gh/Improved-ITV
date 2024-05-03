# Create a concept list from captions
# Jiaxin Wu
# 2020.02.04
# concept_phase V1: verb phase and noun phase
# concept_phase V2: have proposition phase, quantity phase
# concept_phase V3: add lemmatization
from __future__ import print_function
import pickle
from collections import Counter
import json
import argparse
import os
import sys
import re
from basic.constant import ROOT_PATH, logger
from basic.common import makedirsforfile, checkToSkip
from basic.generic_utils import Progbar
from allennlp.predictors.predictor import Predictor
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
from nltk.corpus import wordnet


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def get_lemma(sent,nlp_lemmatizer):
    ##--way1:using  stanfordnlp lemmatizer
    # doc = nlp_lemmatizer(sent)
    # lemmas = []
    # for sent in doc.sentences:
    #     for word in sent.words:
    #         if word.lemma=='drine' and word.text=='drone':
    #             lemmas.append(word.text)
    #         else:
    #             lemmas.append(word.lemma)

    ##--way 2: using nltk
    lemmas=[]
    for word in nltk.word_tokenize(sent):
        lemma=nlp_lemmatizer.lemmatize(word, get_wordnet_pos(word))
        lemmas.append(lemma)
    return lemmas



def from_flickr_json(path):
    dataset = json.load(open(path, 'r'))['images']
    captions = []
    for i, d in enumerate(dataset):
        captions += [str(x['raw']) for x in d['sentences']]

    return captions



def get_concepts_from_parse_tree(tree, stop_words,caption,caption_lemmas,nlp_lemmatizer,islemma=True):
    concepts = []
    if isLeaf(tree):
        concept_new = None
        attr = tree['nodeType']
        word = tree['word']

        if (word == 'indoors') | (word == 'outdoors') | (word == 'daytime') | (word == 'nighttime') | (attr == 'NN') | (
                attr == 'NNS') | (attr == 'NNP') | (attr == 'NNPS') | (attr == 'VB') | (attr == 'VBG') | (
                attr == 'VBZ') | (attr == 'VBD') | (attr == 'VBP') | (attr == 'VBN'):
            if not word in stop_words:
                if islemma:
                    word_idx = np.where(np.array(caption.split())==word)[0]
                    if len(word_idx) > 0:
                        concept_new = caption_lemmas[word_idx[0]]
                    else:
                        print('!!'+word + ' is not in caption:' + caption)
                else:
                    concept_new = word
        return concept_new
    else:
        childrens = tree['children']
        for child in childrens:
            ##deal with word: NN, NNS, VBG
            concept_new = get_concepts_from_parse_tree(child, stop_words,caption,caption_lemmas,nlp_lemmatizer,islemma=islemma)
            if not concept_new is None:
                if isinstance(concept_new, list):
                    concepts = concepts + concept_new
                else:
                    concepts.append(concept_new)
            ##deal with phrase: 2,3words-NP, 2,3words-VP
            attr = child['nodeType']
            if (attr == 'NP') | (attr == 'VP') | (attr == 'NNP') | (attr == 'ADJP') | (attr == 'PP') | (attr == 'QP'):
                words = child['word']
                new_tokens = remove_stop_words(words, stop_words)
                new_tokens = remove_stop_words(' '.join(new_tokens), ['outdoors', 'indoors'])
                if (len(new_tokens) == 2) | (len(new_tokens) == 3) | (len(new_tokens) == 4)| (new_tokens == 'daytime') | (new_tokens == 'nighttime'):
                    if islemma:
                        concept_new = []
                        for word in new_tokens:
                            word_idx = np.where(np.array(caption.split()) == word)[0]
                            if len(word_idx)>0:
                                concept_new.append(caption_lemmas[word_idx[0]])
                            else:
                                print('!!'+word+' is not in caption:'+caption)
                        concepts.append(' '.join(concept_new))
                    else:
                        concepts.append(' '.join(new_tokens))


    return concepts


def isLeaf(tree):
    allkeys = tree.keys()
    if 'children' in allkeys:
        return False
    else:
        return True


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower()


def remove_stop_words(string, stop_words):
    new_tokens = []
    tokens = string.strip().lower().split()
    for word in tokens:
        if not word in stop_words:
            new_tokens.append(word)
    return new_tokens


def from_txt(txt):
    captions = []
    cap_ids = []
    with open(txt, 'rb') as reader:
        for line in reader:
            line = line.decode()
            cap_id, caption = line.split(' ', 1)
            captions.append(caption.strip())
            cap_ids.append(cap_id)
    return captions, cap_ids

def from_phrase(txt):
    captions = []
    cap_ids = []
    with open(txt, 'r',encoding="utf-8") as reader:
        for line in reader:
            line = line.encode("utf-8").decode("latin1")
            cap_id = line.split(':')[0]
            caption = line.split(':')[1:-1]
            captions.append(':'.join(caption).strip())
            cap_ids.append(cap_id)
    return captions, cap_ids
def build_concept_phrase(option,collection,nlp_lemmatizer,lemma = True):
    """Build a simple vocabulary wrapper."""

    cap_file = os.path.join(option.rootpath, collection, 'TextData', collection+'.'+option.capname)
    new_text_file = cap_file + '.concept_phrase_re.txt'

    if checkToSkip(new_text_file, option.overwrite):
        sys.exit(0)
    makedirsforfile(new_text_file)

    counter = Counter()
    print('processing %s'%cap_file)
    captions, cap_ids = from_txt(cap_file)
    # captions, cap_ids = from_phrase(cap_file)
    pbar = Progbar(len(captions))
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

    stop_word_file = os.path.join('/vireo00/nikki/AVS_data', 'stopwords_en_nikki.txt')
    stop_words = []
    with open(stop_word_file, 'rb') as reader:
        for word in reader:
            word = word.decode().strip()
            stop_words.append(word)

    capid_caption_conceptphrase = []
    for i, caption in enumerate(captions):
        cap_id = cap_ids[i]
        caption_clear = clean_str(caption.lower())
        if caption_clear =='':
            print("error in "+cap_id)
            continue
        caption_lemmas = get_lemma(caption_clear,nlp_lemmatizer)
        out = predictor.predict(caption_clear)
        constituency_tree_str = out['hierplane_tree']['root']
        concepts = get_concepts_from_parse_tree(constituency_tree_str, stop_words,caption_clear,caption_lemmas,nlp_lemmatizer,islemma=lemma)
        concepts = set(concepts)
        concepts = list(concepts)
        counter.update(concepts)
        capid_caption_conceptphrase.append(cap_id + '::' + caption + '::' + ','.join(concepts) + '\n')
        pbar.add(1)
        # if i > 20:
        #     break

    # Discard if the occurrence of the word is less than min_word_cnt.
    thresholds = option.threshold
    for threshold in thresholds.split(','):
        threshold = int(threshold)

        counter_file = os.path.join(option.rootpath, option.collection, 'TextData', 'concept_phrase',
                                   'concept_phrase_frequency_count_gt%s_re.txt' % threshold)
        words = []
        for word, cnt in counter.items():
            if (word=='or')|(word=='and')|(word=='not'):
                continue
            if word.find('drine')>-1:
                word = word.replace('drine','drone')
            if cnt >= threshold:
                if not word in stop_words:
                    words.append(word)

        # Create a vocab wrapper and add some special tokens.
        concept_counter_list = []
        for word, cnt in counter.items():
            if (word=='or')|(word=='and')|(word=='not'):
                continue
            if word.find('drine') > -1:
                word=word.replace('drine', 'drone')
            if cnt >= threshold:
                if not word in stop_words:
                    concept_counter_list.append([word, cnt])
        concept_counter_list.sort(key=lambda x: x[1], reverse=True)
        with open(counter_file, 'w') as writer:
            writer.write('\n'.join(map(lambda x: x[0] + ' %d' % x[1], concept_counter_list)))
        logger.info("Saved vocabulary counter file to %s", counter_file)


    outlines = []
    for line in capid_caption_conceptphrase:
        line = line.encode("utf-8").decode("latin1")
        outlines.append(line)
    with open(new_text_file, 'w', encoding='utf-8') as writer:
        writer.writelines(outlines)
    print('done')


def main(option):
    rootpath = option.rootpath
    collection = option.collection

    nlp_lemmatizer = WordNetLemmatizer()
    build_concept_phrase(option,collection,nlp_lemmatizer,lemma=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='root path. (default: %s)' % ROOT_PATH)
    parser.add_argument('collection', type=str, help='collection tgif|msrvtt10k')
    parser.add_argument('--capname', type=str, default='caption.txt', help='cap postfix if caption file')
    parser.add_argument('--threshold', type=str, default='5,10,15,20,25,30,40', help='threshold to build vocabulary. (default: 5)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1],
                        help='overwrite existed vocabulary file. (default: 0)')

    opt = parser.parse_args()
    print(json.dumps(vars(opt), indent=2))

    main(opt)

