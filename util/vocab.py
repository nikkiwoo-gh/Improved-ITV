import os
import pickle
from collections import Counter
import json
import argparse
import sys
from utils import clean_str, Progbar,from_txt,checkToSkip,makedirsforfile


ROOT_PATH = '/vireo00/nikki/AVS_data'
class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, text_style):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.text_style = text_style

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx and 'bow' not in self.text_style:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class Concept(object):
    """ concept wrapper"""

    def __init__(self):
        self.concept2idx = {}
        self.idx2concept = {}
        self.idx2contractIdx = {}
        self.concept2contractconcept = {}
        self.idx = 0
        self.num_contradict_paris = 0

    def add_concept(self, concept):
        if concept not in self.concept2idx:
            self.concept2idx[concept] = self.idx
            self.idx2concept[self.idx] = concept
            self.idx += 1

    def add_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_concept_str = contract_pair[0]
        ori_concept = ori_concept_str.replace('_',' ')
        ori_concept_idx = None
        if ori_concept in self.concept2idx:
            ori_concept_idx = self.concept2idx[ori_concept]
        contract_concept_str = contract_pair[1]
        contract_concepts = contract_concept_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_concepts:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.concept2idx:
                idx = self.concept2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_concept_idx is None)&(len(contract_idxes)>0):
            self.idx2contractIdx[ori_concept_idx] = contract_idxes
            self.concept2contractconcept[ori_concept] = ','.join(icontra_ori_list)
            self.num_contradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_concept)

    def __call__(self, concept):
        if concept not in self.concept2idx:
            return self.concept2idx['<unk>']
        return self.concept2idx[concept]

    def __len__(self):
        return len(self.concept2idx)

def build_vocab(collection, text_style, threshold=4, rootpath=ROOT_PATH):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    cap_file = os.path.join(rootpath, collection, 'TextData', '%s.caption.txt'%collection)
    cap_ids,captions = from_txt(cap_file)
    pbar = Progbar(len(captions))

    for i, caption in enumerate(captions):
        tokens = clean_str(caption.lower())
        counter.update(tokens)

        pbar.add(1)
        # if i % 1000 == 0:
        #     print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary(text_style)
    if 'rnn' in text_style:
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab, counter

class Concept_phase(object):
    """ concept wrapper"""

    def __init__(self):
        self.phrase2idx = {}
        self.idx2phrase = {}
        self.idx2contractIdx = {}
        self.phrase2contractphrase = {}
        self.idx2GlobalContractIdx = {}
        self.phrase2GlobalContractphrase = {}
        self.idx = 0
        self.num_contradict_paris = 0
        self.num_GlobalContradict_paris = 0

    def add_phrase(self, phrase):
        if phrase not in self.phrase2idx:
            self.phrase2idx[phrase] = self.idx
            self.idx2phrase[self.idx] = phrase
            self.idx += 1

    def add_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2contractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2contractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_contradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)


    def add_global_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2GlobalContractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2GlobalContractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_GlobalContradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)

    def __call__(self, phrase):
        if phrase not in self.phrase2idx:
            return self.phrase2idx['<unk>']
        return self.phrase2idx[phrase]

    def __len__(self):
        return len(self.phrase2idx)

class Concept_phrase(object):
    """ concept wrapper"""

    def __init__(self):
        self.phrase2idx = {}
        self.idx2phrase = {}
        self.idx2contractIdx = {}
        self.phrase2contractphrase = {}
        self.idx2GlobalContractIdx = {}
        self.phrase2GlobalContractphrase = {}
        self.idx = 0
        self.num_contradict_paris = 0
        self.num_GlobalContradict_paris = 0

    def add_phrase(self, phrase):
        if phrase not in self.phrase2idx:
            self.phrase2idx[phrase] = self.idx
            self.idx2phrase[self.idx] = phrase
            self.idx += 1

    def add_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2contractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2contractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_contradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)


    def add_global_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2GlobalContractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2GlobalContractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_GlobalContradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)

    def __call__(self, phrase):
        if phrase not in self.phrase2idx:
            return self.phrase2idx['<unk>']
        return self.phrase2idx[phrase]

    def __len__(self):
        return len(self.phrase2idx)

def main(option):
    rootpath = option.rootpath
    collection = option.collection
    threshold = option.threshold
    text_style = option.text_style

    vocab_file = os.path.join(rootpath, collection, 'TextData', 'vocabulary',
            text_style, 'word_vocab_%d.pkl'%threshold)
    counter_file = os.path.join(os.path.dirname(vocab_file), 'word_vocab_counter_%s.txt'%threshold)

    if checkToSkip(vocab_file, option.overwrite):
        sys.exit(0)
    makedirsforfile(vocab_file)


    vocab, word_counter = build_vocab(collection, text_style, threshold=threshold, rootpath=rootpath)
    with open(vocab_file, 'wb') as writer:
        pickle.dump(vocab, writer, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to %s", vocab_file)

    word_counter = [(word, cnt) for word, cnt in word_counter.items() if cnt >= threshold]
    word_counter.sort(key=lambda x: x[1], reverse=True)
    with open(counter_file, 'w') as writer:
        writer.write('\n'.join(map(lambda x: x[0]+' %d'%x[1], word_counter)))
    print("Saved vocabulary counter file to %s", counter_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='root path. (default: %s)'%ROOT_PATH)
    parser.add_argument('collection', type=str, help='collection tgif|msrvtt10k')
    parser.add_argument('--threshold', type=int, default=5, help='threshold to build vocabulary. (default: 5)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--text_style', type=str, choices=['rnn', 'bow'], default='bow',
                        help='text style for vocabulary. (default: bow)')
    opt = parser.parse_args()
    print(json.dumps(vars(opt), indent = 2))

    main(opt)

