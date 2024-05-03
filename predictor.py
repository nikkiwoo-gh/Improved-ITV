from __future__ import print_function
import pickle
import time
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.backends.cudnn.enabled = False

import sys

sys.path.append('./util')
from util.vocab import Concept_phrase
import torch
from torch.autograd import Variable
import evaluation
from model import Improved_ITV
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder
import logging
import json
import numpy as np
import argparse
from basic.util import read_dict
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.generic_utils import Progbar


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=10, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.match.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt',
                        help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18.')
    parser.add_argument('--query_num_all', type=int, default=90,
                        help='number of querys for test.')
    parser.add_argument('--query_sigmoid_threshold', type=float, default=0.99,
                        help='threshold for concept selection.')
    parser.add_argument('--concept_selection', type=str, default=None,help='way for concept selection')
    args = parser.parse_args()
    return args


def encode_data(model, data_loader, return_ids=True,sigmoid=True,dim=11147):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embeddings = None
    sigmoid_outs = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        if sigmoid:
            emb,sigmoid_out = model(datas,sigmoid_output=sigmoid)
            prob_mask = sigmoid_out<=0.5
            sigmoid_out[prob_mask]=0
        else:
            emb = model(datas,sigmoid_output=sigmoid)
        # initialize the numpy arrays given the size of the embeddings
        if embeddings is None:
            embeddings = np.zeros((len(data_loader.dataset), emb.size(1)))
            sigmoid_outs = np.zeros((len(data_loader.dataset),dim))
        # preserve the embeddings by copying from gpu and converting to numpy
        embeddings[idxs] = emb.data.cpu().numpy().copy()
        if sigmoid:
            sigmoid_outs[idxs] = sigmoid_out.data.cpu().numpy().copy()
        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if sigmoid:
        return embeddings, sigmoid_outs,ids
    else:
        if return_ids == True:
            return embeddings, ids,
        else:
            return embeddings

def compute_distances(model, data_loader,query_embs,bert_sim_concept_vectors,iw2v_concept_vectors,w2v_concept_vectors,nonUL_concept_vectors_ori_all,UL_concept_vectors_combined_all,return_ids=True,sigmoid=True,dim=11147):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embedding_matrix = None
    bert_sim_concept_matrix_all = None
    iw2v_concept_matrix_all = None
    w2v_sim_concept_matrix_all = None
    nonUL_concept_decoded_matrix_all = None
    UL_concept_decoded_combined_matrix_all = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))

    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        if sigmoid:
            emb,sigmoid_out = model(datas)
            prob_mask = sigmoid_out<=0.5
            sigmoid_out[prob_mask]=0
        else:
            emb = model(datas)
        # initialize the numpy arrays given the size of the embeddings
        if embedding_matrix is None:
            embedding_matrix = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
            bert_sim_concept_matrix_all = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
            iw2v_concept_matrix_all = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
            w2v_sim_concept_matrix_all = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
            nonUL_concept_decoded_matrix_all = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
            UL_concept_decoded_combined_matrix_all = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
        # preserve the embeddings by copying from gpu and converting to numpy
        embedding_matrix[:,idxs] = query_embs.dot(emb.data.cpu().numpy().copy().T)

        if sigmoid:
            bert_sim_concept_matrix_all[:, idxs] = bert_sim_concept_vectors.dot(sigmoid_out.data.cpu().numpy().copy().T)
            iw2v_concept_matrix_all[:, idxs] = iw2v_concept_vectors.dot(sigmoid_out.data.cpu().numpy().copy().T)
            w2v_sim_concept_matrix_all[:, idxs] = w2v_concept_vectors.dot(sigmoid_out.data.cpu().numpy().copy().T)
            nonUL_concept_decoded_matrix_all[:, idxs] = nonUL_concept_vectors_ori_all.dot(sigmoid_out.data.cpu().numpy().copy().T)
            UL_concept_decoded_combined_matrix_all[:, idxs] = UL_concept_vectors_combined_all.dot(sigmoid_out.data.cpu().numpy().copy().T)
        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if sigmoid:
        return embedding_matrix, bert_sim_concept_matrix_all,iw2v_concept_matrix_all,w2v_sim_concept_matrix_all,nonUL_concept_decoded_matrix_all,UL_concept_decoded_combined_matrix_all,ids
    else:
        if return_ids == True:
            return embedding_matrix, ids,
        else:
            return embedding_matrix

def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    concept_selection = opt.concept_selection
    # encoder_resume_name = os.path.join(opt.encoder_resume_name, opt.checkpoint_name)
    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    matching_best_rsum = checkpoint['matching_best_rsum']
    classification_best_rsum = checkpoint['classification_best_rsum']

    print("=> loaded checkpoint '{}' (epoch {}, matching_best_rsum {},classification_best_rsum {})"
          .format(resume, start_epoch, matching_best_rsum, classification_best_rsum))

    options = checkpoint['opt']
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")
    if not hasattr(options, 'loss_type'):
        setattr(options, "loss_type", "favorBCEloss")

    if not hasattr(options, 'decoder_layers'):
        setattr(options, "decoder_layers", "0-2048")
    if not hasattr(options, 'ul_alpha'):
        setattr(options, "ul_alpha", 0.0)
    if not hasattr(options, "with_textual_mapping"):
        setattr(options, "with_textual_mapping", True)
    if not hasattr(options, 'global_UL'):
        setattr(options, "global_UL", False)
    if not hasattr(options, 'concept_phrase'):
        setattr(options, "concept_phrase", options.concept_phase)
        options.concept_phrase.idx2phrase = options.concept_phrase.idx2phase
        options.concept_phrase.phrase2idx = options.concept_phrase.phase2idx
        options.concept_phrase.phrase2contractphrase = options.concept_phrase.phase2contractphase
        del options.concept_phrase.idx2phase
        del options.concept_phrase.phase2idx
        del options.concept_phrase.phase2contractphase
        options.concept_bank = 'concept_phrase'
    if not hasattr(options, 'contradicted_matrix_sp'):
        setattr(options, "contradicted_matrix_sp", options.contradicted_matrix_local_sp)
        del  options.contradicted_matrix_local_sp
    model = Improved_ITV(options)

    model.load_state_dict(checkpoint['model'])

    model.vid_encoder.eval()
    model.text_encoder.eval()
    model.unify_decoder.eval()
    trainCollection = options.trainCollection
    valCollection = options.valCollection

    visual_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature))
    assert options.visual_feat_dim == visual_feat_file.ndims
    if 'motion_feature' in options:
        motion_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', options.motion_feature))
        assert options.motion_feat_dim == motion_feat_file.ndims

    ##load textual feature
    textual_feat =BigFile(os.path.join(rootpath, testCollection, 'FeatureData', options.textual_feature))
    opt.textual_feat_dim = textual_feat.ndims

    video2frames = read_dict(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature,'video2frames.txt'))

    # set concept phase list for multi-label classification
    concept_phrase=model.concept_phrase
    concept2vec = get_text_encoder('bow')(concept_phrase, istimes=0)
    options.concept_list_size = len(concept_phrase)



    ##word+phrase
    concept_file = os.path.join(opt.rootpath, options.trainCollection, 'TextData', 'concept_phrase',options.concept_bank+'_frequency_count_gt'+str(options.concept_fre_threshold)+'.txt')

    ##add contraction pairs
    contradiction_file = concept_file+'.contradict.contradict_pairs'
    # contradiction_file = concept_file+'.wordNetAndVerbOcean.contradict.contradict_pairs'

    # contradiction_file = os.path.join(rootpath, trainCollection, 'TextData', 'concept',
    #                             'concept_frequency_count_gt' + str(
    #                                       options.concept_fre_threshold)+'.lemma.stanfordnlp.enhanced.txt.contradict.contradict_pairs')

    with open(contradiction_file, 'r') as reader:
        lines = reader.readlines()

    for line in lines:
        if line.find('//') < 0:
            concept_phrase.add_contradict(line)

    if 'motion_feature' in options:
        visual_loader = data.get_Improved_ITV_vid_data_loader(visual_feat_file,motion_feat_file, opt.batch_size, opt.workers, video2frames)
    else:
        visual_loader = data.get_vis_data_loader(visual_feat_file, opt.batch_size, opt.workers, video2frames)

    modelname = opt.logger_name[opt.logger_name.index('run'):]
    query_num = opt.query_num_all
    concept_dim = len(concept_phrase)
    thetas = [0.0,0.3,0.5,1.0]
    output_dir = resume.replace(trainCollection, testCollection)
    query_sets = []
    queryset2queryidxs = {}
    queryidxstart = 0
    query_embs_all = np.zeros([query_num,options.visual_mapping_layers[1]])
    bert_sim_concept_vectors_all= np.zeros([query_num,options.concept_list_size])
    iw2v_concept_vectors_all = np.zeros([query_num, options.concept_list_size])
    w2vsim_query_concept_vectors_all = np.zeros([query_num, options.concept_list_size])
    nonUL_query_decoded_combined_concept_vectors_all= np.zeros([query_num, options.concept_list_size])
    UL_query_decoded_combined_concept_vectors_all= np.zeros([query_num, options.concept_list_size])

    query_ids_all = []
    query_sigmoid_threshold = opt.query_sigmoid_threshold
    for query_set in opt.query_sets.strip().split(','):
        narrative_flag = False

        query_sets.append(query_set)
        output_dir_tmp = output_dir.replace(testCollection, '%s/results/%s/%s/%s' % (testCollection,query_set, trainCollection, valCollection),1)
        pred_result_file = os.path.join(output_dir_tmp, 'id.sent')

        print(pred_result_file)
        if checkToSkip(pred_result_file, opt.overwrite):
            continue
        try:
            makedirsforfile(pred_result_file)
        except Exception as e:
            print(e)

        # data loader prepare

        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)
        if query_file.find('narrative')>-1:
            narrative_flag=True
        # set data loader
        query_loader = data.get_Improved_ITV_txt_data_loader(query_file,textual_feat, opt.batch_size, opt.workers,isNarrative=narrative_flag)

        query_selections=None
        print("matched concepts are from file:"+query_file+'\n')
        ##way1 direct match for concept selection
        # concept_vectors, query_ids2,query_selections = evaluation.get_concept_vector(query_file, concept2vec)
        ##way2 similarity match for concept selection
        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)
        bert_sim_concept_vectors, query_ids2_bert_sim,query_selections_bert_sim = evaluation.get_concept_vector_BySim(query_file, concept2vec=concept2vec)
        savename = query_file + '.query_selection.bert_sim.lemma'
        lines=[]
        for i, iquery_id in enumerate(query_ids2_bert_sim):
            lines.append("#%s:%s\n" % (iquery_id, query_selections_bert_sim[i]))
            print("#%s:%s" % (iquery_id, query_selections_bert_sim[i]))
        with open(savename, 'w') as writer:
            writer.writelines(lines)
            print('save in %s\n' % savename)

        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)
        # ori_concept_vectors, ori_query_ids2, ori_query_selections = evaluation.get_concept_vector(query_file,concept2vec)


        # w2v_token_top1_query_file = query_file + '.w2v_token_top1.concept_selection'
        # w2v_token_top1_concept_vectors, query_ids2, query_selections = evaluation.get_concept_vector(w2v_token_top1_query_file, concept2vec,spliter=',')
        # print(w2v_token_top1_query_file)
        # if query_selections is not None:
        #     for i,iquery_id in enumerate(query_ids2):
        #         lines.append("#%s:%s\n"%(iquery_id,query_selections[i]))
        #         print("#%s:%s"%(iquery_id,query_selections[i]))

        # w2v_query_topk_query_file = query_file + '.w2v_query_topk(query_length).concept_selection'
        # w2v_query_topk_concept_vectors, query_ids2_w2v, w2v_query_selections = evaluation.get_concept_vector(w2v_query_topk_query_file, concept2vec,spliter=',')
        # print(w2v_query_topk_query_file)
        # for i,iquery_id in enumerate(query_ids2_w2v):
        #     lines.append("#%s:%s\n"%(iquery_id,w2v_query_selections[i]))
        #     print("#%s:%s"%(iquery_id,w2v_query_selections[i]))
        #
        # i2wv_sim_threshold = 0.8
        # i_w2v_query_file = query_file + '.i-w2v_threshold%f.concept_selection' % (i2wv_sim_threshold)
        # i_w2v_concept_vectors, i_w2v_query_ids2, i_w2v_query_selections = evaluation.get_concept_vector(i_w2v_query_file, concept2vec,spliter=',')
        # print(i_w2v_query_file)
        # for i,iquery_id in enumerate(i_w2v_query_ids2):
        #     lines.append("#%s:%s\n"%(iquery_id,i_w2v_query_selections[i]))
        #     print("#%s:%s"%(iquery_id,i_w2v_query_selections[i]))

        # #way3 extraction from the saved similarity concept selection
        if concept_selection is not None:
            nonUL_concept_file = os.path.join(rootpath, testCollection, 'TextData',query_set+'.decoded_egt%fAndContraryConcept.lemma.enhanced.%s'%(query_sigmoid_threshold,concept_selection))
            nonUL_concept_vectors, nonUL_query_ids2,nonUL_query_selections = evaluation.get_concept_vector(nonUL_concept_file, concept2vec,spliter=',')

            ##print concept selection:
            lines = []
            for i,iquery_id in enumerate(nonUL_query_ids2):
                lines.append("#%s:%s\n"%(iquery_id,nonUL_query_selections[i]))
                print("#%s:%s"%(iquery_id,nonUL_query_selections[i]))


        start = time.time()
        query_concept_sigmoid=None


        # query_embs, query_ids = encode_data(model.embed_txt, query_loader,sigmoid=False,dim=concept_dim)
        query_embs, query_concept_sigmoid,query_ids = encode_data(model.embed_txt, query_loader,sigmoid=True,dim=concept_dim)
        print("encode text time: %.3f s" % (time.time() - start))
        query_concept_sigmoid_new = np.zeros([len(query_ids),query_concept_sigmoid.shape[1]])
        query_concept_sigmoid_combined_new = np.zeros([len(query_ids),query_concept_sigmoid.shape[1]])
        # query_concept_sigmoid_minusContrary = np.zeros([len(query_ids),query_concept_sigmoid.shape[1]])
        if query_concept_sigmoid is not None:
            lines = []
            for i,iquery_id in enumerate(query_ids):
                contrary_words=[]
                query_sigmoid = query_concept_sigmoid[i,:]
                query_concept_decoding =[concept2vec.vocab.idx2phrase[idx] for idx in np.where(query_sigmoid >= query_sigmoid_threshold)[0]]
                concept_mapping =[]
                for idx in np.where(query_sigmoid >= query_sigmoid_threshold)[0]:
                    word = concept_phrase.idx2phrase[idx]
                    # if word=="video" or word=="clip"or word=="show"or word=="something"or word=="someone":
                    if word=="video" or word=="clip":
                        continue
                    concept_mapping.append(word+':%.3f'%query_sigmoid[idx])
                    query_concept_sigmoid_new[i,idx] =query_sigmoid[idx]
                    query_concept_sigmoid_combined_new[i,idx] =query_sigmoid[idx]
                    # query_concept_sigmoid_minusContrary[i,idx] =query_sigmoid[idx]
                ##union with similarity search
                for word in query_selections_bert_sim[i].split(','):
                    if word in concept_phrase.phrase2contractphrase.keys():
                        for icontrary in concept_phrase.phrase2contractphrase[word].split(','):
                            if not icontrary in query_selections_bert_sim[i].split(','):
                                contrary_words = contrary_words + [icontrary]
                    if len(word)>0 and (word not in query_concept_decoding):
                        phraseidx = concept2vec.vocab.phrase2idx[word]
                        concept_mapping=concept_mapping+[concept2vec.vocab.idx2phrase[phraseidx]+':%.3f'%query_sigmoid[phraseidx]]
                        query_concept_sigmoid_combined_new[i, phraseidx] = query_sigmoid[phraseidx]
                        # query_concept_sigmoid_minusContrary[i, phraseidx] = query_sigmoid[phraseidx]
                contrary_words = list(set(contrary_words))
                for contrary_word in contrary_words:
                    if contrary_word not in query_selections_bert_sim[i].split(','):
                        if contrary_word in concept_phrase.phrase2idx:
                            contrary_idx = concept_phrase.phrase2idx[contrary_word]
                            contrary_prob = query_sigmoid[contrary_idx]
                            # query_sigmoidy_concept_sigmoid_minusContrary[i, contrary_idx] = contrary_prob-1.0
                            concept_mapping = concept_mapping + ['--%s:%.3f' % (contrary_word, contrary_prob)]
                print("#%s:%s" % (iquery_id, ','.join(concept_mapping)))
                lines.append(iquery_id+' '+','.join(concept_mapping)+'\n')
            # savename=query_file+'.decoded_egt%fAndContraryConcept.lemma.enhanced.%s'%(query_sigmoid_threshold,modelname[0:10])
            # savename=query_file+'.decoded_concepts_egt%fAndContraryConcept.lemma.enhanced.ori.%s'%(query_sigmoid_threshold,modelname)
            savename=query_file+'.decoded_concepts.%s'%(modelname)
            with open(savename,'w') as writer:
                writer.writelines(lines)
                print('save in %s\n'%savename)

        # query_concept_sigmoid[query_concept_sigmoid<query_sigmoid_threshold]=0


        queryidxs =  np.arange(queryidxstart, queryidxstart + len(query_ids))
        queryset2queryidxs[query_set] =queryidxs
        query_embs_all[queryidxs,:]=query_embs
        # bert_sim_concept_vectors_all[queryidxs,:]=bert_sim_concept_vectors
        # iw2v_concept_vectors_all[queryidxs, :] = i_w2v_concept_vectors
        # w2vsim_query_concept_vectors_all[queryidxs,:]=w2v_query_topk_concept_vectors
        if concept_selection is not None:
            nonUL_query_decoded_combined_concept_vectors_all[queryidxs,:]=nonUL_concept_vectors
        UL_query_decoded_combined_concept_vectors_all[queryidxs,:]=query_concept_sigmoid_combined_new

        queryidxstart=queryidxstart+len(query_ids)
        query_ids_all = query_ids_all+query_ids
        ##make sure query_ids and query_ids2 are the same
    embedding_matrix_all = None
    start = time.time()
    if embedding_matrix_all is None:
        embedding_matrix_all, bert_sim_concept_matrix_all,iw2v_concept_matrix_all,w2v_sim_concept_matrix_all,nonUL_concept_decoded_matrix_all,UL_concept_decoded_combined_matrix_all,vis_ids = compute_distances(model.embed_vis, visual_loader,
                                                                              query_embs_all, bert_sim_concept_vectors_all,iw2v_concept_vectors_all,w2vsim_query_concept_vectors_all,nonUL_query_decoded_combined_concept_vectors_all,
                                                                                                         UL_query_decoded_combined_concept_vectors_all,
                                                                              sigmoid=True, dim=concept_dim)
        print("encode image time: %.3f s" % (time.time() - start))

    for query_set in query_sets:
        output_dir_tmp = output_dir.replace(testCollection, '%s/results/%s/%s/%s' % (testCollection,query_set, trainCollection, valCollection),1)
        query_idx  =[]
        for i,sample in enumerate(queryset2queryidxs[query_set]):
            query_idx.append(int(sample))
        query_ids = np.array(query_ids_all)[query_idx]
        query_ids = [item.replace('_narrative','') for item in query_ids]
        print(query_ids)
        embedding_matrix = embedding_matrix_all[query_idx,:]
        nanidx = np.isnan(embedding_matrix)
        embedding_matrix[nanidx] = 0
        rows_min = np.min(embedding_matrix, 1)[:, np.newaxis]
        rows_max = np.max(embedding_matrix, 1)[:, np.newaxis]
        print('embedding matrix min:%.2f, max:%.2f\n'%(np.min(rows_min),np.max(rows_max)))

        embedding_matrix_norm = (embedding_matrix - rows_min) / ((rows_max - rows_min))

        del embedding_matrix
        # #bert concept sim
        # bert_sim_concept_matrix= bert_sim_concept_matrix_all[query_idx,:]
        # nanidx = np.isnan(bert_sim_concept_matrix)
        # bert_sim_concept_matrix[nanidx] = 0
        # rows_min = np.min(bert_sim_concept_matrix, 1)[:, np.newaxis]
        # rows_max = np.max(bert_sim_concept_matrix, 1)[:, np.newaxis]
        # print('concept matrix min:%.2f, max:%.2f\n'%(np.min(rows_min),np.max(rows_max)))
        # bert_sim_concept_matrix_norm = (bert_sim_concept_matrix - rows_min) / ((rows_max - rows_min))
        # del bert_sim_concept_matrix
        # print("mapping concept time: %.3f s" % (time.time() - start))
        #
        # #iw2v concept  sim
        # iw2v_sim_concept_matrix= iw2v_concept_matrix_all[query_idx,:]
        # nanidx = np.isnan(iw2v_sim_concept_matrix)
        # iw2v_sim_concept_matrix[nanidx] = 0
        # rows_min = np.min(iw2v_sim_concept_matrix, 1)[:, np.newaxis]
        # rows_max = np.max(iw2v_sim_concept_matrix, 1)[:, np.newaxis]
        # print('concept matrix min:%.2f, max:%.2f\n'%(np.min(rows_min),np.max(rows_max)))
        # iw2v_sim_concept_matrix_norm = (iw2v_sim_concept_matrix - rows_min) / ((rows_max - rows_min))
        # del iw2v_sim_concept_matrix
        # print("mapping concept time: %.3f s" % (time.time() - start))
        #
        # #w2v concept sim
        # w2v_sim_concept_matrix= w2v_sim_concept_matrix_all[query_idx,:]
        # nanidx = np.isnan(w2v_sim_concept_matrix)
        # w2v_sim_concept_matrix[nanidx] = 0
        # rows_min = np.min(w2v_sim_concept_matrix, 1)[:, np.newaxis]
        # rows_max = np.max(w2v_sim_concept_matrix, 1)[:, np.newaxis]
        # print('concept matrix min:%.2f, max:%.2f\n'%(np.min(rows_min),np.max(rows_max)))
        # w2v_sim_concept_matrix_norm = (w2v_sim_concept_matrix - rows_min) / ((rows_max - rows_min))
        # del w2v_sim_concept_matrix
        # print("mapping concept time: %.3f s" % (time.time() - start))
        #
        # #nonUL concept sim
        # nonUL_concept_matrix= nonUL_concept_decoded_matrix_all[query_idx,:]
        # nanidx = np.isnan(nonUL_concept_matrix)
        # nonUL_concept_matrix[nanidx] = 0
        # rows_min = np.min(nonUL_concept_matrix, 1)[:, np.newaxis]
        # rows_max = np.max(nonUL_concept_matrix, 1)[:, np.newaxis]
        # print('concept matrix min:%.2f, max:%.2f\n'%(np.min(rows_min),np.max(rows_max)))
        # nonUL_concept_matrix_norm = (nonUL_concept_matrix - rows_min) / ((rows_max - rows_min))
        # del nonUL_concept_matrix
        # print("mapping concept time: %.3f s" % (time.time() - start))

        #UL concept sim
        UL_concept_decoded_matrix= UL_concept_decoded_combined_matrix_all[query_idx,:]
        nanidx = np.isnan(UL_concept_decoded_matrix)
        UL_concept_decoded_matrix[nanidx] = 0
        rows_min = np.min(UL_concept_decoded_matrix, 1)[:, np.newaxis]
        rows_max = np.max(UL_concept_decoded_matrix, 1)[:, np.newaxis]
        print('concept matrix min:%.2f, max:%.2f\n'%(np.min(rows_min),np.max(rows_max)))
        UL_concept_decoded_matrix_norm = (UL_concept_decoded_matrix - rows_min) / ((rows_max - rows_min))
        del UL_concept_decoded_matrix
        print("mapping concept time: %.3f s" % (time.time() - start))


        for theta in thetas:
            pred_result_file = os.path.join(output_dir_tmp, 'id.sent.sim.%.2f.combinedDecodedConcept_theta'%(query_sigmoid_threshold) + str(theta).replace('.', '_') + '_score')
            print(pred_result_file)
            combined_matrix = (1 - theta) * embedding_matrix_norm + (theta) * UL_concept_decoded_matrix_norm
            combined_inds = np.argsort(combined_matrix, axis=1)
            with open(pred_result_file, 'w') as fout:
                for index in range(combined_inds.shape[0]):
                    ind = combined_inds[index][::-1]
                    fout.write(query_ids[index] + ' ' + ' '.join(
                        [vis_ids[i] + ' %s' % combined_matrix[index][i] for i in ind]) + '\n')


        # for theta in [1.0]:
        #     BERTsim_pred_result_file = os.path.join(output_dir_tmp, 'id.sent.BERTsim.combined_theta' + str(theta).replace('.', '_') + '_score')
        #     iw2v_pred_result_file = os.path.join(output_dir_tmp, 'id.sent.iw2vsim.combined_theta' + str(theta).replace('.', '_') + '_score')
        #     w2v_pred_result_file = os.path.join(output_dir_tmp, 'id.sent.W2Vsim.combined_theta' + str(theta).replace('.', '_') + '_score')
        #     nonUL_pred_result_file = os.path.join(output_dir_tmp, 'id.sent.nonUL.combined_theta' + str(theta).replace('.', '_') + '_score')
        #
        #     ##bert sim output
        #     combined_matrix = (1 - theta) * embedding_matrix_norm + (theta) * bert_sim_concept_matrix_norm
        #     combined_inds = np.argsort(combined_matrix, axis=1)
        #     with open(BERTsim_pred_result_file, 'w') as fout:
        #         for index in range(combined_inds.shape[0]):
        #             ind = combined_inds[index][::-1]
        #             fout.write(query_ids[index] + ' ' + ' '.join(
        #                 [vis_ids[i] + ' %s' % combined_matrix[index][i] for i in ind]) + '\n')
        #     print(BERTsim_pred_result_file)
        #
        #     ##iw2v output
        #     combined_matrix = (1 - theta) * embedding_matrix_norm + (theta) * iw2v_sim_concept_matrix_norm
        #     combined_inds = np.argsort(combined_matrix, axis=1)
        #     with open(iw2v_pred_result_file, 'w') as fout:
        #         for index in range(combined_inds.shape[0]):
        #             ind = combined_inds[index][::-1]
        #             fout.write(query_ids[index] + ' ' + ' '.join(
        #                 [vis_ids[i] + ' %s' % combined_matrix[index][i] for i in ind]) + '\n')
        #     print(iw2v_pred_result_file)
        #
        #     ##w2v sim output
        #     combined_matrix = (1 - theta) * embedding_matrix_norm + (theta) * w2v_sim_concept_matrix_norm
        #     combined_inds = np.argsort(combined_matrix, axis=1)
        #     with open(w2v_pred_result_file, 'w') as fout:
        #         for index in range(combined_inds.shape[0]):
        #             ind = combined_inds[index][::-1]
        #             fout.write(query_ids[index] + ' ' + ' '.join(
        #                 [vis_ids[i] + ' %s' % combined_matrix[index][i] for i in ind]) + '\n')
        #     print(w2v_pred_result_file)
        #
        #     ##nonUL output
        #     combined_matrix = (1 - theta) * embedding_matrix_norm + (theta) * nonUL_concept_matrix_norm
        #     combined_inds = np.argsort(combined_matrix, axis=1)
        #     with open(nonUL_pred_result_file, 'w') as fout:
        #         for index in range(combined_inds.shape[0]):
        #             ind = combined_inds[index][::-1]
        #             fout.write(query_ids[index] + ' ' + ' '.join(
        #                 [vis_ids[i] + ' %s' % combined_matrix[index][i] for i in ind]) + '\n')
        #     print(nonUL_pred_result_file)



if __name__ == '__main__':
    main()
