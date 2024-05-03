import logging
import torch
import numpy as np
import time
import wandb

from scipy.spatial import distance
from util.metric import getScorer
from util.utils import AverageMeter, LogCollector,Progbar
from sentence_transformers import SentenceTransformer,util


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm

def cal_error(videos, captions, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = -1*np.dot(captions, videos.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    return errors

def v2t(c2v, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2v: (5N, N) matrix of caption to video errors
    """
    assert c2v.shape[0] / c2v.shape[1] == n_caption, c2v.shape
    ranks = np.zeros(c2v.shape[1])

    for i in range(len(ranks)):
        d_i = c2v[:, i]
        inds = np.argsort(d_i)

        rank = np.where((inds/n_caption) == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])

def t2v(c2v,  n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2v: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2v.shape)
    assert c2v.shape[0] / c2v.shape[1] == n_caption, c2v.shape
    ranks = np.zeros(c2v.shape[0])

    for i in range(len(ranks)):
        d_i = c2v[i]
        inds = np.argsort(d_i)

        rank = np.where(inds == int(i/n_caption))[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return map(float, [r1, r5, r10, medr, meanr])

# mAP for Text-to-Video Retrieval
def t2v_map(c2v, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2v: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2v.shape)
    assert c2v.shape[0] / c2v.shape[1] == n_caption, c2v.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2v.shape[0]):
        d_i = c2v[i, :]
        labels = [0]*len(d_i)
        labels[int(i/n_caption)] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


# mAP for Video-to-Text Retrieval
def v2t_map(c2v, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2v: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2v.shape)
    assert c2v.shape[0] / c2v.shape[1] == n_caption, c2v.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2v.shape[1]):
        d_i = c2v[:, i]
        labels = [0]*len(d_i)
        labels[i*n_caption:(i+1)*n_caption] = [1]*n_caption

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


def encode_data(model, data_loader, log_step=10, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    end = time.time()

    # numpy array to keep all the data
    video_embs = None
    cap_embs = None
    concept_vectors = None
    class_vid_outs  = None
    class_text_outs = None
    video_ids = ['']*len(data_loader.dataset)
    caption_ids = ['']*len(data_loader.dataset)
    captions_ori = [''] * len(data_loader.dataset)
    for i, (videos, captions,concept_bows,caption_ori, idxs, cap_ids, vid_ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings and interpretations
        vid_emb, cap_emb = model.forward_matching(videos, captions, True)
        class_vid_out,class_text_out = model.forward_classification(vid_emb,cap_emb, True)
        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            concept_vectors = np.zeros((len(data_loader.dataset), concept_bows.size(1)))
            class_vid_outs  = np.zeros((len(data_loader.dataset), concept_bows.size(1)))
            class_text_outs = np.zeros((len(data_loader.dataset), concept_bows.size(1)))
        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        cap_embs[idxs] = cap_emb.data.cpu().numpy().copy()
        concept_vectors[idxs] = concept_bows.data.cpu().numpy().copy()
        class_vid_outs[idxs] = class_vid_out.data.cpu().numpy().copy()
        class_text_outs[idxs] = class_text_out.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            caption_ids[idx] = cap_ids[j]
            video_ids[idx] = vid_ids[j]
            captions_ori[idx] = caption_ori[j]
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        del videos, captions

    if return_ids == True:
        return video_embs, cap_embs,class_vid_outs,class_text_outs, concept_vectors,captions_ori,video_ids, caption_ids
    else:
        return video_embs, cap_embs,class_vid_outs,class_text_outs,concept_vectors,captions_ori

def eval_multi_label_classifiction_perf(outs,labels,topn=10):
    recall = []
    pred_class_num = []
    gt_num = []
    match_num_sum = []
    for i in range(outs.shape[0]):
        topn_pred_class = np.zeros([outs.shape[1]])
        outs_i = np.squeeze(outs[i, :])
        label_i = np.squeeze(labels[i, :])
        if np.isnan(label_i.sum()):
            label_i = np.zeros([len(label_i)])

        ##get the performance of topn
        rankidx = np.argsort(outs_i)[::-1]
        rankidx = rankidx[0:topn]

        ##get the number of predicted class
        outs_pred_class_index = np.where(outs_i > 0.5)[0]
        outs_pred_num = outs_pred_class_index.size
        pred_class_num.append(outs_pred_num)
        i_gt_number = np.sum(label_i)
        if np.isnan(i_gt_number):
            i_gt_number = 0
        gt_num.append(i_gt_number)


        recallAtk_i = 0.0
        if i_gt_number==0:
            print('i_gt_number==0')

        topn_pred_class[rankidx] = 1
        match_candidate = np.multiply(topn_pred_class, label_i)
        if len(np.where(match_candidate > 0))>0:
            match_index = np.where(match_candidate > 0)[0]
            match_num = len(match_index)*1.0
        else:
            match_num =0


        if (match_num > 0):
            if i_gt_number>topn:
                 recallAtk_i = match_num*1.0 / topn
            else:
                recallAtk_i = match_num * 1.0 / i_gt_number

        match_num_sum.append(match_num)
        recall.append(recallAtk_i)


    return match_num_sum,recall,gt_num,pred_class_num


def eval_ITV(opt, val_loader, model, concept2vec,measure='cosine',topn=10,return_perf=True,capid2conceptvecs=None):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # compute the encoding for all the validation video and captions


    video_embs, cap_embs, class_vid_outs_ori,class_text_outs_ori,concept_vectors_ori,captions_list_ori,video_ids, caption_ids = encode_data(model, val_loader, opt.log_step,
                                                                                    logging.info)

    ##first evaluation encoder
    # we load data as video-sentence pairs
    # but we only need to forward each video once for evaluation
    # so we get the video set and mask out same videos with feature_mask
    feature_mask = []
    evaluate_videos = set()
    for video_id in video_ids:
        feature_mask.append(video_id not in evaluate_videos)
        evaluate_videos.add(video_id)
    video_embs = video_embs[feature_mask]
    class_vid_outs = class_vid_outs_ori[feature_mask]
    concept_vectors = concept_vectors_ori[feature_mask]
    captions_list = list(np.array(captions_list_ori)[feature_mask])
    video_ids = [x for idx, x in enumerate(video_ids) if feature_mask[idx] is True]
    class_text_outs_filter = class_text_outs_ori
    class_text_outs_filter[class_text_outs_filter<=0.99]=0
    class_vid_outs_filter=class_vid_outs
    class_vid_outs_filter[class_vid_outs_filter<=0.5]=0

    sim_martix=cap_embs.dot(video_embs.T)

    c2v_all_errors_emb = cal_error(video_embs, cap_embs, measure)
    c2v_all_errors_concept = cal_error(class_vid_outs_filter, class_text_outs_filter, measure)



    ##first evaluation encoder
    if opt.val_metric == "recall" and (not opt.testCollection=='msvd'):

        # video retrieval
        (r1i, r5i, r10i, medri, meanri) = t2v(c2v_all_errors_emb, n_caption=opt.n_caption)
        print(" * Embedding matching")
        print(" * Text to video:")
        print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
        print(" * medr, meanr: {}".format([round(medri, 3), round(meanri, 3)]))
        print(" * " + '-' * 10)
        (c_r1i, c_r5i, c_r10i, c_medri, c_meanri) = t2v(c2v_all_errors_concept, n_caption=opt.n_caption)
        print(" * Concept matching")
        print(" * Text to video:")
        print(" * r_1_5_10: {}".format([round(c_r1i, 3), round(c_r5i, 3), round(c_r10i, 3)]))
        print(" * medr, meanr: {}".format([round(c_medri, 3), round(c_meanri, 3)]))
        print(" * " + '-' * 10)

        # caption retrieval
        (r1, r5, r10, medr, meanr) = v2t(c2v_all_errors_emb, n_caption=opt.n_caption)
        print(" * Embedding Matching")
        print(" * Video to text:")
        print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
        print(" * medr, meanr: {}".format([round(medr, 3), round(meanr, 3)]))
        print(" * " + '-' * 10)

        (c_r1, c_r5, c_r10, c_medr, c_meanr) = v2t(c2v_all_errors_concept, n_caption=opt.n_caption)
        print(" * Concept Matching")
        print(" * Video to text:")
        print(" * r_1_5_10: {}".format([round(c_r1, 3), round(c_r5, 3), round(c_r10, 3)]))
        print(" * medr, meanr: {}".format([round(c_medr, 3), round(c_meanr, 3)]))
        print(" * " + '-' * 10)
        # record metrics in wandb
        wandb.log({"val/r1": r1}, step=model.Eiters)
        wandb.log({"val/r5": r5}, step=model.Eiters)
        wandb.log({"val/r10": r10}, step=model.Eiters)
        wandb.log({"val/medr": medr}, step=model.Eiters)
        wandb.log({"val/meanr": meanr}, step=model.Eiters)
        wandb.log({"val/r1i": r1i}, step=model.Eiters)
        wandb.log({"val/r5i": r5i}, step=model.Eiters)
        wandb.log({"val/r10i": r10i}, step=model.Eiters)
        wandb.log({"val/medri": medri}, step=model.Eiters)
        wandb.log({"val/meanri": meanri}, step=model.Eiters)




    elif opt.val_metric == "map":
        v2t_map_score =v2t_map(c2v_all_errors_emb, n_caption=opt.n_caption)
        t2v_map_score = t2v_map(c2v_all_errors_emb, n_caption=opt.n_caption)
        con_v2t_map_score =v2t_map(c2v_all_errors_concept, n_caption=opt.n_caption)
        con_t2v_map_score = t2v_map(c2v_all_errors_concept, n_caption=opt.n_caption)
        print('embedding v2t_map', v2t_map_score)
        print('embedding t2v_map', t2v_map_score)
        print('concept v2t_map', con_v2t_map_score)
        print('concept t2v_map', con_t2v_map_score)
        wandb.log({"val/v2t_map": v2t_map_score}, step=model.Eiters)
        wandb.log({"val/t2v_map": t2v_map_score}, step=model.Eiters)


    encoder_currscore = 0
    if opt.val_metric == "recall" and (not opt.testCollection=='msvd'):
        if opt.direction == 'v2t' or opt.direction == 'all':
            encoder_currscore += (r1 + r5 + r10)
        if opt.direction == 't2v' or opt.direction == 'all':
            encoder_currscore += (r1i + r5i + r10i)
    elif opt.val_metric == "map":
        if opt.direction == 'v2t' or opt.direction == 'all':
            encoder_currscore += v2t_map_score
        if opt.direction == 't2v' or opt.direction == 'all':
            encoder_currscore += t2v_map_score

    concept_vectors_text_ori = concept_vectors_ori
    if capid2conceptvecs is not None:
        concept_vectors_text_ori= np.zeros([concept_vectors_ori.shape[0],concept_vectors_ori.shape[1]])
        prob = Progbar(len(caption_ids))
        for icap_idx,icap_id in enumerate(caption_ids):
            prob.add(1)
            concept_vectors_text_ori[icap_idx] = capid2conceptvecs[icap_id]
    match_num_sum_vid,recall_vid,gt_vid_num,decoded_concept_vid_num = eval_multi_label_classifiction_perf(class_vid_outs,concept_vectors,topn=topn)
    match_num_sum_text,recall_text,gt_text_num,decoded_concept_text_num = eval_multi_label_classifiction_perf(class_text_outs_ori,concept_vectors_text_ori,topn=topn)

    print(" * Video: multi-label classification:")
    print(" * average matchnum@10(vid,text): {},{}".format(np.average(match_num_sum_vid),np.average(match_num_sum_text)))
    print(" * average recall@10(vid,text):%.4f,%.4f"%(np.average(recall_vid),np.average(recall_text)))
    print(" * average #decoded_concepts(vid,text): {},{}".format(np.average(decoded_concept_vid_num),np.average(decoded_concept_text_num)))
    print(" * average gt_num(vid,text): {},{}".format(np.average(gt_vid_num),np.average(gt_text_num)))
    print(" * "+'-'*10)

    decoder_cur_vid_recall = np.average(recall_vid)
    decoder_cur_text_recall = np.average(recall_text)

    wandb.log({"val/#decoded_concepts_vid": np.average(decoded_concept_vid_num)}, step=model.Eiters)
    wandb.log({"val/#decoded_concepts_text": np.average(decoded_concept_text_num)}, step=model.Eiters)
    wandb.log({"val/decoder_vid_recall": np.average(recall_vid)}, step=model.Eiters)
    wandb.log({"val/decoder_text_recall": np.average(recall_text)}, step=model.Eiters)
    wandb.log({"val/rsum": encoder_currscore}, step=model.Eiters)



    if return_perf:
        return encoder_currscore,decoder_cur_vid_recall, decoder_cur_text_recall
    else:
        return captions_list,captions_list_ori, class_vid_outs,class_text_outs_ori, concept_vectors,concept_vectors_text_ori, sim_martix,video_ids,caption_ids

def get_concept_vector_BySim(query_file,concept2vec=None,concept_bank=None):
    """map all captions into concept vector
       concept selection: BERT embedding similarity
    """

    stop_word_file = 'stopwords_en.txt'
    stop_words = []
    with open(stop_word_file, 'rb') as reader:
        for word in reader:
            word = word.decode().strip()
            stop_words.append(word)

    stop_words.append('one')
    embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    if concept_bank is None:
        concept_bank=[item for item in concept2vec.vocab.phrase2idx.keys()]

    corpus_embeddings = embedder.encode(concept_bank, convert_to_tensor=True,show_progress_bar=False)

    captions = {}
    cap_ids = []
    with open(query_file, 'r', encoding='iso-8859-1') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            captions[cap_id] = caption
            cap_ids.append(cap_id)
    if concept2vec is not None:
        concept_vectors = np.zeros([len(cap_ids),concept2vec.ndims])
    query_selections = [""]*len(cap_ids)
    index = 0
    pbar = Progbar(len(cap_ids))
    for cap_id in cap_ids:
        caption = captions[cap_id]
        for cap in caption.split():
            if cap not in stop_words:
                query_embedding = embedder.encode(cap, convert_to_tensor=True,show_progress_bar=False)
                cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
                cos_scores = cos_scores.data.cpu().numpy().squeeze()
                top_results = np.argpartition(-cos_scores, range(5))[0:5]
                most_similar_idx = top_results[0]
                sim_score = cos_scores[most_similar_idx]
                if sim_score>0.9:
                    matched_concept=concept_bank[most_similar_idx]
                    # print("%s match to concept:%s"%(cap,matched_concept))
                    if concept2vec is not None:
                        concept_vectors[index,concept2vec.vocab.phrase2idx[matched_concept]]=1
                    query_selections[index]=query_selections[index]+','+matched_concept
        index = index+1
        pbar.add(1)
    if concept2vec is not None:
        return concept_vectors,cap_ids,query_selections
    else:
        return cap_ids, query_selections

