import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm  # clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from loss import TripletLoss, likelihoodBCEloss,unlikelihoodBCEloss
from util.bigfile import BigFile


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we))
    return np.array(we)

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features

class Text_one_layer_encoder(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Text_one_layer_encoder, self).__init__()
        self.text_norm = opt.text_norm
        self.dropout = nn.Dropout(p=opt.dropout)
        self.with_textual_mapping = opt.with_textual_mapping
        # multi fc layers
        if self.with_textual_mapping:
            self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)


    def forward(self, text, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows, cap_mask = x
        features = text


        # mapping to common space
        if self.with_textual_mapping:
            features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        if np.sum(np.isnan(features.data.cpu().numpy())) > 0:
            print('features is nan')

        return features

class Video_encoder(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Video_encoder, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.vconcate

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.visual_kernel_sizes
        ])

        # visual mapping
        self.visual_mapping = MFC(opt.visual_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

    def forward(self, videos):
        """Extract video feature vectors."""

        videos, motions,videos_origin, lengths, vidoes_mask = videos

        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1, -1, gru_init_out.size(2))  # (N,C,F1)
        gru_init_out = gru_init_out * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        ##level 4 motion feature (e.g., slowfast)
        motion_out = motions
        # concatenation
        if self.concate == 'full':  # level 1+2+3
            features = torch.cat((gru_out, con_out, org_out,motion_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out, con_out), 1)

        # mapping to common space
        features = self.visual_mapping(features)
        if self.visual_norm:
            features = l2norm(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_encoder, self).load_state_dict(new_state)


class Text_multi_level_encoder(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Text_multi_level_encoder, self).__init__()
        self.text_norm = opt.text_norm
        self.dropout = nn.Dropout(p=opt.dropout)
        self.tconcate = opt.tconcate

        # multi fc layers
        self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size * 2
        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.text_kernel_sizes
        ])
        self.init_weights()


    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, text, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows, cap_mask = x
        cap_wids, cap_bows, lengths, cap_mask = text

        org_out = cap_bows
        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
        gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        features = torch.cat((gru_out, con_out, org_out), 1)

        # mapping to common space
        features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        if np.sum(np.isnan(features.data.cpu().numpy())) > 0:
            print('features is nan')

        return features


class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoder.state_dict(), self.text_encoder.state_dict() ,self.unify_decoder.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])
        self.unify_decoder.load_state_dict(state_dict[2])

    def forward_loss(self, cap_emb, vid_emb ,pred_vid_class ,pred_text_class ,class_label ,*agrs, **kwargs):

        # 1. Compute the triplet loss given pairs of video and caption embeddings

        matching_loss = self.matching_loss(cap_emb, vid_emb)


        # 2. Compute the likelihood loss of concept decoding loss

        labels = Variable(class_label, requires_grad=False)  ##cap_bow may have value larger than 1
        if torch.cuda.is_available():
            labels = labels.cuda()
        likelihoodLoss_vid = self.likelihoodLoss(pred_vid_class ,labels)
        likelihoodLoss_text = self.likelihoodLoss(pred_text_class, labels)

        likelihoodLoss  = likelihoodLoss_vid +likelihoodLoss_text
        loss = matching_loss + likelihoodLoss
        # 3. Compute the unlikelihood loss of concept decoding loss

        if self.unlikelihood and not self.contradicted_matrix_sp is None :
            unlikelihoodLoss_vid = self.unlikelihoodLoss(pred_vid_class ,labels)
            unlikelihoodLoss_text = self.unlikelihoodLoss(pred_text_class ,labels)
            unlikelihoodLoss = unlikelihoodLoss_vid +unlikelihoodLoss_text

            loss = loss +self.ul_alpha *unlikelihoodLoss

        self.logger.update('Le', loss.item(), vid_emb.size(0))

        if self.unlikelihood and not self.contradicted_matrix_sp is None :
            return loss, matching_loss, likelihoodLoss,likelihoodLoss_vid, likelihoodLoss_text,unlikelihoodLoss,unlikelihoodLoss_vid,unlikelihoodLoss_text

        return loss, matching_loss, likelihoodLoss,likelihoodLoss_vid, likelihoodLoss_text



    def train_dualtask(self, videos, captions, class_label ,captions_text ,lengths, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb = self.forward_matching(videos, captions, False)
        pred_vid_class ,pred_text_class = self.forward_classification(vid_emb ,cap_emb ,False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()


        if self.unlikelihood:
            loss, loss_matching, likelihoodLoss,likelihoodLoss_vid, likelihoodLoss_text,unlikelihoodLoss, unlikelihoodLoss_vid,unlikelihoodLoss_text  = self.forward_loss \
                    (cap_emb, vid_emb, pred_vid_class ,pred_text_class, class_label)
        else:
            loss, loss_matching, likelihoodLoss,likelihoodLoss_vid, likelihoodLoss_text= self.forward_loss(cap_emb, vid_emb,pred_vid_class,pred_text_class,class_label)

        loss_value = loss.item()
        loss_matching_value = loss_matching.item()
        likelihoodLoss_value = likelihoodLoss.item()
        likelihoodloss_vid_value = likelihoodLoss_vid.item()
        likelihoodloss_text_value = likelihoodLoss_text.item()
        if self.unlikelihood:
            unlikelihoodLoss_value = unlikelihoodLoss.item()
            unlikelihoodLoss_vid_value = unlikelihoodLoss_vid.item()
            unlikelihoodLoss_text_value = unlikelihoodLoss_text.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        if self.unlikelihood:
            return vid_emb.size(0), loss_value, loss_matching_value, likelihoodLoss_value,likelihoodloss_vid_value ,likelihoodloss_text_value ,unlikelihoodLoss_value ,unlikelihoodLoss_vid_value ,unlikelihoodLoss_text_value
        else:
            return vid_emb.size(0), loss_value ,loss_matching_value,likelihoodLoss_value ,likelihoodloss_vid_value ,likelihoodloss_text_value


class Improved_ITV(BaseModel):
    """
    Improved ITV network
    """
    def __init__(self, opt):
        # Build Models
        self.modelname = opt.postfix
        self.grad_clip = opt.grad_clip
        self.vid_encoder = Video_encoder(opt)
        self.text_encoder = Text_one_layer_encoder(opt)
        self.decoder_num_layer=len(opt.decoder_mapping_layers)

        if len(opt.decoder_mapping_layers)==2:
            self.unify_decoder =MFC(opt.decoder_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)
        elif len(opt.decoder_mapping_layers)==3:
            mapping_layer1 = [opt.decoder_mapping_layers[0],opt.decoder_mapping_layers[1]]
            mapping_layer2 = [opt.decoder_mapping_layers[1],opt.decoder_mapping_layers[2]]
            self.unify_decoder=nn.ModuleList([MFC(mapping_layer1, opt.dropout, have_bn=False, have_last_bn=False),
                                              MFC(mapping_layer2, opt.dropout, have_bn=True,
                                                  have_last_bn=True)])

        else:
            NotImplemented
        self.sigmod = nn.Sigmoid()

        self.loss_type = opt.loss_type
        self.unlikelihood=opt.unlikelihood
        self.ul_alpha = opt.ul_alpha
        self.concept_phrase = opt.concept_phrase
        if self.unlikelihood:
            self.contradicted_matrix_sp = opt.contradicted_matrix_sp
        print(self.vid_encoder)
        print(self.text_encoder)
        print(self.unify_decoder)
        print(self.sigmod)
        if torch.cuda.is_available():
            self.vid_encoder.cuda()
            self.text_encoder.cuda()
            self.unify_decoder.cuda()
            cudnn.benchmark = True

        # Loss and Optimize
        self.matching_loss  = TripletLoss(margin=opt.margin,
                                              measure=opt.measure,
                                              max_violation=opt.max_violation,
                                              cost_style=opt.cost_style,
                                            direction=opt.direction)

        self.likelihoodLoss = likelihoodBCEloss(opt.multiclass_loss_lamda,opt.cost_style)

        if self.unlikelihood:
            self.unlikelihoodLoss = unlikelihoodBCEloss(self.contradicted_matrix_sp,opt.cost_style)

        params_end_text = list(self.text_encoder.parameters())
        params_end_vid = list(self.vid_encoder.parameters())
        params_unify_dec = list(self.unify_decoder.parameters())
        self.params_end_text = params_end_text
        self.params_end_vid = params_end_vid
        self.params_unify_dec=params_unify_dec
        params= params_end_text+params_end_vid+params_unify_dec
        self.params = params


        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0


    def forward_matching(self, videos, targets, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        frames,motions,mean_origin, video_lengths, vidoes_mask = videos
        frames = Variable(frames, requires_grad=True)
        if volatile:
            with torch.no_grad():
                frames = Variable(frames)
        if torch.cuda.is_available():
            frames = frames.cuda()

        motions = Variable(motions, requires_grad=True)
        if volatile:
            with torch.no_grad():
                motions = Variable(motions)
        if torch.cuda.is_available():
            motions = motions.cuda()

        mean_origin = Variable(mean_origin, requires_grad=True)
        if volatile:
            with torch.no_grad():
                mean_origin = Variable(mean_origin)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, requires_grad=True)
        if volatile:
            with torch.no_grad():
                vidoes_mask = Variable(vidoes_mask)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)

        # text data
        textual_feat = targets
        if textual_feat is not None:
            textual_feat = Variable(textual_feat)
            if volatile:
                with torch.no_grad():
                    textual_feat = Variable(textual_feat)
            if torch.cuda.is_available():
                textual_feat = textual_feat.cuda()

        text_data = textual_feat


        vid_emb = self.vid_encoder(videos_data)
        cap_emb = self.text_encoder(text_data)
        return vid_emb, cap_emb

    def forward_classification(self, vid_embs,text_embs, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        if self.decoder_num_layer>2:
            for decod in self.unify_decoder:
                vid_embs = decod(vid_embs)
                text_embs = decod(text_embs)
        else:
            text_embs=self.unify_decoder(text_embs)
            vid_embs = self.unify_decoder(vid_embs)
        pred_vid=self.sigmod(vid_embs)
        pred_text=self.sigmod(text_embs)


        return pred_vid,pred_text

    def embed_vis(self, vis_data, volatile=True,sigmoid_output=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data

        if volatile:
            with torch.no_grad():
                frames = Variable(frames)
        else:
            frames = Variable(frames, requires_grad=True)
        if torch.cuda.is_available():
            frames = frames.cuda()

        if volatile:
            with torch.no_grad():
                motions = Variable(motions)
        else:
            motions = Variable(motions, requires_grad=True)

        if torch.cuda.is_available():
            motions = motions.cuda()

        if volatile:
            with torch.no_grad():
                mean_origin = Variable(mean_origin)
        else:
            mean_origin = Variable(mean_origin, requires_grad=True)


        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if volatile:
            with torch.no_grad():
                vidoes_mask = Variable(vidoes_mask)
        else:
            vidoes_mask = Variable(vidoes_mask, requires_grad=True)

        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, motions,mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_encoder(vis_data)
        pred= self.vid_encoder(vis_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                pred = decod(pred)
        else:
            pred=self.unify_decoder(pred)
        sigmoid_out=self.sigmod(pred)

        if sigmoid_output:
            return embs,sigmoid_out
        else:
            return embs

    def embed_vis_emb_only(self, vis_data, volatile=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        motions = Variable(motions, volatile=volatile)
        if torch.cuda.is_available():
            motions = motions.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_encoder(vis_data)
        return embs

    def embed_vis_concept_only(self, vis_data, volatile=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data
        if volatile:
            with torch.no_grad:
                frames = Variable(frames)
        else:
            frames = Variable(frames, requires_grad=True)
        if torch.cuda.is_available():
            frames = frames.cuda()

        if volatile:
            with torch.no_grad:
                motions = Variable(motions)
        else:
            motions = Variable(motions, requires_grad=True)
        if torch.cuda.is_available():
            motions = motions.cuda()

        if volatile:
            with torch.no_grad:
                mean_origin = Variable(mean_origin)
        else:
            mean_origin = Variable(mean_origin, requires_grad=True)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if volatile:
            with torch.no_grad:
                vidoes_mask = Variable(vidoes_mask)
        else:
            vidoes_mask = Variable(vidoes_mask, requires_grad=True)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()

        vis_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)
        vid_embs = self.vid_encoder(vis_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                vid_embs = decod(vid_embs)
        else:
            vid_embs=self.unify_decoder(vid_embs)
        sigmoid_out=self.sigmod(vid_embs)
        return sigmoid_out


    def embed_txt(self, txt_data, volatile=True,sigmoid_output=False):
        # text data
        textual_feat = txt_data
        if textual_feat is not None:
            textual_feat = Variable(textual_feat, volatile=volatile)
            if torch.cuda.is_available():
                textual_feat = textual_feat.cuda()

        txt_data = textual_feat
        text_emb = self.text_encoder(txt_data)
        if sigmoid_output:
            pred = self.text_encoder(txt_data)
            if self.decoder_num_layer > 2:
                for decod in self.unify_decoder:
                    pred = decod(pred)
            else:
                pred=self.unify_decoder(pred)
            sigmoid_out=self.sigmod(pred)
            return text_emb, sigmoid_out
        else:
            return text_emb

    def embed_txt_concept_only(self, txt_data, volatile=True):
        # text data
        textual_feat = txt_data
        if textual_feat is not None:
            textual_feat = Variable(textual_feat, volatile=volatile)
            if torch.cuda.is_available():
                textual_feat = textual_feat.cuda()


        txt_data = textual_feat
        text_emb = self.text_encoder(txt_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                text_emb = decod(text_emb)
        else:
            text_emb=self.unify_decoder(text_emb)
        sigmoid_out=self.sigmod(text_emb)
        return sigmoid_out


class ITV(BaseModel):
    """
    ITV network
    """
    def __init__(self, opt):
        # Build Models
        self.modelname = opt.postfix
        self.grad_clip = opt.grad_clip
        self.vid_encoder = Video_encoder(opt)
        self.text_encoder = Text_multi_level_encoder(opt)
        self.decoder_num_layer=len(opt.decoder_mapping_layers)
        if 'ul_type' in opt:
            self.ul_type = opt.ul_type
        if len(opt.decoder_mapping_layers)==2:
            self.unify_decoder =MFC(opt.decoder_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)
        elif len(opt.decoder_mapping_layers)==3:
            mapping_layer1 = [opt.decoder_mapping_layers[0],opt.decoder_mapping_layers[1]]
            mapping_layer2 = [opt.decoder_mapping_layers[1],opt.decoder_mapping_layers[2]]
            self.unify_decoder=nn.ModuleList([MFC(mapping_layer1, opt.dropout, have_bn=False, have_last_bn=False),
                                              MFC(mapping_layer2, opt.dropout, have_bn=True,
                                                  have_last_bn=True)])

        else:
            NotImplemented
        self.sigmod = nn.Sigmoid()

        self.loss_type = opt.loss_type
        self.unlikelihood=opt.unlikelihood
        self.ul_alpha = opt.ul_alpha
        self.concept = opt.concept
        if self.unlikelihood:
            self.contradicted_matrix_sp = opt.contradicted_matrix_sp

        print(self.vid_encoder)
        print(self.text_encoder)
        print(self.unify_decoder)
        print(self.sigmod)
        if torch.cuda.is_available():
            self.vid_encoder.cuda()
            self.text_encoder.cuda()
            self.unify_decoder.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer

        self.matching_loss  = TripletLoss(margin=opt.margin,
                                              measure=opt.measure,
                                              max_violation=opt.max_violation,
                                              cost_style=opt.cost_style,
                                            direction=opt.direction)

        self.likelihoodLoss = likelihoodBCEloss(opt.multiclass_loss_lamda,opt.cost_style)

        if self.unlikelihood:
            self.unlikelihoodLoss = unlikelihoodBCEloss(self.contradicted_matrix_sp,opt.cost_style)

        params_end_text = list(self.text_encoder.parameters())
        params_end_vid = list(self.vid_encoder.parameters())
        params_unify_dec = list(self.unify_decoder.parameters())
        self.params_end_text = params_end_text
        self.params_end_vid = params_end_vid
        self.params_unify_dec=params_unify_dec
        params= params_end_text+params_end_vid+params_unify_dec
        self.params = params


        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0


    def forward_matching(self, videos, targets, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        frames,motions,mean_origin, video_lengths, vidoes_mask = videos
        frames = Variable(frames, requires_grad=True)
        if volatile:
            with torch.no_grad():
                frames = Variable(frames)
        if torch.cuda.is_available():
            frames = frames.cuda()

        motions = Variable(motions, requires_grad=True)
        if volatile:
            with torch.no_grad():
                motions = Variable(motions)
        if torch.cuda.is_available():
            motions = motions.cuda()

        mean_origin = Variable(mean_origin, requires_grad=True)
        if volatile:
            with torch.no_grad():
                mean_origin = Variable(mean_origin)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, requires_grad=True)
        if volatile:
            with torch.no_grad():
                vidoes_mask = Variable(vidoes_mask)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)

        # text data
        captions, cap_bows, lengths, cap_masks = targets
        if captions is not None:
            captions = Variable(captions)
            if volatile:
                with torch.no_grad():
                    captions = Variable(captions)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows)
            if volatile:
                with torch.no_grad():
                    cap_bows = Variable(cap_bows)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks)
            if volatile:
                with torch.no_grad():
                    cap_masks = Variable(cap_masks)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()


        text_data = (captions, cap_bows, lengths, cap_masks)


        vid_emb = self.vid_encoder(videos_data)
        cap_emb = self.text_encoder(text_data)
        return vid_emb, cap_emb

    def forward_classification(self, vid_embs,text_embs, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        if self.decoder_num_layer>2:
            for decod in self.unify_decoder:
                vid_embs = decod(vid_embs)
                text_embs = decod(text_embs)
        else:
            text_embs=self.unify_decoder(text_embs)
            vid_embs = self.unify_decoder(vid_embs)
        pred_vid=self.sigmod(vid_embs)
        pred_text=self.sigmod(text_embs)


        return pred_vid,pred_text

    def embed_vis(self, vis_data, volatile=True,sigmoid_output=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data

        if volatile:
            with torch.no_grad():
                frames = Variable(frames)
        else:
            frames = Variable(frames, requires_grad=True)
        if torch.cuda.is_available():
            frames = frames.cuda()

        if volatile:
            with torch.no_grad():
                motions = Variable(motions)
        else:
            motions = Variable(motions, requires_grad=True)

        if torch.cuda.is_available():
            motions = motions.cuda()

        if volatile:
            with torch.no_grad():
                mean_origin = Variable(mean_origin)
        else:
            mean_origin = Variable(mean_origin, requires_grad=True)


        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if volatile:
            with torch.no_grad():
                vidoes_mask = Variable(vidoes_mask)
        else:
            vidoes_mask = Variable(vidoes_mask, requires_grad=True)

        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, motions,mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_encoder(vis_data)
        pred= self.vid_encoder(vis_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                pred = decod(pred)
        else:
            pred=self.unify_decoder(pred)
        sigmoid_out=self.sigmod(pred)

        if sigmoid_output:
            return embs,sigmoid_out
        else:
            return embs

    def embed_vis_emb_only(self, vis_data, volatile=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        motions = Variable(motions, volatile=volatile)
        if torch.cuda.is_available():
            motions = motions.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_encoder(vis_data)
        return embs

    def embed_vis_concept_only(self, vis_data, volatile=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data
        if volatile:
            with torch.no_grad:
                frames = Variable(frames)
        else:
            frames = Variable(frames, requires_grad=True)
        if torch.cuda.is_available():
            frames = frames.cuda()

        if volatile:
            with torch.no_grad:
                motions = Variable(motions)
        else:
            motions = Variable(motions, requires_grad=True)
        if torch.cuda.is_available():
            motions = motions.cuda()

        if volatile:
            with torch.no_grad:
                mean_origin = Variable(mean_origin)
        else:
            mean_origin = Variable(mean_origin, requires_grad=True)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if volatile:
            with torch.no_grad:
                vidoes_mask = Variable(vidoes_mask)
        else:
            vidoes_mask = Variable(vidoes_mask, requires_grad=True)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()

        vis_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)
        vid_embs = self.vid_encoder(vis_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                vid_embs = decod(vid_embs)
        else:
            vid_embs=self.unify_decoder(vid_embs)
        sigmoid_out=self.sigmod(vid_embs)
        return sigmoid_out


    def embed_txt(self, txt_data, volatile=True,sigmoid_output=False):
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()

        txt_data = (captions, cap_bows, lengths, cap_masks)
        text_emb = self.text_encoder(txt_data)
        if sigmoid_output:
            pred = self.text_encoder(txt_data)
            if self.decoder_num_layer > 2:
                for decod in self.unify_decoder:
                    pred = decod(pred)
            else:
                pred=self.unify_decoder(pred)
            sigmoid_out=self.sigmod(pred)
            return text_emb, sigmoid_out
        else:
            return text_emb

    def embed_txt_concept_only(self, txt_data, volatile=True):
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()

        txt_data = (captions, cap_bows, lengths, cap_masks)
        text_emb = self.text_encoder(txt_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                text_emb = decod(text_emb)
        else:
            text_emb=self.unify_decoder(text_emb)
        sigmoid_out=self.sigmod(text_emb)
        return sigmoid_out





