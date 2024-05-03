import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score


class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, s, im):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()


class likelihoodBCEloss(nn.Module):
    """
    positive cross entropy
    Math:
    """

    def __init__(self,loss_lambda=0.1,cost_style='mean'):
        super(likelihoodBCEloss, self).__init__()
        self.loss_lambda = loss_lambda
        self.cost_style = cost_style

    def forward(self, outs,labels):
        ##compute the unlikehood loss
        loss_lambda = self.loss_lambda
        postive_mask = labels.float()
        negative_mask = 1-postive_mask
        BCEloss= -labels * torch.log(outs+1e-05) - (1 - labels) * torch.log(1 - outs+1e-05)
        postiive_loss =BCEloss*postive_mask
        negative_loss = BCEloss * negative_mask

        invalid_sample_idx = torch.where(torch.sum(postive_mask,1)==0)[0].data.cpu().numpy()
        postiive_loss_batch = torch.sum(postiive_loss,1)/torch.sum(postive_mask,1)
        idx = list(range(0,len(postiive_loss_batch)))
        for iidx in invalid_sample_idx:
            del idx[iidx]
        postiive_loss_batch_new = postiive_loss_batch[idx]
        if torch.sum(torch.isnan(postiive_loss_batch_new).float())>0:
            print('loss in nan')
            loss=torch.tensor(0.0)
            return loss

        negative_loss_batch = torch.sum(negative_loss, 1) / torch.sum(negative_mask, 1)
        negative_loss_batch_new = negative_loss_batch[idx]
        likelyhoodloss = loss_lambda*postiive_loss_batch_new +(1-loss_lambda)*negative_loss_batch_new

        if self.cost_style == 'sum':
            loss = torch.sum(likelyhoodloss)
        else:
            loss = torch.mean(likelyhoodloss)

        if np.isnan(loss.data.cpu().numpy()):
            print('loss in nan')
        return loss



class unlikelihoodBCEloss(nn.Module):
    """
    unlikelihood loss to make constrains on contradicted concept pairs
    Math:
    """

    def __init__(self,contradicted_matrix=None,cost_style='mean'):
        super(unlikelihoodBCEloss, self).__init__()
        self.cost_style = cost_style
        self.contradicted_matrix = contradicted_matrix.to_dense().float()

    def forward(self, outs,labels):
        postive_mask = labels.float()
        ##compute unlikehood loss
        ##compute the number of contradicted pairs in each sample
        contradicted_matrix_mask = torch.sum(self.contradicted_matrix, 1) > 0
        contradicted_matrix_mask = contradicted_matrix_mask.view(contradicted_matrix_mask.shape[0], 1)
        ##each training sample will have how many pairs of contradicted concepts to compute
        contradicted_num_each_sample = torch.matmul(labels, contradicted_matrix_mask.float()).squeeze()
        contradicted_num_positive = contradicted_num_each_sample > 0
        ##compute the UL loss
        prob = torch.log(1 - outs + 1e-05) ##log(1-\hat{g}_t)
        mask = 1-postive_mask  ##(1-g_t)
        prob_mask = prob*mask
        unlikelyhoodloss = torch.matmul(prob_mask,torch.transpose(self.contradicted_matrix, 1, 0))
        unlikelyhoodloss = -unlikelyhoodloss*postive_mask
        ##take the sum of all contradicted pairs
        unlikelyhoodloss_sum = torch.sum(unlikelyhoodloss, 1)
        unlikelyhoodloss_batch_nonzero = unlikelyhoodloss_sum[unlikelyhoodloss_sum>0]
        dividor = torch.sum((unlikelyhoodloss>0).float(),1)[unlikelyhoodloss_sum>0]
        unlikelyhoodloss_batch = unlikelyhoodloss_batch_nonzero/dividor

        if self.cost_style == 'sum':
            loss = torch.sum(unlikelyhoodloss_batch)
        else:
            loss = torch.mean(unlikelyhoodloss_batch)

        if np.isnan(loss.data.cpu().numpy()):
            print('loss in nan')
        return loss



class normalLikelihoodBCEloss(nn.Module):

    def __init__(self,cost_style='mean'):
        super(normalLikelihoodBCEloss, self).__init__()
        self.cost_style = cost_style

    def forward(self, outs,labels):
        BCEloss= -labels * torch.log(outs+1e-05) - (1 - labels) * torch.log(1 - outs+1e-05)
        BCEloss_batch=torch.mean(BCEloss, 1)
        if self.cost_style == 'sum':
            loss = torch.sum(BCEloss_batch)
        else:
            loss = torch.mean(BCEloss_batch)

        if np.isnan(loss.data.cpu().numpy()):
            print('loss in nan')
        return loss


