import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

import utils.config as config


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        answer_feature = 300
        vision_features = config.output_features
        glimpses = 2
        drop_rate = 0.5

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=drop_rate,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=drop_rate,
        )
        self.transform_qv = Transformer(
            in_features=glimpses * vision_features + question_features, # concatenation
            out_features=1024,
            drop=drop_rate
        )
        self.transform_q = Transformer(
            in_features=question_features, 
            out_features=1024,
            drop=drop_rate
        )
        self.classifier = Classifier(
            in_features=1024, 
            out_features=config.max_answers,
            drop=drop_rate,
        )
        self.score = SocreModule(
            qv_features=1024,
            a_features=300,
            drop=drop_rate,
            relation='concat'
        )
        self.pair_loss = PairWiseLoss(margin=0.0)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len, answ):
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8) # l2 normalization on depth dimension
        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        combined = self.transform_qv(combined)
        q = self.transform_q(q)

        score_vq = self.score(combined, answ)
        score_q = self.score(q, answ)
        pair_loss = self.pair_loss(score_vq, score_q)

        answer_vq = self.classifier(combined)
        answer_q = self.classifier(q)

        return answer_vq, answer_q, pair_loss


class Classifier(nn.Sequential):
    def __init__(self, in_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, out_features))


class Transformer(nn.Sequential):
    def __init__(self, in_features, out_features, drop=0.0):
        super(Transformer, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, out_features))
        self.add_module('relu', nn.ReLU())


class SocreModule(nn.Module):
    def __init__(self, qv_features, a_features, drop=0.0, relation='concat'):
        super(SocreModule, self).__init__()
        self.relation = relation
        if self.relation == 'concat':
            qv_features = qv_features + a_features
            self.score_com = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(qv_features, qv_features//2),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(qv_features//2, a_features),
                nn.ReLU()
            )
        else:
            self.qv_map_a = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(qv_features, a_features),
                nn.ReLU()
            )
            self.score_com = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(a_features, a_features),
                nn.ReLU(),
                # nn.Dropout(drop),
                # nn.Linear(a_features, 1),
                # nn.ReLU()
            )

    def forward(self, qv, a):
        if self.relation == 'concat':
            return self.score_com(torch.cat([qv, a], dim=1))
        else: 
            qv = self.qv_map_a(qv)
            if self.relation == 'add':
                vqa = qv + a
            elif self.relation == 'mul':
                vqa = qv * a
            else:
                raise ValueError('could not understand relation type.')
            return self.score_com(vqa)


class PairWiseLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(PairWiseLoss, self).__init__()
        self.loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, x1, x2):
        y = torch.ones(x1.shape[-1]).cuda()
        return self.loss(x1, x2, y)


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True) # useful of variable lengths for rnn
        _, (_, c) = self.lstm(packed)
        # _, (c) = self.lstm(packed)
        return c.squeeze(0)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x)) # two glimpses
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1) # can be updated
    attention = F.softmax(attention, dim=-1)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map) # four dimension
    return tiled
