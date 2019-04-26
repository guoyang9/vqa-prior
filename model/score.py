import torch
import torch.nn as nn


class Transformer(nn.Sequential):
    def __init__(self, in_features, out_features, drop=0.0):
        """ transforming features into a same latent space. """
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
                raise ValueError('cannot understand relation type.')
            return self.score_com(vqa)


class PairWiseLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(PairWiseLoss, self).__init__()
        self.loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, x1, x2):
        y = torch.ones(x1.shape[-1]).cuda()
        return self.loss(x1, x2, y)
