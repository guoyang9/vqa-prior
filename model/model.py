import torch
import torch.nn as nn

import utils.config as config
import model.text as text
import model.attention as attention
import model.score as score


class Net(nn.Module):
    """ embedding_tokens: the number of question word embeddings. """
    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        answer_feature = 300
        glimpses = 2
        drop_rate = 0.5
        vision_features = config.output_features

        self.text = text.TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=drop_rate,
        )
        self.attention = attention.Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=drop_rate,
        )
        self.transform_qv = score.Transformer(
            in_features=glimpses*vision_features+question_features,
            out_features=1024,
            drop=drop_rate
        )
        self.transform_q = score.Transformer(
            in_features=question_features, 
            out_features=1024,
            drop=drop_rate
        )
        self.score = score.SocreModule(
            qv_features=1024,
            a_features=300,
            drop=drop_rate,
            relation='mul'
        )
        self.pair_loss = score.PairWiseLoss(margin=0.0)
        self.classifier = Classifier(
            in_features=1024, 
            out_features=config.max_answers,
            drop=drop_rate,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len, answ):
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8) # l2 normalization on depth dimension
        a = self.attention(v, q)
        v = attention.apply_attention(v, a)

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
