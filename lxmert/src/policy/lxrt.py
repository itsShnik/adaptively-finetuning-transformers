# written by Nikhil Shah

import torch.nn as nn

from param import args
from lxrt.entry import PolicyLXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20

class PolicyLXRT(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        
        # Build LXRT encoder
        #TODO: Make a new class in entry file
        self.policy_lxrt_encoder = PolicyLXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.policy_lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_blocks)
        )
        self.logit_fc.apply(self.policy_lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.policy_lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


