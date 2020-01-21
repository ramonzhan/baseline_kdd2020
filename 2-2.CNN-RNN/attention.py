import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__()
        self.HIDDEN_DIM = kwargs["HIDDEN_DIM"]
        self.encoderHiddendim = int(self.HIDDEN_DIM)
        self.decoderHiddendim = int(self.HIDDEN_DIM)
        self.W = nn.Parameter(torch.Tensor(self.encoderHiddendim, self.decoderHiddendim))
        init.xavier_uniform_(self.W)
        self.softmax = nn.Softmax(dim=2)

    def calculateWithMatrix(self, decoderFeature, encoderFeature):
        subResult = F.linear(decoderFeature, self.W, None)
        attn = torch.bmm(subResult, encoderFeature.transpose(2,1))
        attn = self.softmax(attn)
        sumResult = torch.bmm(attn, encoderFeature)

        return attn, sumResult

    def forward(self, decoderFeature, encoderFeature):
        batchSize = decoderFeature.size(0)
        subResult = F.linear(decoderFeature, self.W, None).view(batchSize, 1, self.encoderHiddendim)

        attn = torch.bmm(subResult, encoderFeature.transpose(2,1))
        attn = self.softmax(attn)

        sumResult = torch.bmm(attn, encoderFeature).view(batchSize, self.encoderHiddendim)
        return attn, sumResult