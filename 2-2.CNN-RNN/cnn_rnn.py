import torch
import torch.nn as nn
import torch.nn.functional as F
from cnnencoder import CNN
from attention import AttentionLayer

class CNN_RNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_RNN, self).__init__()
        self.HIDDEN_DIM = kwargs["HIDDEN_DIM"]
        self.GPU = kwargs["GPU"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.cnnencoder = CNN(**kwargs).cuda(self.GPU)
        # self.attention = AttentionLayer(**kwargs).cuda(self.GPU)
        self.decoderCell = nn.LSTMCell(self.WORD_DIM + self.HIDDEN_DIM, self.HIDDEN_DIM).cuda(self.GPU)
        self.decoderLinearTanh = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM).cuda(self.GPU)
        self.decoderLinear = nn.Linear(self.HIDDEN_DIM, self.CLASS_SIZE).cuda(self.GPU)
        #tgt的词表
        self.embedding = nn.Embedding(self.CLASS_SIZE + 1, self.WORD_DIM).cuda(self.GPU)
        self.decoderSoftmax = nn.Softmax(dim=1).cuda(self.GPU)


    def forward(self, inp, tgt, max_len, mode = "train"):
        batch_size = inp.size(0)
        tgt_len = tgt.size(1)

        #encoder
        cnn_feature = self.cnnencoder(inp)
        hx = cx = cnn_feature

        #decoder
        if mode == "train":
            decoderInputinit = torch.zeros((batch_size, 1, self.WORD_DIM)).cuda(self.GPU)
            decoderInput = self.embedding(tgt).cuda(self.GPU)
            decoderInput = torch.cat([decoderInputinit, decoderInput], 1)
            decoderOutput = []
            for idx in range(tgt_len):
                hx, cx = self.decoderCell(torch.cat([decoderInput[:, idx, :], cnn_feature], 1), (hx, cx))
                decoderOutput.append(hx.view(batch_size, 1, self.HIDDEN_DIM))
            decoderOutput = torch.cat(decoderOutput, 1)

            outputFeature = self.decoderLinearTanh(decoderOutput)
            outputFeature = torch.tanh(outputFeature)
            outputFeature = self.decoderLinear(outputFeature)

            return outputFeature
        else:
            decoderInputInit = torch.zeros((batch_size, self.WORD_DIM)).cuda(self.GPU)
            output = []
            decoderInputforTest = decoderInputInit
            for idx in range(max_len):
                hx, cx = self.decoderCell(torch.cat([decoderInputforTest[:, :], cnn_feature], 1), (hx, cx))

                decoderOutput = self.decoderLinearTanh(hx)
                decoderOutput = torch.tanh(decoderOutput)
                decoderOutput = self.decoderLinear(decoderOutput)
                decoderOutput = self.decoderSoftmax(decoderOutput)

                generatePro = torch.log(decoderOutput)
                topGenProb, topGenPos = torch.topk(generatePro, 1)

                output.append(topGenPos)
                decoderInputforTest = self.embedding(topGenPos).view(batch_size, self.WORD_DIM)

            output = torch.cat(output, dim=1)
            return output




