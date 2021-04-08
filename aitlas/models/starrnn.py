"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of StarRNN model:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/StarRNN.py

"""

"""
__author__ = Türkoglu Mehmet Özgür <ozgur.turkoglu@geod.baug.ethz.ch>
"""

import torch.nn as nn
import torch.utils.data
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import math

from ..base import BaseMulticlassClassifier
from .schemas import StarRNNSchema

#__all__ = ['StarRNN']

class StarRNN(BaseMulticlassClassifier):

    schema = StarRNNSchema

    def __init__(self, config):

        BaseMulticlassClassifier.__init__(self, config)
        
        device=torch.device("cpu") # how to handle this?

        #self.modelname = f"StarRNN_input-dim={input_dim}_num-classes={num_classes}_" \
        #                 f"hidden-dims={hidden_dims}_num-layers={num_layers}_dropout={dropout}"

        self.d_model = self.config.num_layers*self.config.hidden_dims
        
        if self.config.use_layernorm:
            self.model.inlayernorm = nn.LayerNorm(self.config.input_dim)
            self.model.clayernorm = nn.LayerNorm((self.config.hidden_dims + self.config.hidden_dims * self.config.bidirectional) )

        self.model.block = torch.nn.Sequential(
            StarLayer(input_dim=self.config.input_dim, hidden_dim=self.config.hidden_dims, droput_factor=self.config.dropout, device=device),
            *[StarLayer(input_dim=self.config.hidden_dims, hidden_dim=self.config.hidden_dims, droput_factor=self.config.dropout, device=device)] * (self.config.num_layers-1)
        )

        if self.config.bidirectional:
            hidden_dims = self.config.hidden_dims * 2
        else:
            hidden_dims = self.config.hidden_dims

        self.model.linear_class = nn.Linear(hidden_dims, self.config.num_classes, bias=True)

        if self.config.use_batchnorm:
            if self.config.bidirectional:
                self.model.bn = nn.BatchNorm1d(hidden_dims*2)
            else:
                self.model.bn = nn.BatchNorm1d(hidden_dims)

        #self.to(device)

    def _logits(self, x):
        #x = x.transpose(1,2)

        if self.config.use_layernorm:
            x = self.model.inlayernorm(x)

        outputs = self.model.block(x)
        
        if self.config.use_batchnorm:
            outputs = outputs[:,-1:,:]
            b,t,d = outputs.shape
            o_ = outputs.view(b, -1, d).permute(0,2,1)
            outputs = self.model.bn(o_).permute(0, 2, 1).view(b,t,d)

        h=outputs[:,-1,:] 
        
        if self.config.use_layernorm:
            h = self.model.clayernorm(h)

        logits = self.model.linear_class.forward(h)

        return logits

    def forward(self, x):
        logits = self._logits(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        # stack the lists to new tensor (b,d,t,h,w)
        return logprobabilities

    '''def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot'''


class StarCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(StarCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x_K = nn.Linear(input_size, hidden_size, bias=bias)
        self.x_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.h_K = nn.Linear(hidden_size, hidden_size, bias=bias)

        # self.reset_parameters()
        init.orthogonal_(self.x_K.weight)
        init.orthogonal_(self.x_z.weight)
        init.orthogonal_(self.h_K.weight)

        #        bias_f= np.log(np.random.uniform(1,45,hidden_size))
        #        bias_f = torch.Tensor(bias_f)
        #        self.bias_K = Variable(bias_f.cuda(), requires_grad=True)

        self.x_K.bias.data.fill_(1.)
        self.x_z.bias.data.fill_(0)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x_K = self.x_K(x)
        gate_x_z = self.x_z(x)
        gate_h_K = self.h_K(hidden)

        gate_x_K = gate_x_K.squeeze()
        gate_x_z = gate_x_z.squeeze()
        gate_h_K = gate_h_K.squeeze()

        # K_gain = torch.sigmoid(gate_x_K + gate_h_K + self.bias_K )
        K_gain = torch.sigmoid(gate_x_K + gate_h_K)
        z = torch.tanh(gate_x_z)

        h_new = K_gain * hidden + (1 - K_gain) * z
        # h_new = hidden + K_gain * ( z - hidden)
        h_new = torch.tanh(h_new)

        return h_new


class StarLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, droput_factor=0.2, batch_norm=True, layer_norm=False, device=torch.device("cpu")):
        super(StarLayer, self).__init__()
        # Hidden dimensions
        self.device = device
        self.hidden_dim = hidden_dim
        self.cell = StarCell(input_dim, hidden_dim, bias)
        self.droput_factor = droput_factor
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        if self.droput_factor != 0:
            self.naive_dropout = nn.Dropout(p=droput_factor)

        if batch_norm:
            # print('batch norm')
            self.bn_layer = nn.BatchNorm1d(hidden_dim)
        if layer_norm:
            self.layer_norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(x.size(0), self.hidden_dim)).to(self.device)
        outs = Variable(torch.zeros(x.size(0), x.shape[1], self.hidden_dim)).to(self.device)

        hn = h0

        for seq in range(x.size(1)):
            hn = self.cell(x[:, seq], hn)

            if self.droput_factor != 0:
                outs[:, seq, :] = self.naive_dropout(hn)
            else:
                outs[:, seq, :] = hn

        # batch normalization:
        if self.batch_norm:
            outs = self.bn_layer(outs.permute(0, 2, 1)).permute(0, 2, 1)

            # layer normalization:
        if self.layer_norm:
            outs = self.layer_norm_layer(outs)

        return outs