import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv

import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

class Embedder(nn.Module):

    def __init__(   self,
                    dims,
                    num_rels,
                    num_bases = -1,
                    conv_output = False,
                    self_loop = False,
                    verbose = True):
        super(Embedder, self).__init__()
        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.conv_output = conv_output
        self.self_loop = self_loop
        self.verbose = verbose

        self.layers = self.build_model()
        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = self.dims[-2:]
        if self.verbose:
            print("short, ", short)
            print('last_hidden, last', last_hidden, last)

        # input feature is just node degree
        i2h = self.build_hidden_layer(1, self.dims[0])
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = self.build_hidden_layer(dim_in, dim_out)
            layers.append(h2h)

        # hidden to output
        h2o = self.build_output_layer(last_hidden, last)
        layers.append(h2o)
        return layers

        @property
        def current_device(self):
            """
            return: current device this model is on
            """
            return next(self.parameters()).device

        def build_hidden_layer(self, in_dim, out_dim):
            return RelGraphConv(in_dim, out_dim, self.num_rels,
                                num_bases=self.num_bases,
                                activation=F.relu,
                                self_loop=self.self_loop)

        # No activation for the last layer
        def build_output_layer(self, in_dim, out_dim, conv=False):
            if self.conv_output:
                return RelGraphConv(in_dim, out_dim, self.num_rels,
                                    num_bases = self.num_bases,
                                    self_loop = self.self_loop,
                                    activation=None)
            else:
                return nn.Linear(in_dim, out_dim)

        # TODO: Edit the forward function to fit my edata
        def forward(self, g):
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
            for i, layer in enumerate(self.layers):
                if not self.conv_output and (i == len(self.layers) - 1):
                    h = layer(h)
                else:
                    h = layer(g, h, g.edata['one_hot'])
            g.ndata['h'] = h
            return g.ndata['h']

########################################################################
# Define full R-GCN Model
# ~~~~~~~~~~~~~~~~~~~~~~~

class Model(nn.Module):
    def __init__(self,
                dims,
                num_rels,
                num_bases = -1,
                conv_output = True,
                self_loop = False,
                hard_embed = False,
                similarity = True,
                normalize = False,
                weighted = False,
                verbose = True):
        """

        :param dims: the embeddings dimensions, a list of type [128, 128, 32]
        :param num_rels: number of possible edge types
        :param num_bases: technical RGCN option
        :param similarity: if we want to use cosine similarities instead of distances

        """
        super(Model, self).__init__()
        self.verbose = verbose
        self.dims = dims
        self.dimension_embedding = dims[-1]

        self.num_rels = num_rels
        self.num_bases = num_bases

        self.similarity = similarity
        self.normalize = normalize
        self.weighted = weighted
        self.self_loop = self_loop

        # create rgcn layers for the embedder
        self.embedder = Embedder(dims = dims,
                                num_rels = num_rels,
                                num_bases = num_bases,
                                self_loop = self_loop,
                                conv_output = conv_output,
                                verbose=verbose)

    def forward(self, g):
        # if hard embed, the embeddings are directly g.ndata['h'] otherwise we compute
        self.embedder(g)

        # if using similarity as supervision, we should normalize the embeddings,
        # as their norm got unconstrained
        if self.similarity and self.normalize:
            g.ndata['h'] = F.normalize(g.ndata['h'], p=2, dim=1)
        return g.ndata['h']

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.paramters()).device

    # Below are loss computation function related to this model
    @staticmethod
    def matrix_cosine(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    @staticmethod
    def matrix_dist(a , plus_one = False):
        """
        Pairwise dist of set of vector of size b
        returns a matrix of size (a, a)
        :param a: a torch Tensor of size a, b
        :param plus_one: if we want to get positive values
        """
        if plus_one:
            return torch.norm(a[:, None] - a, dim=2, p=2) + 1
        else:
            return torch.norm(a[:, None] - a, dim = 2, p=2)

    @staticmethod
    def weighted_MSE(output, target, weight):
        if weight is None:
            return torch.nn.MSELoss()(output, target)
        return torch.mean(weight * (output - target) **2)







