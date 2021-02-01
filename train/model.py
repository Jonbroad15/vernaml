import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv
from functools import partial
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

def model_from_hparams(hparams, verbose=True):
    """
    Just interfacing to create a model directly from an hparam file
    :param hparams:
    :return:
    """
    num_rels = hparams.get('argparse', 'num_edge_types')
    model = Model(dims=hparams.get('argparse', 'embedding_dims'),
                  self_loop=hparams.get('argparse', 'self_loop'),
                  num_rels=num_rels,
                  lin_output = hparams.get('argparse', 'lin_output'),
                  num_bases=-1,
                  verbose=verbose)
    return model

class Embedder(nn.Module):

    def __init__(   self,
                    dims,
                    num_rels,
                    num_bases = -1,
                    lin_output = False,
                    self_loop = False,
                    verbose = True):
        super(Embedder, self).__init__()
        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.verbose = verbose
        self.lin_output = lin_output

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
    # TODO: add a softmax layer to squish a vector between 0 and 1
    def build_output_layer(self, in_dim, out_dim):
        if self.lin_output: return nn.Linear(in_dim, out_dim)

        return RelGraphConv(in_dim, out_dim, self.num_rels,
                            num_bases = self.num_bases,
                            activation=torch.sigmoid)


    def forward(self, g):
        h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            if self.lin_output and (i == len(self.layers) - 1):
                h = torch.sigmoid(layer(h))
                h = h.view(-1)
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
                self_loop = False,
                lin_output = False,
                weighted = False,
                verbose = True):
        """

        :param dims: the embeddings dimensions, a list of type [128, 128, 32]
        :param num_rels: number of possible edge types
        :param num_bases: technical RGCN option

        """
        super(Model, self).__init__()
        self.verbose = verbose
        self.dims = dims
        self.dimension_embedding = dims[-1]

        self.num_rels = num_rels
        self.num_bases = num_bases

        self.weighted = weighted
        self.self_loop = self_loop

        # create rgcn layers for the embedder
        self.embedder = Embedder(dims = dims,
                                num_rels = num_rels,
                                num_bases = num_bases,
                                self_loop = self_loop,
                                lin_output = lin_output,
                                verbose=verbose)

    def forward(self, g):
        self.embedder(g)
        return g.ndata['h']

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device







