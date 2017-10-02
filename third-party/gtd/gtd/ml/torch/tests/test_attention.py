import numpy as np
import torch

from gtd.ml.torch.attention import Attention
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.torch.utils import assert_tensor_equal


class TestAttention(object):

    def test_forward(self):
        float_tensor = lambda arr: torch.FloatTensor(arr)
        float_tensor_var = lambda arr: GPUVariable(torch.FloatTensor(arr))

        batch_size, num_cells = 5, 2
        memory_dim, query_dim, attn_dim = 4, 3, 2

        memory_transform = np.array([ # Wh: (memory_dim x attn_dim)
                                    [.1, .5],
                                    [.2, .6],
                                    [.3, .7],
                                    [.4, .8],
                                ])
        query_transform = np.array([ # Ws: (query_dim x attn_dim)
                                   [.3, .4],
                                   [.2, .5],
                                   [.1, .6],
                                ])
        v_transform = np.array([ # v: (attn_dim x 1)
                               [.1],
                               [.8],
                            ])

        mem_values = np.array([ # (batch_size x num_cells x memory_dim)
                              [
                                [.1, .2, .3, .4],
                                [.4, .5, .6, .7],
                              ],
                              [
                                [.2, .3, .4, .5],
                                [.6, .7, .8, .9],
                              ],
                              [
                                [.3, .4, .5, .6],
                                [.7, .8, .9, .1],
                              ],
                              [
                                [-8, -9, -10, -11],
                                [-12, -13, -14, -15],
                              ],
                              [
                                [8, 9, 10, 11],
                                [12, 13, 14, 15],
                              ]
                        ])
        mem_values = float_tensor_var(mem_values)
        mem_mask = np.array([
                            [1, 0],
                            [1, 1],
                            [1, 0],
                            [0, 0],
                            [0, 1],
                          ])
        mem_mask = float_tensor_var(mem_mask)
        memory_cells = SequenceBatch(values=mem_values, mask=mem_mask)
        query = np.array([ # (batch_size x query_dim)
                         [.1, .2, .3],
                         [.4, .5, .6],
                         [.7, .8, .9],
                         [10, 11, 12],
                         [13, 14, 15],
                    ])
        query = float_tensor_var(query)

        # compute manually
        # Et = np.array([ [ 0.65388812,  0.81788159],
        #                 [ 0.81039669,  0.87306204],
        #                 [ 0.86236411,  0.86977563]])
        manual_weights = np.array([[ 1.,  0.],
                                   [ 0.4843,  0.5156],
                                   [ 1.,  0.],
                                   [0, 0],
                                   [0, 1.],
                                   ])
        manual_context = np.array([[ 0.1,  0.2,  0.3,  0.4],
                                   [ 0.4062,  0.5062,  0.6062,  0.7062],
                                   [ 0.3,  0.4,  0.5,  0.6],
                                   [0, 0, 0, 0],
                                   [12, 13, 14, 15],
                                   ])

        # compute with module
        attn = Attention(memory_dim, query_dim, attn_dim)
        attn.memory_transform.data.set_(float_tensor(memory_transform))
        attn.query_transform.data.set_(float_tensor(query_transform))
        attn.v_transform.data.set_(float_tensor(v_transform))

        attn_out = attn(memory_cells, query)
        assert_tensor_equal(attn_out.weights, manual_weights, decimal=4)
        assert_tensor_equal(attn_out.context, manual_context, decimal=4)