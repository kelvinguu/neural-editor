from collections import defaultdict
from itertools import izip

import numpy as np
import torch
from gtd.ml.torch.utils import GPUVariable

from gtd.ml.torch.seq_batch import SequenceBatch


class Alignments(object):
    """
    Attributes:
        indices (Variable): of shape (batch_size, max_target_seq_length, max_alignments)
        mask (Variable): of shape (batch_size, max_target_seq_length, max_alignments)

    max_alignments is always at least 1, so that indices and mask do not have a dimension of 0 (which confuses
        downstream Torch ops)
    """
    def __init__(self, source_words, target_words):
        """Represent alignments as a Tensor.

        Args:
            source_words (list[list[unicode]]): batch of source sequences
            target_words (list[list[unicode]]): batch of target sequences
        """
        assert len(source_words) == len(target_words)
        # compute alignments
        alignments_batch = [self._align(s, t) for s, t in izip(source_words, target_words)]

        # compute dimensions of alignment tensor
        batch_size = len(alignments_batch)
        max_target_seq_length = 0
        max_alignments = 1  # make this dimension at least 1, so that we don't get a tensor with no entries
        for alignments in alignments_batch:
            max_target_seq_length = max(max_target_seq_length, len(alignments))
            for align in alignments:
                max_alignments = max(max_alignments, len(align))

        indices = -1 * np.ones((batch_size, max_target_seq_length, max_alignments), dtype=np.int64)
        # filled with -1's for padding.
        # int64 gets converted into torch.LongTensor

        for i, alignments in enumerate(alignments_batch):
            for j, align in enumerate(alignments):
                for k, idx in enumerate(align):
                    indices[i, j, k] = idx

        mask = (indices != -1).astype(np.float32)
        indices[indices == -1] = 0
        self._indices = GPUVariable(torch.from_numpy(indices))
        self._mask = GPUVariable(torch.from_numpy(mask))

    @property
    def indices(self):
        return self._indices

    @property
    def mask(self):
        return self._mask

    @classmethod
    def _align(self, source_seq, target_seq):
        """For each target word, give its positions in the source sequence.

        Args:
            source_seq (list[unicode])
            target_seq (list[unicode])

        Returns:
            alignments (list[list[int]]): alignments[i] is an ordered list of the indices where target_seq[i]
                appears in source_seq.
        """
        alignments_dict = defaultdict(list)
        for i, word in enumerate(source_seq):
            alignments_dict[word].append(i)

        alignments = []
        for word in target_seq:
            alignments.append(alignments_dict[word])

        return alignments

    def split(self):
        """Split alignments object into per-time-step alignments.

        Returns:
            list[SequenceBatch]: where each element has shape (batch_size, max_alignments)
        """
        indices_list = [v.squeeze(1) for v in self.indices.split(1, dim=1)]
        mask_list = [v.squeeze(1) for v in self.mask.split(1, dim=1)]
        return [SequenceBatch(i, m) for i, m in izip(indices_list, mask_list)]