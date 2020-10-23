from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import numpy as np
import six
import abc

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

from asr_e2e.utils.utils import deco_print, clip_last_batch
from asr_e2e.optimizers import optimize_loss, get_regularization_loss
from asr_e2e.utils.utils import check_params

@six.add_metaclass(abc.ABCMeta)

class Model:
    @staticmethod
    def get_required_params():
        return {
                'batch_size_per_gpu':int,
                'data_layer': None
                }


