from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import abc
import copy

import six
import tensorflow as tf

# from asr_e2e.optimizers.mp_wrapper import mp_regularizer_wrapper
from asr_e2e.utils.utils import check_params, cast_types

@six.add_metaclass(abc.ABCMeta)
class Decoder:
    
    @staticmethod
    def get_required_params():
        return {}

    @staticmethod
    def get_optional_method():
        return {
                "regularizer": None,
                "regularizer_params": dict,
                "initializer": None,
                "initializer_params": dict,
                "dtype": [tf.float32, tf.float16, "mixed"]}

    def __init__(self, params, model, name = "decoder", mode = "train"):
        
        check_params(params, self.get_required_params(), self.get_optional_method())
        self._params = params
        self._model = model

        if "dtype" not in self._params:
            if self._model:
                self._params["dtype"] = self._model.params["dtype"]
            else:
                self._params["dtype"] = tf.float32

        self._name = name
        self._mode = mode
        self._compiled = False

    def decode(self, input_dict):
        if not self._compiled:
            if "regularizer" not in self._params:
                if self._model and "regularizer" in self._model.params:
                    self._params["regularizer"] = copy.deepcopy(self._model.params["regularizer"])
                    self._params["regularizer_params"] = copy.deepcopy(self._model.params["regularizer_params"])

            if "regularizer" in self._params:
                init_dict = self._params.get("regularizer_params", None)
                if self._params["regularizer"] is not None:
                    self._params["regularizer"] = self._params["regularizer"](**init_dict)
                # if self._params["dtype"] == "mixed":
                #    self._params["regularizer"] = mp_regularizer_wrapper(self._params["regularizer"])

        if "initializer" in self._params:
            init_dict = self.params.get("initializer_params", {})
            initializer = self._params["initializer"](**init_dict)
        else:
            initializer = None

        self._compiled = True

        with tf.variable_scope(self._name, initializer = initializer, dtype = self.params["dtype"]):
            return self._decode(self._cast_types(input_dict))

    def _cast_types(self, input_dict):
        return cast_types(input_dict, self.params["dtype"])

    def _decode(self, input_dict):
        pass

    @property
    def params(self):
        return self._params

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return self._name



