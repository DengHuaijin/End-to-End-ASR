from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import abc
import copy

import six
import tensorflow as tf

from asr_e2e.optimizers.mp_wrapper import mp_regularizer_wrapper
from asr_e2e.utils.utils import check_params, cast_types

@six.addmetaclass(abc.ABCMeta)
class Encoder:

    @staticmethod
    def get_required_params():
        return {}

    @staticmethod
    def get_optional_params():
        return {
                "regularizer": None,
                "regularizer_params": dict,
                "initializer": None,
                "initializer_params": dict,
                "dtype": [tf.float32, tf.float16, "mixed"]
               }

    def __init__(self, params, model, name = "encoder", mode = "train"):
    
        check_params(params, self.get_required_params(), self.get_optional_params())
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

    def encode(self, input_dict):

        if not self._compiled:
            if "regularizer" not in self._params:
                if self._model and "regularizer" in self._model.params:
                    self._params["regularizer"] = copy.deepcopy(self._model.params["regularizer"])
                    self._params["regularizer_params"] = copy.deepcopy(self._model.params["regularizer_params"])

            if "regularizer" in self._params:
                init_dict = self._params.get("regularizer_params", {})
                if self._params["regularizer"] is not None:
                    self._params["regularizer"] = self._params["regularizer"](**init_dict)
                if self._params["dtype"] == "mixed":
                    self._params["regularizer"] = mp_regularizer_wrapper(self._params["regularizer"])

            if self._params["dtype"] == "mixed":
                self._params["dtype"]  =tf.float16

        if "initializer" in self.params:
            init_dict = self.params.get("initializer_params", {})
            initializer = self.params["initializer"](**init_dict)
        else:
            initializer = None

        self._compiled = True
        
        witf tf.variable_scope(self._name, initializer = initializer, dtype = self.params["dtype"]):
            return self._encode(self._cast_types(input_dict))

    def _cast_types(self, input_dict):
        """
        This function performs automatic cast of all inputs to encoder type.
        """
        return cast_types(input_dict, self.params["dtype"])
    
    @abc.abstractmethod
    def _encode(self, input_dict):
        """
        This function should construct encoder graph.
        Inputs
            input_dict:
                If the encoder is used with models.encoder_decoder class,
                input_dict will have the content:
                    {
                        "source_tensors":data_layer.input_tensors["source_tensors"]
                    }
        Returns:
            dict:
                {
                    "outputs": outputs,
                    "state": state
                }
        """
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

