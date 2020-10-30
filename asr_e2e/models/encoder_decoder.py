from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from asr_e2e.models.model import Model
from asr_e2e.utils.utils import deco_print

class EncoderDecoderModel(Model):
    """
    Standard encoder-decoder class with one encoder and one decoder.
    """

    @staticmethod
    def get_required_params():
        return dict(Model.get_required_params(),
                **{"encoder": None,
                   "decoder": None})

    @staticmethod
    def get_optional_params():
        return dict(Model.get_optional_params(),
                **{"encoder_params": dict,
                   "decoder_params": dict,
                   "loss": None,
                   "loss_params": dict})
    
    def __init__(self, params, mode = "train"):
        """
        和Model基类一样，构图的代码不在init中完成，
        而在self._build_forward_pass_graph()中完成
        """

        super(EncoderDecoderModel, self).__init__(params = params, mode = mode)

        if "encoder_params" not in self.params:
            self.params["encoder_params"] = {}
        if "decoder_params" not in self.params:
            self.params["decoder_params"] = {}
        if "loss_params" not in self.params:
            self.params["loss_params"] = {}

        self._encoder = self._create_encoder()
        self._decoder = self._create_decoder()

        if self.mode == "train" or self.mode == "eval":
            self._loss_computator = self._create_loss()
        else:
            self._loss_computator = None

    def _create_encoder(self):
        """
        Return encoder class derived from encoders.encoder.Encoder
        """

        params = self.params["encoder_params"]
        return self.params["encoder"](params = params, mode = self.mode, model = self)

    def _create_decoder(self):
        params = self.params["decoder_params"]
        return self.params["decoder"](params = params, mode = self.mode, model = self)

    def _create_loss(self):
        """
        Return the instance of class derived from losses.loss.Loss
        """
        return self.params["loss"](params = self.params["loss_params"], model = self)

    def _build_forward_pass_graph(self, input_tensors, gpu_id = 0):
        """
        This function connects encoder, decoder and loss together.
        As an input for encoder it will specify source tensors ( as returned from the data layer).
        As an input for decoder it will specify target tensors as well as all output returned from encoder.
        As an input for loss it will specify target tensors and all output returned from decoder.

        Inputs
            input_tensors(dict): 
                "source_tensors"
                "target_tensors" (train or eval)
        Returns
            tuple: tuple containing loss tensor as returned from loss.compute_loss()
            and list of output tensors, which is taken from decoder.decode()["outputs"]
        """
        if not isinstance(input_tensors, dict) or "source_tensors" not in input_tensors:
            raise ValueError("input tensors should be a dict containing 'source_tensors' key")

        if not isinstance(input_tensors["source_tensors"], list):
            raise ValueError("source_tensors should be a list")

        source_tensors = input_tensors["source_tensors"]
        if self.mode == "train" or self.mode == "eval":
            if "target_tensors" not in input_tensors:
                raise ValueError("Input tensors should contain 'target_tensors' key")
            if  not isinstance(input_tensors["target_tensors"], list):
                raise ValueError("target_tensors should be a list")
            target_tensors = input_tensors["target_tensors"]

        with tf.variable_scope("ForwardPass"):
            encoder_input = {"source_tensors": source_tensors}
            encoder_output = self.encoder.encode(input_dict = encoder_input)

            decoder_input = {"encoder_output": encoder_output}
            if self.mode == "train" or self.mode == "eval":
                decoder_input["target_tensors"] = target_tensors
            decoder_output = self.decoder.decode(input_dict = decoder_input)

            model_outputs = decoder_output.get("outputs", None)

            if self.mode == "train" or self.mode == "eval":
                with tf.variable_scope("Loss"):
                    loss_input_dict = {
                            "decoder_output": decoder_output,
                            "target_tensors": target_tensors}
                    loss = self.loss_computator.compute_loss(loss_input_dict)
            else:
                deco_print("Inference mode, Loss part of graph isn't build")
                loss = None
        return loss, model_outputs
    
    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def loss_computator(self):
        return self._loss_computator

