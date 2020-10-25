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
	"""
	returns:
		Dict:
	"""
        return {
                'batch_size_per_gpu':int,
                'data_layer': None
                }
	
	@staticmethod
	def get_optional_params():
		return {
				'logdir': str,
				'num_gpus': int,
				'gpu_ids': list,
				'load_model': str,
				'save_summaries_steps': None,
				'print_loss_steps': None,
				'save_checkpoint_steps': None,
				'num_checkpoints': int,
				'restore_best_checkpoints': bool,
				'eval_steps': int,
				'finetune': bool,
				'eval_batch_size_per_gpu': int,
				
				'random_seed': int,
				'num_epochs': int,
				'max_steps': int,
				
				'data_layer_params': dict,
				'optimizer': None,
				'optimizer_params': dict,
				'initializer': None,
				'initializer_params': dict,
				'regularizer': None,
				'regularizer_params': dict,
				'dtype': [tf.float16, tf.float32, 'mixed'],
				'lr_policy': None,
				'lr_policy_params': dict,
				'max_grad_norm': float,
				'loss_scaling': None,
				'loss_scaling_params': dict,
				'summaries': list,
				'lm_vocal_file': str,
				'processed_data_folder': str,
				}
	
	def __init__(self, params, mode="train"):
		"""
		params: dict
		mode: train - all parts of the graph will be built (model loss optimizer)
			  eval - (model loss)
		"""
		
		check_params(params, self.get_required_params(), self.get_optional_params())
		
		self._params = copy.deepcopy(params)
		
		#parameter checks
		self._mode = mode
		self._interactive = False
		
		if self._mode not in ["train", "infer", "eval"]:
			raise ValueError("Mode has to be one of ["train", "eval"]")
		
		if "max_steps" in params and "num_epochs" in params:
			raise ValueError("You can't provide both of them")
			
		if mode == "train":
			if "max_steps" not in params and "num_epochs" not in params:
				raise ValueError("For the training mode, either max_steps and num_epochs has to be provided")
				
		none_list = ["print_samples_steps", "print_loss_steps", "save_checkpoint_steps", "save_summaries_steps"]
		for param in none_list:
			if param not in self._params:
				self._params[params] = None
		
		self._params["num_checkpoints"] = self._params.get("num_checkpoints", 5)
		self._params["finetune"] = self._params.get("finetune", False)
		self._params["load_model"] = self._params.get("load_model", None)
		self._params["eval_batch_size_per_gpu"] = self._params.get("eval_batch_size_per_gpu", self._params["batch_size_per_gpu"])
		
		# checking that freq of samples and loss are aligned
		s_fr = self._params["print_samples_steps"]
		l_fr = self._params["print_loss_steps"]
		
		if s_fr is not None and l_fr is not None and s_fr % l_fr != 0:
			raise ValueError("print_sample_steps has to be the multiple of print_loss_steps")
		
		if "data_type" not in self._params:
			self._params["data_type"] = tf.float32
		
		dl_params = self._params.get("data_layer_params", {})
		
		if mode == "train":
			dl_params["batch_size"] = self._params["batch_size_per_gpu"]
		else:
			dl_params["batch_size"] = self._params["eval_batch_size_per_gpu"]
		
		if "lm_vocal_file" in self._params:
			dl_params["lm_vocal_file"] = self._params["lm_vocal_file"]
		
		if "processed_data_folder" in self._params:
			dl_params["processed_data_folder"] = self._params["processed_data_folder"]
		
		dl_params["mode"] = self._mode
		
		if "gpu_ids" in self._params:
			self._gpu_ids = self._params["gpu_ids"]
		elif "num_gpus" in self._params:
			self._gpu_ids = range(self._params["num_gpus"])
		else:
			raise ValueError("Either gpu_ids or num_gpus has to be specified in the config")
		
		self._data_layers = []
		for worker_id in range(self.num_gpus):
			self._data_layers.append(self._params["data_layer"](
				params = dl_params, model = self,
				num_workers = self.num_gpus, worker_id = worker_id))
		
		if self._mode = "train":
			if "max_steps" in self._params:
				slef._last_step = self._params["max_steps"]
				self._step_in_epoch = None
			else:
				# doing a few steps if data size is not divisible by the batch size
				self._step_in_epoch = self.get_data_layer().get_size_in_samples() // self.get_data_layer().params["batch_size"]
				
				if self._step_in_epoch is None:
					raise ValueError("The data_layer is not compatible with epoch execution")
				
				self._step_in_epoch //= self.num_gpus
				self._step_in_epoch //= self._params.get("iter_size", 1)
				
				if self._step_in_epoch == 0:
					raise ValueError("Overall batch size is too big for this dataset")
				self._last_step = self._params["num_epochs"] * self._step_in_epoch
				
		self._outputs = [None] * self.num_gpus
		
		self.loss = None
		self.train_op = None
		self.eval_losses = None
		self._num_objects_per_step = None
		self.skip_update_ph = None
		
	def compile(self, force_var_refuse = False, checkpoint = False):
	"""
	Tensorflow graph is build here.
	"""
		if "initializer" not in self.params:
			initializer = None
		else:
			init_dict = self.params.get("initializer_params",{})
			initializer = self.params["initializer"](**init_dict)
		
		losses = []
		for gpu_cnt, gpu_id in enumerate(self._gpu_ids):
			with tf.device("/gpu:{}".format(gpu_id)), tf.variable_scope(
			name_or_scope = tf.get_variable_scope(), reuse = force_var_reuse or (gpu_cnt > 0),
			initializer = initializer,
			dtype = self.get_tf_dtype()):
				
				deco_print("Building graph on GPU:{}".format(gpu_id))
				
				self.get_data_layer(gpu_cnt).build_graph()
				input_tensors = self.get_data_layer(gpu_cnt).input_tensors
				
				loss, self._outputs[gpu_cnt] = self._build_forward_pass_graph(input_tensors, gpu_id = gpu_cnt)
				if self._outputs[gpu_cnt] is not None and not isinstance(self._outputs[gpu_cnt], list):
					raise ValueError("Decoder outputs have to be either None or list")
				if self._mode == "train" or self._mode == "eval":
					losses.append(loss)
			
			
		
		
		

