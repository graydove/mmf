import json
import logging
import math
import os
import zipfile
from collections import Counter

# from mmf.common.constants import CLEVR_DOWNLOAD_URL
from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.builders.dpc.dataset import DPCDataset
from mmf.utils.download import download
from mmf.utils.general import get_mmf_root
from mmf.datasets.builders.vqa2 import VQA2Builder


logger = logging.getLogger(__name__)


@registry.register_builder("dpc")
class DPCBuilder(VQA2Builder):
	def __init__(self):
		super().__init__("dpc")
		self.dataset_name = "dpc"
		self.set_dataset_class(DPCDataset)
		# self.dataset_class = DPCDataset

	# TODO: Deprecate this method and move configuration updates directly to processors
	def update_registry_for_model(self, config):
		registry.register(
			self.dataset_name + "_text_vocab_size",
			self.dataset.text_processor.get_vocab_size(),
		)

		if hasattr(self.dataset, "answer_processor"):
			registry.register(
				self.dataset_name + "_num_final_outputs",
				self.dataset.answer_processor.get_vocab_size(),
			)

			registry.register(
				self.dataset_name + "_answer_processor", self.dataset.answer_processor
			)

	@classmethod
	def config_path(cls):
		return "configs/datasets/dpc/defaults.yaml"

	def load(self, config, *args, **kwargs):
		annotation_style = config.get("annotation_style", self.dataset_name)
		if annotation_style == "textcaps":
			self.dataset_class = TextCapsDataset

		dataset = super().load(config, *args, **kwargs)
		dataset.dataset_name = self.dataset_name
		return dataset
