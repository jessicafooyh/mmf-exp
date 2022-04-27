# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.coco import MaskedCOCOBuilder
from .builder import VCRBuilder

from .masked_dataset import MaskedVCRDataset


@registry.register_builder("masked_vcr")
class MaskedVCRBuilder(VCRBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_vcr"
        self.set_dataset_class(MaskedVCRDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/vcr/masked.yaml"
