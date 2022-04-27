# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.vcr.dataset import VCRDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("vcr")
class VCRBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="vcr", dataset_class=VCRDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/vcr/defaults.yaml"
