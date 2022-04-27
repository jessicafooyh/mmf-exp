import random

from mmf.common.sample import Sample
from mmf.datasets.builders.coco import COCODataset
from .dataset import VCRDataset
import os


class MaskedVCRDataset(VCRDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_vcr"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        # if self._use_features:
        #     features = self.features_db[idx]
        #     if hasattr(self, "transformer_bbox_processor"):
        #         features["image_info_0"] = self.transformer_bbox_processor(
        #             features["image_info_0"]
        #         )

        #     if self.config.get("use_image_feature_masks", False):
        #         current_sample.update(
        #             {
        #                 "image_labels": self.masked_region_processor(
        #                     features["image_feature_0"]
        #                 )
        #             }
        #         )

        #     current_sample.update(features)
        # else:
            # image_path = str(sample_info["image_name"]) + ".jpg"
            # current_sample.image = self.image_db.from_path(image_path)["images"][0]

        image_path = sample_info["img_fn"]
        try:
            current_sample.image = self.image_db.from_path(image_path)["images"][0]  
        except FileNotFoundError:
            print('unable to find file: ', image_path)
            head, tail = os.path.split(image_path)
            image_path = os.path.join(head[:-1] + '_', tail)
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self._add_masked_caption(sample_info, current_sample)
        return current_sample

    def _add_masked_caption(self, sample_info, current_sample):

        answers = sample_info["answer_choices"]
        rationales = sample_info["rationale_choices"]
        captions = []
        correct_caption_index = len(rationales) * sample_info["answer_label"] + sample_info["rationale_label"] - 1
        for ans in answers:
            for rat in rationales:
                ans_list = [ans_tok[0]+1 if isinstance(ans_tok, list) else ans_tok for ans_tok in ans]
                ans_str = ' '.join([str(elem) for elem in ans_list])
                rat_list = [rat_tok[0]+1 if isinstance(rat_tok, list) else rat_tok for rat_tok in rat]
                rat_str = ' '.join([str(elem) for elem in rat_list])
                captions.append(ans_str + ' ' + rat_str)

        # captions = sample_info["captions"]
        image_id = sample_info["img_id"]
        num_captions = len(captions)

        selected_caption_index = random.randint(0, num_captions - 1)

        # other_caption_indices = [
        #     i for i in range(num_captions) if i != selected_caption_index
        # ]
        selected_caption = captions[correct_caption_index]
        other_caption = None
        is_correct = -1

        if self._two_sentence:
            if random.random() > self._two_sentence_probability:
                other_caption = captions[selected_caption_index]
                is_correct = False
            else:
                other_caption = captions[correct_caption_index]
                is_correct = True
        elif self._false_caption:
            if random.random() < self._false_caption_probability:
                selected_caption = captions[selected_caption_index]
                is_correct = False
            else:
                is_correct = True

        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_b": other_caption,
                "is_correct": is_correct,
            }
        )
        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample

