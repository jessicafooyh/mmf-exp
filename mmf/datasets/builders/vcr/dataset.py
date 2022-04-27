# Copyright (c) Facebook, Inc. and its affiliates.
from concurrent.futures import process
import logging

import torch
import tqdm
from string import punctuation
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import is_main
import json

logger = logging.getLogger(__name__)


class VCRDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "vcr"
        super().__init__(name, config, dataset_type, index=imdb_file_index)

        self._should_fast_read = self.config.get("fast_read", False)
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

    def init_processors(self):
        super().init_processors()
        if not self._use_features:
            self.image_db.transform = self.image_processor

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            logger.info(
                f"Starting to fast read {self.dataset_name} {self.dataset_type} "
                + "dataset"
            )
            self.cache = {}
            for idx in tqdm.tqdm(
                range(len(self.annotation_db)), miniters=100, disable=not is_main()
            ):
                self.cache[idx] = self.load_item(idx)

    def __getitem__(self, idx):
        if self._should_fast_read is True and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        # text_processor_argument = {
        #     "tokens": sample_info["question"],
        #     "text": sample_info["question_orig"],
        # }

        # processed_question = self.text_processor(text_processor_argument)

        # current_sample.text = processed_question["text"]
        # if "input_ids" in processed_question:
        #     current_sample.update(processed_question)

        # current_sample.question_id = torch.tensor(
        #     sample_info["question_number"], dtype=torch.int
        # )

        # if isinstance(sample_info["img_id"], int):
        #     current_sample.image_id = torch.tensor(
        #         sample_info["img_id"], dtype=torch.int
        #     )
        # else:
        #     current_sample.image_id = sample_info["img_id"]

        # if "question" in sample_info:
        #     current_sample.text_len = torch.tensor(
        #         len(sample_info["question"]), dtype=torch.int
        #     )
        
        # image_path = sample_info["img_fn"] ## + ".jpg"
        # current_sample.image = self.image_db.from_path(image_path)["images"][0]  ##TODO: CHECK THIS!!!
        # # # Add details for OCR like OCR bbox, vectors, tokens here
        # # current_sample = self.add_ocr_details(sample_info, current_sample) ##TODO: CHECK THIS!!!
        # # # Depending on whether we are using soft copy this can add
        # # # dynamic answer space
        # # current_sample = self.add_answer_info(sample_info, current_sample) 

        answer_rationale_str = sample_info["answer_orig"].replace('.', '') + ' because ' + sample_info["rationale_orig"].replace('.', '') 

        print('uh', self.text_processor)

        processed_caption = self.text_processor({"text": answer_rationale_str})
        current_sample.text = processed_caption["text"]
        current_sample.caption_len = torch.tensor(
            len(processed_caption["text"]), dtype=torch.int
        )

        if isinstance(sample_info["img_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["img_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["img_id"]
        
        print('helpppp', processed_caption["text"])

        current_sample.answers = torch.stack([processed_caption["text"]])

        return current_sample

    # def add_obj_details(self, sample_info, sample):
    #     ## TODO: not sure if bboxes can be added like ocr tokens
    #     bbox_file_path = sample_info["metadata_fn"]
    #     with open(bbox_file_path) as f:
    #         bbox_info = json.load(f)
        

        

    def add_ocr_details(self, sample_info, sample):
        if self.use_ocr:
            # Preprocess OCR tokens
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]
            ]
            # Get embeddings for tokens
            context = self.context_processor({"tokens": ocr_tokens})
            sample.context = context["text"]
            sample.context_tokens = context["tokens"]
            sample.context_feature_0 = context["text"]
            sample.context_info_0 = Sample()
            sample.context_info_0.max_features = context["length"]

            order_vectors = torch.eye(len(sample.context_tokens))
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors

        if self.use_ocr_info and "ocr_info" in sample_info:
            sample.ocr_bbox = self.bbox_processor({"info": sample_info["ocr_info"]})[
                "bbox"
            ]

        return sample

    def add_answer_info(self, sample_info, sample):
        
        if "answer_choices" in sample_info:
            answers = self._concat_answer_rationale(sample_info)
            answer_processor_arg = {"answers": answers}

            # if self.use_ocr:
            #     answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

            # sample.answers = processed_soft_copy_answers["answers"]
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample
    
    def _concat_answer_rationale(self, sample_info):

        answer = sample_info["answer_choices"][sample_info['answer_label']]
        answer_no_obj = [] ## remove the '[0]' etc object labels from the answers and replace w the object class name
        for ans in answer:
            if isinstance(ans, list):
                obj = sample_info["objects"][ans[0]]
                answer_no_obj.append(obj)
            else:
                answer_no_obj.append(ans)
        answer = answer_no_obj
                
        rationale = sample_info["rationale_choices"][sample_info['rationale_label']]
        rationale_no_obj = [] ## remove the '[0]' etc object labels from the answers and replace w the object class name
        for rat in rationale:
            if isinstance(rat, list):
                obj = sample_info["objects"][rat[0]]
                rationale_no_obj.append(obj)
            else:
                rationale_no_obj.append(rat)
        rationale = rationale_no_obj

        # just concatenate canswer with rationale
        answer.append("because")

        return answer + rationale ## not sure if answers should be the correct key

    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)
            # actual_answer = report.answers[idx]

            predictions.append(
                {
                    "question_id": question_id.item(),
                    "answer": answer,
                    # "actual_answers": actual_answer,
                    # "question_tokens": report.question_tokens[idx],
                    # "image_id": report.image_id[idx].item()
                }
            )

        return predictions
