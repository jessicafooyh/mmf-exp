# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json

from mmf.datasets.databases.annotation_database import AnnotationDatabase
from mmf.utils.general import get_absolute_path


class VCRAnnotationDatabase(AnnotationDatabase):
    def __init__(self, config, path, *args, **kwargs):
        path = path.split(",")
        super().__init__(config, path, *args, **kwargs)

    def load_annotation_db(self, path):
        with open(path) as f:
            annotations = [json.loads(line) for line in f]

        data = []
        
        for annotation in annotations:
            answer = annotation["answer_choices"][annotation['answer_label']]
            answer_no_obj = [] ## remove the '[0]' etc object labels from the answers and replace w the object class name
            for ans in answer:
                if isinstance(ans, list):
                    obj = annotation["objects"][ans[0]]
                    answer_no_obj.append(obj)
                else:
                    answer_no_obj.append(ans)
            answer = answer_no_obj
                    
            rationale = annotation["rationale_choices"][annotation['rationale_label']]
            rationale_no_obj = [] ## remove the '[0]' etc object labels from the answers and replace w the object class name
            for rat in rationale:
                if isinstance(rat, list):
                    obj = annotation["objects"][rat[0]]
                    rationale_no_obj.append(obj)
                else:
                    rationale_no_obj.append(rat)
            rationale = rationale_no_obj

            # just concatenate canswer with rationale
            answer.append("because")
            annotation["answers"] = answer + rationale ## not sure if answers should be the correct key
            data.append(copy.deepcopy(annotation))
            
        
        self.data = data


        # # Expect two paths, one to questions and one to annotations
        # assert (
        #     len(path) == 2
        # ), "OKVQA requires 2 paths; one to questions and one to annotations"

        # with open(path[0]) as f:
        #     path_0 = json.load(f)
        # with open(path[1]) as f:
        #     path_1 = json.load(f)

        # if "annotations" in path_0:
        #     annotations = path_0
        #     questions = path_1
        # else:
        #     annotations = path_1
        #     questions = path_0

        # # Convert to linear format
        # data = []
        # question_dict = {}
        # for question in questions["questions"]:
        #     question_dict[question["question_id"]] = question["question"]

        # for annotation in annotations["annotations"]:
        #     annotation["question"] = question_dict[annotation["question_id"]]
        #     answers = []
        #     for answer in annotation["answers"]:
        #         answers.append(answer["answer"])
        #     annotation["answers"] = answers
        #     data.append(copy.deepcopy(annotation))

        # self.data = data
