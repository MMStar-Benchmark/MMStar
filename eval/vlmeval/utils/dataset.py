import hashlib

import pandas as pd

from ..smp import *
from .custom_prompt import CustomPrompt
from .dataset_config import DATASET_TYPE, dataset_URLs


def isliststr(s):
    return (s[0] == '[') and (s[-1] == ']')


class TSVDataset(CustomPrompt):

    def __init__(self, dataset='MMStar', skip_noimg=True):

        self.data_root = LMUDataRoot()
        assert osp.exists(self.data_root)

        self.dataset = dataset
        self.dataset_type = DATASET_TYPE(dataset)

        if dataset in dataset_URLs:
            url = dataset_URLs[dataset]
            file_name = url.split('/')[-1]
            data_path = osp.join(self.data_root, file_name)

            if osp.exists(data_path):
                pass
            else:
                warnings.warn('The dataset tsv is not downloaded')
                download_file(url, data_path)
        else:
            data_path = osp.join(self.data_root, dataset + '.tsv')
            assert osp.exists(data_path)

        data = load(data_path)
        self.skip_noimg = skip_noimg
        if skip_noimg:
            data = data[~pd.isna(data['image'])]

        data['index'] = [str(x) for x in data['index']]
        data['image'] = [str(x) for x in data['image']]

        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        data['image'] = [
            eval(image_map[k]) if isliststr(image_map[k]) else image_map[k]
            for k in data['index']
        ]
        if 'image_path' in data:
            data['image_path'] = [
                eval(pths) if isliststr(pths) else pths for pths in data['image_path']
            ]
        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data

    def __len__(self):
        return len(self.data)

    def build_prompt(self, line, dataset=None, for_llm=False):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)

        prompt = line['question']
        if DATASET_TYPE(dataset) == 'multi-choice':
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if (
                'hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                prompt += 'Please select the correct answer from the options above. \n'
                if for_llm:
                    prompt += "Answer with the option's letter from the given choices directly, such as answer letter 'A' only. \n"
        elif DATASET_TYPE(dataset) == 'Y/N' and for_llm:
            prompt += 'Answer the question using a single word or phrase. \n'

        return dict(image=tgt_path, text=prompt)

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mmqa_display(line)
