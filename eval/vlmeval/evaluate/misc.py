import os

from vlmeval.api import OpenAIWrapper


def build_judge(version, **kwargs):
    model_map = {
        'gpt-4-turbo': 'gpt-4-1106-preview',
        'gpt-4-0613': 'gpt-4-0613',
        'gpt-4-0314': 'gpt-4-0314',
        'chatgpt-1106': 'gpt-3.5-turbo-1106',
        'chatgpt-0613': 'gpt-3.5-turbo-0613'
    }
    model_version = model_map[version]

    model = OpenAIWrapper(model_version, **kwargs)
    return model
