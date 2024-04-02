from functools import partial

from .llm import LLaMA2, NousYi
from .vlm import LLaVA_Next

models = {
    'LLaMA2-7B': partial(LLaMA2, model_path='NousResearch/Nous-Hermes-2-Yi-34B'),
    'Nous_Yi_34B': partial(NousYi, model_path='NousResearch/Nous-Hermes-2-Yi-34B'),
    'llava_next_yi_34b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-34b-hf'),
}

supported_VLM = {}
for model_set in [models]:
    supported_VLM.update(model_set)
