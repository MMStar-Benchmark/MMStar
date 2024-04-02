import copy
import os.path as osp
import warnings

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from vlmeval.smp import get_cache_path, splitlen


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class NousYi:

    INSTALL_REQ = True

    def __init__(self,
                 model_path='NousResearch/Nous-Hermes-2-Yi-34B',
                 in_context=False,
                 **kwargs):
        self.is_llm = True
        self.in_context = in_context
        if self.in_context:
            self.messages = [
                [
                    {"role": "system", "content": "You are a helpful assistant. If you can't judge the answer, give your guess as to the most likely answer. Just give the answer, don't say anything beyond that, no analysis required."},
                    {"role": "user", "content": 'Question: Which one is the correct caption of this image?\nOptions:\nA. The man is waving his hand.\nB. The man is sitting by the edge of the bathtub, gazing at the woman who is lying in the bathtub taking a bath.\nC. The boy is carrying the smiling girl on his back.\nD. The two dirty-faced children turn around and gaze intently.\nPlease select the correct answer from the options above.\nAnswer with the option\'s letter from the given choices directly, such as answer letter \'A\' only.'},
                    {"role": "assistant", "content": 'B'},
                    {"role": "user", "content": 'Question: What\'s the function of the demonstrated object?\nOptions:\nA. oepn the door\nB. drink water\nC. carry personal belongings\nD. exercise\nPlease select the correct answer from the options above.\nAnswer with the option\'s letter from the given choices directly, such as answer letter \'A\' only.'},
                    {"role": "assistant", "content": 'D'}
                ]
            ]
        else:
            self.messages = []

        if splitlen(model_path, '/') == 2 and not osp.exists(model_path):
            if get_cache_path(model_path) is None:
                snapshot_download(repo_id=model_path)

        disable_torch_init()

        load_kwargs = {"device_map": "auto"}
        load_kwargs["torch_dtype"] = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **load_kwargs
        )
        self.model.eval()

        kwargs_default = dict(
            do_sample=kwargs.get('do_sample', False),
            temperature=kwargs.get('temperature', 0),
            num_beams=kwargs.get('num_beams', 1),
            conv_mode='mpt',
            top_p=kwargs.get('num_beams', None),
            max_new_tokens=kwargs.get('max_new_tokens', 1024))
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. ")

        self.stop_str = '<|im_end|>'

    def generate(self, image_path, prompt, dataset=None):
        from .utils import KeywordsStoppingCriteria

        qs = prompt
        if not self.in_context:
            messages = []
        else:
            messages = copy.deepcopy(self.messages[0])
        messages.append({"role": "user", "content": qs})
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True if self.in_context else False).cuda()

        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                do_sample=self.kwargs['do_sample'],
                temperature=self.kwargs['temperature'],
                top_p=self.kwargs['top_p'],
                num_beams=self.kwargs['num_beams'],
                stopping_criteria=[stopping_criteria],
                max_new_tokens=self.kwargs['max_new_tokens'],
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(self.stop_str):
            outputs = outputs[: -len(self.stop_str)]
        outputs = outputs.strip()
        return outputs
