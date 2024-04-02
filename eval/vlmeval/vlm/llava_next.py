import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from ..smp import *
from ..utils import DATASET_TYPE, CustomPrompt


class LLaVA_Next(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self,
                 model_pth='llava-hf/llava-v1.6-vicuna-7b-hf',
                 **kwargs):

        self.model_pth = model_pth
        if '34b' in model_pth.lower():
            self.processor = LlavaNextProcessor.from_pretrained(
                self.model_pth, use_fast=False)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_pth)

        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_pth, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model = model.eval()
        self.model = model.cuda()
        self.gen_mode = kwargs.pop('gen_mode', 'mm')
        kwargs_default = dict(do_sample=False, temperature=0,
                              max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def apply_prompt_template(self, prompt):
        model_pth = self.model_pth.lower()
        if 'mistral' in model_pth:
            s = f'[INST] <image>\n {prompt} [/INST]'
        elif 'vicuna' in model_pth:
            s = (
                'A chat between a curious human and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                f'USER: <image>\n{prompt} ASSISTANT:'
            )
        elif '34b' in model_pth:
            s = (
                f'<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|>'
                '<|im_start|>assistant\n'
            )
        else:
            raise NotImplementedError(
                f'Prompt template for {model_pth} not implemented.')
        if self.gen_mode != 'mm':
            s = s.replace('<image>\n', '')
        return s

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if (
            'hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(
                prompt) else '\nAnswer the question directly.'
        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path)
        prompt_wtmpl = self.apply_prompt_template(prompt)
        inputs = self.processor(prompt_wtmpl, image if self.gen_mode == 'mm' else None,
                                return_tensors='pt').to('cuda')

        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        if '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()

        return answer
