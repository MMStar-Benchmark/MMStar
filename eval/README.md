# Evaluation Guidelines

We provide detailed instructions for evaluation.

## Environment

You should prepare the environment as follows:

``` bash
conda create -n mmstar python=3.10
conda activate mmstar

cd eval
pip install -e .
```

## View Data

We release the val set of MMStar for benchmarking on the leader board, which contains 1,500 visual-indispensible evaluation samples.
You can download and view the dataset from Huggingface by the following command:

```python
from datasets import load_dataset

dataset = load_dataset("Lin-Chen/MMStar", "val")

# take a close look of evaluation samples
print(dataset["val"][0])
dataset["val"][0]['image']  # display the image
```

## Evaluation

We develop an easy-to-use evaluation pipeline based on the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) repository.
This pipeline is designed to evaluate the accuracy of various LLMs and LVLMs on our MMStar benchmark, along with measuring LVLMs'
multi-modal gain and multi-modal leakage.

We take evaluating LLaVA-Next-34B as an example.

(1) You can evaluate LLaVA-Next with accessing images and obtain $S_v$ as follows:

```bash
bash scripts/run.sh

# we show the detailed commands
torchrun --nproc-per-node=$NUM_GPUS --master_port ${MASTER_PORT} run.py \
    --verbose \
    --data MMStar \
    --model llava_next_yi_34b \
    --max-new-tokens 32 \
    --gen-mode mm
```

(2) Then, you should set `--gen-mode` to `to` for switch to the text-only mode for evaluating LVLMs without accessing images and obtain $S_{wv}$.

```bash
torchrun --nproc-per-node=$NUM_GPUS --master_port ${MASTER_PORT} run.py \
    --verbose \
    --data MMStar \
    --model llava_next_yi_34b \
    --max-new-tokens 32 \
    --gen-mode to
```

(3) You can directly change the `--model` argument to `Nous_Yi_34B` for evaluating the performance of LLaVA-Next's LLM base and obtain $S_t$.

```bash
torchrun --nproc-per-node=$NUM_GPUS --master_port ${MASTER_PORT} run.py \
    --verbose \
    --data MMStar \
    --model Nous_Yi_34B \
    --max-new-tokens 32 \
    --gen-mode mm
```

Finally, you can find outputs and detailed results in the `eval/outputs` directory. Moreover, you can utilize equations $MG=S_v-S_{wv}$ and $ML=max(0,S_{wv}-S_t)$.

We warmly invite you to submit the results of your LVLMs to our leaderboard. Please note that to thoroughly evaluate your own LVLM, you are required to provide us with three result files in xlsx format. These should include the results of your LVLM with visual input, the results of your LVLM without visual input, and the results of your original LLM base without visual input. We have provided a submission format in the `submits` folder. After completing the aforementioned steps, please contact us via chlin@mail.ustc.edu.cn to submit your results and to update the leaderboard.
