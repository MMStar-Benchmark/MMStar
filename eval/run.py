import argparse
import os
import os.path as osp

import torch
import torch.distributed as dist
from vlmeval.evaluate import MMStar_eval
from vlmeval.inference import infer_data_job
from vlmeval.smp import (datetime, get_local_rank_and_world_size, get_logger,
                         get_rank_and_world_size)
from vlmeval.utils import abbr2full, dataset_URLs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--work-dir", type=str,
                        default='outputs', help="select the output directory")
    parser.add_argument("--mode", type=str, default='all',
                        choices=['all', 'infer'])
    parser.add_argument("--nproc", type=int, default=4,
                        help="Parallel API calling")
    parser.add_argument("--ignore", action='store_true',
                        help="Ignore failed indices. ")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--prefetch", action='store_true')
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="max new tokens for generation")
    parser.add_argument("--gen-mode", type=str, default='mm', choices=['mm', 'to'],
                        help="text-only or multi-modal mode for generation")
    parser.add_argument("--in-context", action='store_true',
                        help="use context for llm generation")
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')

    args = parse_args()
    assert len(args.data), "--data should be a list of data files"

    rank, world_size = get_local_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend='nccl', timeout=datetime.timedelta(seconds=5400))

    rank, world_size = get_rank_and_world_size()
    for _, model_name in enumerate(args.model):
        model = None

        pred_root = osp.join(args.work_dir, model_name)
        if args.gen_mode != 'mm':
            pred_root += '_text-only'
        elif args.in_context:
            pred_root += '_in-context'
        os.makedirs(pred_root, exist_ok=True)

        for i, dataset_name in enumerate(args.data):
            if dataset_name not in dataset_URLs:
                dataset_name = abbr2full(dataset_name)

            if dataset_name not in dataset_URLs:
                logger.error(f'Unknown dataset: {dataset_name}. ')
                continue

            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'

            if model is None:
                model = model_name  # which is only a name

            gen_args = {'max_new_tokens': args.max_new_tokens} if 'GPT' not in args.model else {
                'max_tokens': args.max_new_tokens}
            if args.gen_mode != 'mm':
                gen_args.update({'gen_mode': args.gen_mode})
            if args.in_context:
                gen_args.update({'in_context': args.in_context})
            model = infer_data_job(model, work_dir=pred_root, model_name=model_name, dataset_name=dataset_name,
                                   verbose=args.verbose, api_nproc=args.nproc, ignore_failed=args.ignore, **gen_args)

            if rank == 0 and args.mode == 'all':
                MMStar_eval(result_file)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
