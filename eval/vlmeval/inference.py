import datetime

import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.smp import *
from vlmeval.utils import TSVDataset, track_progress_rich

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--nproc", type=int, default=4, required=True)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

# Only API model is accepted


def infer_data_api(work_dir, model_name, dataset_name, index_set, api_nproc=4, **kwargs):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset = TSVDataset(dataset_name)
    data = dataset.data
    data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name](
        **kwargs) if isinstance(model_name, str) else model_name
    is_api = getattr(model, 'is_api', False)
    assert is_api

    lt, indices = len(data), list(data['index'])
    structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(image_path=struct['image'], prompt=struct['text'],
                    dataset=dataset_name) for struct in structs]

    inference_results = track_progress_rich(
        gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    for idx, text in zip(indices, inference_results):
        assert (res[idx] == text if idx in res else True)
        res[idx] = text
    return res


def infer_data(model_name, work_dir, dataset_name, out_file, verbose=False, api_nproc=4, **kwargs):
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        dataset = TSVDataset(dataset_name)
    if world_size > 1:
        dist.barrier()
    dataset = TSVDataset(dataset_name)

    indices = list(range(rank, len(dataset), world_size))
    lt = len(indices)
    data = dataset.data.iloc[indices]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        return
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name](
        **kwargs) if isinstance(model_name, str) else model_name
    is_llm = getattr(model, 'is_llm', False)

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(work_dir=work_dir, model_name=model_name, dataset_name=dataset_name, index_set=set(
            indices), api_nproc=api_nproc, **kwargs)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        dump(res, out_file)
        return model_name

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i], for_llm=is_llm)

        response = model.generate(
            prompt=struct['text'], image_path=struct['image'], dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    dump(res, out_file)
    return model


def infer_data_job(model, work_dir, model_name, dataset_name, verbose=False, api_nproc=4, ignore_failed=False, **kwargs):
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')
    rank, world_size = get_rank_and_world_size()
    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    if not osp.exists(result_file):
        model = infer_data(model, work_dir=work_dir, dataset_name=dataset_name,
                           out_file=out_file, verbose=verbose, api_nproc=api_nproc, **kwargs)
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            data_all = {}
            for i in range(world_size):
                data_all.update(load(tmpl.format(i)))

            data = TSVDataset(dataset_name).data
            assert len(data_all) == len(data)
            data['prediction'] = [str(data_all[x]) for x in data['index']]
            data.pop('image')

            dump(data, result_file)
            for i in range(world_size):
                os.remove(tmpl.format(i))
        return model
    else:
        data = load(result_file)
        failed_set = []
        data['prediction'] = [str(x) for x in data['prediction']]
        for idx, pred in zip(data['index'], data['prediction']):
            if FAIL_MSG in str(pred):
                failed_set.append(idx)
        if len(failed_set) and (not ignore_failed):
            print(
                f'{len(failed_set)} records failed in the original result file {result_file}. ')
            assert rank == 0 and world_size == 1
            failed_set = set(failed_set)
            answer_map = {x: y for x, y in zip(
                data['index'], data['prediction'])}
            res = infer_data_api(work_dir, model_name,
                                 dataset_name, failed_set, api_nproc=api_nproc)
            answer_map.update(res)
            data['prediction'] = [str(answer_map[x]) for x in data['index']]
            dump(data, result_file)
        return model_name
