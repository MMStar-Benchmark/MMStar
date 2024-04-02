from ..smp import listinstr

dataset_URLs = {
    'MMStar': 'https://huggingface.co/datasets/Lin-Chen/MMStar/resolve/main/MMStar.tsv'
}

img_root_map = {k: k for k in dataset_URLs}
img_root_map.update({
    'MMStar': 'MMStar'
})

assert set(dataset_URLs) == set(img_root_map)


def DATASET_TYPE(dataset):
    dataset = dataset.lower()
    if listinstr(['mmstar'], dataset):
        return 'multi-choice'
    else:
        raise ValueError


def abbr2full(s):
    datasets = [x for x in img_root_map]
    ins = [s in d for d in datasets]
    if sum(ins) == 1:
        for d in datasets:
            if s in d:
                return d
    else:
        return None
