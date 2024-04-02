from vlmeval.smp import *
from copy import deepcopy


def MMStar_eval(eval_file):
    MMStar_score_l2 = {
        'coarse perception': {
            'image scene and topic': 0,
            'image style & quality': 0,
            'image emotion': 0
        },
        'fine-grained perception': {
            'object counting': 0,
            'recognition': 0,
            'localization': 0
        },
        'instance reasoning': {
            'single-instance reasoning': 0,
            'cross-instance attribute reasoning': 0,
            'cross-instance relation reasoning': 0
        },
        'logical reasoning': {
            'code & sequence reasoning': 0,
            'diagram reasoning': 0,
            'common reasoning': 0
        },
        'science & technology': {
            'biology & chemistry & physics': 0,
            'electronics & energy & mechanical eng.': 0,
            'geography & earth science & agriculture': 0
        },
        'math': {
            'geometry': 0,
            'numeric commonsense and calculation': 0,
            'statistical reasoning': 0
        },
    }
    MMStar_counter = deepcopy(MMStar_score_l2)
    logger = get_logger('Evaluation')

    data = load(eval_file)
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    for i in tqdm(range(len(lines))):
        line = lines[i]
        predict = str(line['prediction'])
        answers = str(line['answer'])
        ori_bench = str(line['bench'])
        category = str(line['category'])
        l2_category = str(line['l2_category'])
        MMStar_counter[category][l2_category] += 1

        answer = answers.lower().strip().replace('\n', ' ')
        predict = predict.lower().strip().replace('\n', ' ')
        # if ori_bench == 'MathVista' and answer not in ['a', 'b', 'c', 'd']:
        #     if answer in predict:
        #         MMStar_score_l2[category][l2_category] += 1
        # else:
        try:
            if answer == predict[0]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0] == '(' and answer == predict[1]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:7] == 'option ' and answer == predict[7]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:14] == 'the answer is ' and answer == predict[14]:
                MMStar_score_l2[category][l2_category] += 1
        except Exception as e:
            pass

    MMStar_score = {}
    MMStar_score['final score'] = 0
    for k, v in MMStar_score_l2.items():
        MMStar_score[k] = 0
        for l2_k, l2_v in v.items():
            MMStar_score[f'{k}({l2_k})'] = float(l2_v) / \
                float(MMStar_counter[k][l2_k])
            MMStar_score[k] += l2_v
        MMStar_score['final score'] += MMStar_score[k]
        MMStar_score[k] = float(MMStar_score[k]) / 250.0
    MMStar_score['final score'] = float(MMStar_score['final score']) / 1500.0

    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(MMStar_score, score_pth)
    logger.info(
        f'MMStar_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Score: ')
    for key, value in MMStar_score.items():
        logger.info('{}:{}'.format(key, value))
