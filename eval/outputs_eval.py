import json
import nltk
import numpy as np
from rouge import Rouge
from bert_score import score as bert_score
from pycocoevalcap.cider.cider import Cider
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_metrics(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # 初始化工具
    rouge = Rouge()
    cider = Cider()
    smoother = SmoothingFunction()

    refs = [item['input']['reference_text'] for item in data['results']]
    hyps = [item['output']['generated_text'] for item in data['results']]

    # BLEU-4
    bleu_scores = []
    for ref, hyp in zip(refs, hyps):
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        score = sentence_bleu([ref_tokens], hyp_tokens,
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smoother.method1)
        bleu_scores.append(score)

    # ROUGE-L
    rouge_scores = [s['rouge-l']['f'] for s in rouge.get_scores(hyps, refs)]

    # 修正后的CIDEr计算
    cider_scorer = Cider()
    dataset = {i: [r] for i, r in enumerate(refs)}
    hypotheses = {i: [h] for i, h in enumerate(hyps)}
    _, cider_all = cider_scorer.compute_score(dataset, hypotheses)
    
    # 处理不同返回类型
    if isinstance(cider_all, np.ndarray):
        cider_scores = cider_all.tolist()
    elif isinstance(cider_all, float):
        cider_scores = [cider_all] * len(refs)
    else:
        cider_scores = [0.0] * len(refs)

    # BERTScore
    bert_p, bert_r, bert_f1 = bert_score(hyps, refs, lang='en', verbose=True)
    bert_p = bert_p.numpy().tolist()
    bert_r = bert_r.numpy().tolist()
    bert_f1 = bert_f1.numpy().tolist()

    # 填充结果
    for i, item in enumerate(data['results']):
        item['evaluation_scores']['text_metrics']['BLEU']['bleu_4'] = bleu_scores[i]
        item['evaluation_scores']['text_metrics']['ROUGE']['rouge_l']['f'] = rouge_scores[i]
        item['evaluation_scores']['text_metrics']['CIDEr']['cider'] = cider_scores[i]
        item['evaluation_scores']['text_metrics']['BERTScore'].update({
            "precision": bert_p[i],
            "recall": bert_r[i],
            "f1": bert_f1[i]
        })

    # 计算平均值
    data['aggregated_scores']['text_metrics'] = {
        "BLEU-4_mean": np.mean(bleu_scores),
        "ROUGE-L_f1_mean": np.mean(rouge_scores),
        "CIDEr_mean": np.mean(cider_scores),
        "BERTScore_f1_mean": np.mean(bert_f1)
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 使用示例
    calculate_metrics(
        input_path="results/vl2_noft_outouts.json",
        output_path="eval_results/vl2_noft_test.json"
    )