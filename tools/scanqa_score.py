#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Java-free ScanQA scorer for LLaVA-3D predictions.

Metrics: EM@1, refined EM@1 (substring match), BLEU-1..4, ROUGE-L, CIDEr.
(METEOR/SPICE require Java and are not computed by this script.)

Answer cleaning follows llava/eval/scanqa_evaluator.py (LEO convention).

Usage:
  python scanqa_score.py --pred answers.jsonl --gt llava3d_scanqa_val_answer.json
"""
import argparse
import json
import re

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$', '', data)
    data = re.sub('^[ ]+', '', data)
    data = re.sub(' {2,}', ' ', data)
    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç', 'c', data)
    data = re.sub('’', '\'', data)
    data = re.sub(r'\bletf\b', 'left', data)
    data = re.sub(r'\blet\b', 'left', data)
    data = re.sub(r'\btehre\b', 'there', data)
    data = re.sub(r'\brigth\b', 'right', data)
    data = re.sub(r'\brght\b', 'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b', 'TV', data)
    data = re.sub(r'\bchai\b', 'chair', data)
    data = re.sub(r'\bwasing\b', 'washing', data)
    data = re.sub(r'\bwaslked\b', 'walked', data)
    data = re.sub(r'\boclock\b', 'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b', 'o\'clock', data)
    for digit, word in enumerate(['zero', 'one', 'two', 'three', 'four', 'five',
                                  'six', 'seven', 'eight', 'nine', 'ten']):
        data = re.sub(rf'\b{digit}\b', word, data)
    data = re.sub(r'\bnone\b', 'zero', data)
    return data


def answer_match(pred, gts):
    if pred in gts:
        return 1, 1
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or \
           ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


def load_json_or_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        raw = handle.read().strip()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get('answers', rows.get('data', [rows]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='jsonl from model_scanqa.py')
    ap.add_argument('--gt', required=True, help='[{question_id, text: [answers]}]')
    args = ap.parse_args()

    preds = load_json_or_jsonl(args.pred)
    gts = {g['question_id']: g['text'] for g in load_json_or_jsonl(args.gt)}

    acc = refined = 0
    tmp_preds, tmp_targets = {}, {}
    n = 0
    for p in preds:
        qid = p['question_id']
        if qid not in gts:
            continue
        pred_answer = clean_answer(p['text'])
        refs = [clean_answer(a) for a in gts[qid]]
        a, r = answer_match(pred_answer, refs)
        acc += a
        refined += r
        tmp_preds[qid] = [{'caption': pred_answer}]
        tmp_targets[qid] = [{'caption': c.replace('\n', ' ').strip()} for c in refs]
        n += 1

    print(f'n={n}')
    print(f'EM@1        = {acc / n * 100:.2f}')
    print(f'EM@1(refined)= {refined / n * 100:.2f}')

    # java-free tokenization (PTBTokenizer needs java): lowercase + strip punct + split
    def tok(caps):
        out = {}
        for k, v in caps.items():
            s = v[0]['caption'].lower()
            s = re.sub(r'[^a-z0-9\s]', ' ', s)
            out[k] = [' '.join(s.split())]
        return out

    tok_preds = tok(tmp_preds)
    tok_targets = {k: [' '.join(re.sub(r'[^a-z0-9\s]', ' ', c['caption'].lower()).split())
                       for c in v] for k, v in tmp_targets.items()}
    for scorer, names in [(Bleu(4), ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']),
                          (Rouge(), 'ROUGE-L'), (Cider(), 'CIDEr')]:
        score, _ = scorer.compute_score(tok_targets, tok_preds)
        if isinstance(names, list):
            for nm, s in zip(names, score):
                print(f'{nm:12s}= {s * 100:.2f}')
        else:
            print(f'{names:12s}= {score * 100:.2f}')


if __name__ == '__main__':
    main()
