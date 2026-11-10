#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""ScanQA eval runner with GeoGate v0 pruning (LLaVA-3D).

Same protocol as third_party/LLaVA-3D llava/eval/model_scanqa.py v2 settings
(prompt suffix already in question file, greedy), plus:
  --prune-layer / --keep-ratio : GeoGate v0 config (-1 disables -> baseline)
  per-sample wall-clock latency recorded into the answers file.

Run from third_party/LLaVA-3D with adaptive_token on PYTHONPATH.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_videos,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model

from adaptive_token.adaptive_policy import question_keep_ratio
from adaptive_token.geogate_v0 import (
    enable_auto_span,
    enable_geogate,
    set_keep_ratio,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--question-file', required=True)
    ap.add_argument('--video-folder', required=True)
    ap.add_argument('--answers-file', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--conv-mode', default='llava_v1')
    ap.add_argument('--prune-layer', type=int, default=-1)
    ap.add_argument('--keep-ratio', type=float, default=0.10)
    ap.add_argument('--max-keep-ratio', type=float, default=0.50)
    ap.add_argument('--scorer', default='attn',
                    help='attn | stable_attn | random | prompt_mean | inverse_attn')
    ap.add_argument('--coverage-weight', type=float, default=0.25)
    ap.add_argument('--prune-mode', choices=('keep_topk', 'delete_topk'),
                    default='keep_topk')
    ap.add_argument('--delete-n', type=int, default=0)
    ap.add_argument('--coarse-groups', type=int, default=0)
    ap.add_argument('--geo-beta', type=float, default=0.0)
    ap.add_argument('--geo-grouping', action='store_true')
    ap.add_argument('--dynamic-k', action='store_true',
                    help='Instruction-conditioned dynamic K')
    ap.add_argument('--adaptive-onset', action='store_true',
                    help='Detect the readout onset from attention concentration')
    ap.add_argument('--stability-window', type=int, default=3)
    ap.add_argument('--onset-min-layer', type=int, default=10)
    ap.add_argument('--onset-max-layer', type=int, default=18)
    ap.add_argument('--onset-patience', type=int, default=2)
    ap.add_argument('--onset-min-alpha', type=float, default=0.10)
    ap.add_argument('--onset-max-coverage', type=float, default=0.12)
    ap.add_argument('--stages', type=str, default='',
                    help='multi-stage pyramid, e.g. "8:0.5,16:0.5,24:0.5"')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path),
        device_map='cuda:0')
    model.eval()
    stages = None
    if args.stages:
        stages = {int(x.split(':')[0]): float(x.split(':')[1])
                  for x in args.stages.split(',')}
    pruning_enabled = args.prune_layer >= 0 or stages or args.adaptive_onset
    if pruning_enabled:
        enable_geogate(model, prune_layer=args.prune_layer,
                       keep_ratio=args.keep_ratio, coarse_groups=args.coarse_groups,
                       geo_beta=args.geo_beta, geo_grouping=args.geo_grouping,
                       stages=stages,
                       scorer='stable_attn' if args.adaptive_onset else args.scorer,
                       adaptive_onset=args.adaptive_onset,
                       coverage_weight=args.coverage_weight,
                       stability_window=args.stability_window,
                       min_onset_layer=args.onset_min_layer,
                       max_onset_layer=args.onset_max_layer,
                       onset_patience=args.onset_patience,
                       onset_min_alpha=args.onset_min_alpha,
                       onset_max_coverage=args.onset_max_coverage,
                       prune_mode=args.prune_mode,
                       delete_n=args.delete_n)
        enable_auto_span(model, IMAGE_TOKEN_INDEX)
        if args.geo_beta > 0 or args.geo_grouping or args.scorer in ('coverage_attn', 'stable_coverage'):
            from adaptive_token.geo_coords import attach_coord_hook
            attach_coord_hook(model)
        print(f'GeoGate ON: layer={args.prune_layer} keep={args.keep_ratio} '
              f'coarse_groups={args.coarse_groups} geo_beta={args.geo_beta} '
              f'geo_grouping={args.geo_grouping} adaptive_onset={args.adaptive_onset}')

    questions = json.load(open(args.question_file))
    ans_file = open(args.answers_file, 'w')

    for line in tqdm(questions):
        qs = DEFAULT_IMAGE_TOKEN + '\n' + line['text']
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        videos = process_videos(f"{args.video_folder}/{line['video']}",
                                processor['video'], mode='random',
                                device=model.device, text=line['text'])
        kw = {k: videos[k].to(model.device, dtype=torch.bfloat16)
              for k in ('images', 'depths', 'poses', 'intrinsics')}

        used_ratio = args.keep_ratio
        if pruning_enabled:
            if args.dynamic_k:
                used_ratio = question_keep_ratio(line['text'], args.keep_ratio,
                                                  max_ratio=args.max_keep_ratio)
                set_keep_ratio(model, used_ratio)
            else:
                set_keep_ratio(model, args.keep_ratio)
                used_ratio = args.keep_ratio

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids, images=kw['images'], depths=kw['depths'],
                poses=kw['poses'], intrinsics=kw['intrinsics'], image_sizes=None,
                do_sample=False, temperature=0, num_beams=1,
                max_new_tokens=512, use_cache=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        ans_file.write(json.dumps({
            'question_id': line['question_id'], 'prompt': line['text'],
            'text': text, 'latency_s': round(dt, 3),
            'keep_ratio': round(float(used_ratio), 4),
            'onset_layer': getattr(model.model, '_gg_onset_layer', None),
            'adaptive_onset': bool(args.adaptive_onset),
            'seed': int(args.seed),
            'prune_mode': args.prune_mode,
            'delete_n': int(args.delete_n),
            'model_id': 'LLaVA-3D-7B+adaptive_geogate'
            if args.adaptive_onset else
            ('LLaVA-3D-7B+geogate' if pruning_enabled else 'LLaVA-3D-7B'),
        }) + '\n')
        ans_file.flush()
    ans_file.close()


if __name__ == '__main__':
    main()
