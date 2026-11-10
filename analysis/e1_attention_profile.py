#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""E1: layer-wise spatial-token attention redundancy profile for LLaVA-3D.

For each sample: run one forward pass with output_attentions, split tokens into
{system, spatial(3D patch), prompt, answer-gen position}, and record per layer:
  - attention mass flowing from the last prompt token to spatial tokens
  - per-spatial-token received attention (from all tokens) -> concentration stats
  - top-p coverage: fraction of spatial tokens needed to cover 50%/90% of the
    attention mass the generation position pays to spatial tokens

Outputs a JSON with per-layer aggregates over N samples (figures drawn locally).

Usage:
  python e1_attention_profile.py --model /path/to/LLaVA-3D-7B \
    --question-file <llava-3d-scanqa_val_question.json> --video-folder /path/to/scene-data \
    --num-samples 50 --out e1_profile.json
"""
import argparse
import json

import numpy as np
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path
from llava.model.builder import load_pretrained_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--question-file', required=True)
    ap.add_argument('--video-folder', required=True)
    ap.add_argument('--num-samples', type=int, default=50)
    ap.add_argument('--per-scene-cap', type=int, default=1,
                    help='max questions per scene (raise for larger runs)')
    ap.add_argument('--dump-per-sample', action='store_true',
                    help='also dump per-sample per-layer records for offline grouping')
    ap.add_argument('--conv-mode', default='llava_v1')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    tokenizer, model, processor, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model), device_map='cuda:0')
    model.eval()

    questions = json.load(open(args.question_file))
    # spread samples over scenes, up to per-scene cap
    seen, samples = {}, []
    for q in questions:
        c = seen.get(q['video'], 0)
        if c >= args.per_scene_cap:
            continue
        seen[q['video']] = c + 1
        samples.append(q)
        if len(samples) >= args.num_samples:
            break

    n_layers = model.config.num_hidden_layers
    agg = {
        'frac_attn_to_spatial': [[] for _ in range(n_layers)],   # gen-pos attn mass on spatial
        'top50_coverage': [[] for _ in range(n_layers)],         # frac of spatial tokens for 50% mass
        'top90_coverage': [[] for _ in range(n_layers)],
        'gini': [[] for _ in range(n_layers)],                   # inequality of received attention
    }
    n_spatial_list = []
    per_sample = []

    for si, line in enumerate(samples):
        qs = DEFAULT_IMAGE_TOKEN + '\n' + line['text']
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        videos = process_videos(f"{args.video_folder}/{line['video']}", processor['video'],
                                mode='random', device=model.device, text=line['text'])
        kw = {k: videos[k].to(model.device, dtype=torch.bfloat16)
              for k in ('images', 'depths', 'poses', 'intrinsics')}

        # locate spatial token span: image placeholder position + inserted length
        img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0, 0].item()
        with torch.inference_mode():
            (_, position_ids, attention_mask, _, inputs_embeds, _
             ) = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None,
                kw['images'], kw['depths'], kw['poses'], kw['intrinsics'],
                None, image_sizes=None)
            total_len = inputs_embeds.shape[1]
            n_text_after = input_ids.shape[1] - img_pos - 1
            n_spatial = total_len - img_pos - n_text_after
            sp0, sp1 = img_pos, img_pos + n_spatial   # [sp0, sp1)
            out = model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                              position_ids=position_ids, output_attentions=True,
                              use_cache=False, return_dict=True)

        n_spatial_list.append(n_spatial)
        rec = {'question_id': line['question_id'], 'video': line['video'],
               'text': line['text'], 'n_spatial': n_spatial,
               'frac': [], 'top50': [], 'top90': [], 'gini': []} if args.dump_per_sample else None
        for li, attn in enumerate(out.attentions):     # (1, H, T, T)
            a = attn[0].float().mean(dim=0)            # (T, T) head-avg
            gen_row = a[-1]                            # last position attends...
            to_sp = gen_row[sp0:sp1]
            frac = (to_sp.sum() / gen_row.sum()).item()
            agg['frac_attn_to_spatial'][li].append(frac)
            if rec is not None:
                rec['frac'].append(round(frac, 5))
            if to_sp.sum() > 0:
                w = torch.sort(to_sp, descending=True).values
                c = torch.cumsum(w, 0) / w.sum()
                t50 = ((c < 0.5).sum().item() + 1) / n_spatial
                t90 = ((c < 0.9).sum().item() + 1) / n_spatial
                agg['top50_coverage'][li].append(t50)
                agg['top90_coverage'][li].append(t90)
                if rec is not None:
                    rec['top50'].append(round(t50, 5))
                    rec['top90'].append(round(t90, 5))
            recv = a[:, sp0:sp1].sum(dim=0)            # received attention per spatial token
            r, _ = torch.sort(recv)
            i = torch.arange(1, len(r) + 1, device=r.device, dtype=r.dtype)
            gini = ((2 * i - len(r) - 1) * r).sum() / (len(r) * r.sum() + 1e-9)
            agg['gini'][li].append(gini.item())
            if rec is not None:
                rec['gini'].append(round(gini.item(), 5))
        if rec is not None:
            per_sample.append(rec)
        del out
        torch.cuda.empty_cache()
        print(f'[{si + 1}/{len(samples)}] {line["video"]} n_spatial={n_spatial}', flush=True)

    result = {
        'n_samples': len(samples),
        'n_layers': n_layers,
        'n_spatial_mean': float(np.mean(n_spatial_list)),
        'layers': [{
            'layer': li,
            'frac_attn_to_spatial': float(np.mean(agg['frac_attn_to_spatial'][li])),
            'top50_coverage': float(np.mean(agg['top50_coverage'][li])),
            'top90_coverage': float(np.mean(agg['top90_coverage'][li])),
            'gini': float(np.mean(agg['gini'][li])),
        } for li in range(n_layers)],
    }
    if args.dump_per_sample:
        result['per_sample'] = per_sample
    json.dump(result, open(args.out, 'w'), indent=1)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
