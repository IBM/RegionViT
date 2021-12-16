import argparse
import os
import torch
import copy
import math


parser = argparse.ArgumentParser(description='Convert RegionViT cls model for detection')
parser.add_argument('model', type=str, metavar='PATH',
                    help='The pretrained model.')
parser.add_argument('--ori_window_size', '--ows', default=[7], type=int, nargs='+')
parser.add_argument('--new_window_size', '--nws', default=[7], type=int, nargs='+')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='The output path for the converted model.')


def convert(model_path, ori_window_size, new_window_size):

    if len(ori_window_size) == 1:
        ori_window_size = ori_window_size * 4
    if len(new_window_size) == 1:
        new_window_size = new_window_size * 4

    model = torch.load(model_path, map_location='cpu')
    state_dict_name = 'model'
    model = {state_dict_name: model[state_dict_name]}
    new_state_dict = copy.deepcopy(model)
    
    # convert the regional token projection
    v = model[state_dict_name]['cls_token.proj.weight']
    if 4 * new_window_size[0] != v.shape[-2] or 4 * new_window_size[0] != v.shape[-1]:
        print(f"Converting Regional Tokens from ({v.shape[-2]}, {v.shape[-1]}) to ({4 * new_window_size[0]}, {4 * new_window_size[0]}) ")
        o = torch.nn.functional.interpolate(v, size=(4 * new_window_size[0], 4 * new_window_size[0]), mode='bicubic')
        new_state_dict[state_dict_name]['cls_token.proj.weight'] = o

    # convert other stages
    for stage_id, (nws, ows) in enumerate(zip(new_window_size, ori_window_size)):
        if nws == ows:
            continue

        print(f"At stage {stage_id}, resize the rel pos bias: original size {ows}, new size: {nws}", flush=True)
        for k, v in model[state_dict_name].items():
            if '.rel_pos' not in k and '.{stage_id}.' not in k:
                continue

            ori_d = int(math.sqrt(v.shape[1]))
            ori_rel_pos = v.reshape(v.shape[0], ori_d, ori_d)
            if ori_d != 2 * nws - 1:
                print(f"{k}, Converting Rel Pos from ({ori_d}, {ori_d}) to ({2 * nws - 1}, {2 * nws - 1}) ")
                new_rel_pos = torch.nn.functional.interpolate(ori_rel_pos.unsqueeze(0), size=(2 * nws - 1, 2 * nws - 1), mode='bicubic').flatten(2).squeeze(0)
                new_state_dict[state_dict_name][k] = new_rel_pos


    new_state_dict_ = {state_dict_name: {}}
    for k, v in new_state_dict[state_dict_name].items():
        new_k = 'backbone.bottom_up.model.' + k
        new_state_dict_[state_dict_name][new_k] = v

    return new_state_dict_


def main():
    args = parser.parse_args()
    model = convert(args.model, args.ori_window_size, args.new_window_size)
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        name = os.path.join(args.output, os.path.basename(args.model).split(".")[0] + f'_detection.pth')
    else:
        name = os.path.join(os.path.dirname(args.model), os.path.basename(args.model).split(".")[0] + f'_detection.pth')
    print(f"Save model to {name}")
    torch.save(model, name, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
