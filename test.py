from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
import pdb
import os

#===============================================================lds
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_image)
        target = output[0, target_class]
        
        # Backward pass
        target.backward()

        # Compute Grad-CAM
        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()
        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling

        cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)
        cam = np.maximum(cam, 0)  # ReLU to keep positive values
        cam = cv2.resize(cam[0], (input_image.shape[2], input_image.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        return cam
#==================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="demo", help="restult path name")
    parser.add_argument("--maskResultDir", type=str, default="mask_result", help="result dir name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default=r"./data_demo", help="test data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice','f1','pre','rec'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_l", help="sam model_type")  
    parser.add_argument("--sam_checkpoint", type=str, default="", help="sam checkpoint") 
    parser.add_argument("--boxes_prompt", type=bool, default=False, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=8, help="iter num")
    parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=True, help="save reslut")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    if args.resume is not None:
        args.sam_checkpoint = None
    return args



def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )

    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad

def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings, interm_embeddings=None ):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,

            #-------------------------------------------------------
            hq_token_only=False,
            interm_embeddings = interm_embeddings


            #--------------------------------------------------------
        )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append_1(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    model = sam_model_registry[args.model_type](args)
    model.to(args.device)
    model.eval()


    if args.resume is not None:
         with open(args.resume, "rb") as f:
             checkpoint = torch.load(f)
             model.load_state_dict(checkpoint['model'], weights_only=True)
             #optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
             print(f"*******load {args.resume}")
        #checkpoint = torch.load(args.resume)
        #model.load_state_dict(checkpoint['model'],strict=True)

    criterion = FocalDiceloss_IoULoss()

    # 'occlusion', 'salt_and_pepper', 'gaussian' 'apply_motion_blur'  'add_poisson_noise'
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, 
                                mode='test', requires_name=True, point_num=args.point_num, 
                                return_ori_mask=True, prompt_path=args.prompt_path,input_mask=False,
                                # noise_type='apply_motion_blur', 
                                # noise_params={'kernel_size':5}  #apply_motion_blur noise_params={'kernel_size':5}  gaussian  noise_params={'mean': 0, 'var': 10}    #    salt_and_pepper  noise_params={'amount':0.02, 'salt_vs_pepper':0.5}  # 参数调整
                                )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)


    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0].split('.')[0] + '.png' 
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            image_embeddings,interm_embeddings = model.image_encoder(batched_input["image"])
        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,interm_embeddings)
            points_show = None
        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name, f"{args.maskResultDir}")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]

            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, interm_embeddings)
                if iter != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())
        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]


    test_iter_metrics = [metric / l for metric in test_iter_metrics]

    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}

    average_loss = np.mean(test_loss)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
