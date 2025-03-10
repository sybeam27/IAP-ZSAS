import sys
import os
# sys.path.append('./SegmentAnything/GroundingDINO')
# sys.path.append('./SegmentAnything/SAM')
# sys.path.append('./SegmentAnything')
# sys.path.append('./llama3')

sys.path.append(os.path.abspath('./SegmentAnything/GroundingDINO'))
sys.path.append(os.path.abspath('./SegmentAnything/SAM'))
sys.path.append(os.path.abspath('./SegmentAnything'))
sys.path.append(os.path.abspath('./llama3'))

import random
from typing import List

import cv2
import re
import numpy as np
import requests
import stringprep
import json
import torch
import itertools
import torchvision
import torchvision.transforms as TS
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionInpaintPipeline
from io import BytesIO
from matplotlib import pyplot as plt
from torchvision.ops import box_convert
import torchvision.ops as ops
# !pip install spacy
# !python -m spacy download en_core_web_sm
import spacy
# from llama import Llama, Dialog
from ram import inference_ram
from ram.models import ram
import supervision as sv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from segment_anything import SamPredictor, build_sam, build_sam_hq
import SegmentAnything.SAA as SegmentAnyAnomaly
from SegmentAnything.datasets import *
from SegmentAnything.utils.csv_utils import *
from SegmentAnything.utils.eval_utils import *
from SegmentAnything.utils.metrics import *
from SegmentAnything.utils.training_utils import *
import SegmentAnything.GroundingDINO.groundingdino.datasets.transforms as T
from SegmentAnything.GroundingDINO.groundingdino.models import build_model
from SegmentAnything.GroundingDINO.groundingdino.util import box_ops
from SegmentAnything.GroundingDINO.groundingdino.util.inference import annotate
from SegmentAnything.GroundingDINO.groundingdino.util.slconfig import SLConfig
from SegmentAnything.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from SegmentAnything.GroundingDINO.groundingdino.util.inference import predict
from collections import OrderedDict

# from gdino import GroundingDINOAPIWrapper, visualize

def normalize(scores):
    max_value = np.max(scores)
    min_value = np.min(scores)
    
    # min 값과 max 값이 모두 0인 경우
    if (min_value == 0 and max_value == 0) or (min_value == max_value):
        return np.zeros_like(scores)

    norml_scores = (scores - min_value) / (max_value - min_value)
    return norml_scores

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)  # 수정된 부분
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
        
def draw_box(box, draw, label):
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    line_width = int(max(4, min(20, 0.006 * max(draw.im.size))))

    # Draw rectangle
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=line_width)

    if label:
        font_path = os.path.join(
            cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font_size = int(max(12, min(60, 0.02*max(draw.im.size))))
        font = ImageFont.truetype(font_path, size=font_size)
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white", font=font)

def load_image(image_path, gt_path):
    # load image
    raw_image = Image.open(image_path).convert("RGB")  # load image
    source_image = np.asarray(raw_image)
    
    H, W = raw_image.size[1], raw_image.size[0]
    if gt_path == image_path:
        gt_image = np.zeros((H, W), dtype=np.uint8)
    else:
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    _, gt_binary = cv2.threshold(gt_image, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
    gt_mask = torch.tensor(gt_binary, dtype=torch.float)

    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ram_transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(),
        normalize
    ])

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ram_image = raw_image.resize((384, 384))
    ram_image = ram_transform(ram_image).unsqueeze(0)

    image, _ = transform(raw_image, None)  # 3, h, w

    return image, source_image, raw_image, ram_image, gt_image, gt_binary, gt_mask

def process_load_image(image_path, gt_path):
    # load image
    mask_image = cv2.imread(image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    
    raw_image = Image.open(image_path).convert("RGB")
    source_image = np.asarray(raw_image)
    
    H, W = raw_image.size[1], raw_image.size[0]
    if gt_path == image_path:
        gt_image = np.zeros((H, W), dtype=np.uint8)
    else:
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    _, gt_binary = cv2.threshold(gt_image, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
    gt_mask = torch.tensor(gt_binary, dtype=torch.float)

    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ram_transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(),
        normalize
    ])

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ram_image = raw_image.resize((384, 384))
    ram_image = ram_transform(ram_image).unsqueeze(0)

    image, _ = transform(raw_image, None)  # 3, h, w

    return image, mask_image, source_image, raw_image, ram_image, gt_image, gt_binary, gt_mask

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device, with_logits=True):
    # print(caption)
    if isinstance(caption, list):
        caption = ' '.join(caption)
    caption = caption.lower()
    caption = caption.strip()
    # caption = caption.replace(",", ".")
    if not caption.endswith("."):
        caption = caption + "."
    # print('caption :', caption)
    # print(caption)
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()

    filt_mask = (logits_filt.max(dim=1)[0] > box_threshold) 
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):

        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer)

        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())
    # print(pred_phrases)
    
    return boxes_filt, pred_phrases, torch.Tensor(scores)

def dilate_bounding_box(x_min, y_min, x_max, y_max, scale=1.0):
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    width = x_max - x_min
    height = y_max - y_min
    
    new_width = width * scale
    new_height = height * scale
    
    new_x_min = cx - new_width / 2
    new_y_min = cy - new_height / 2
    new_x_max = cx + new_width / 2
    new_y_max = cy + new_height / 2
    
    return new_x_min, new_y_min, new_x_max, new_y_max

def dilate_segment_mask(mask, kernel_size=5, iterations=1):
    """
    SAM에서 출력된 segmentation mask를 넓히는 함수

    :param mask: 이진 세그멘테이션 마스크 (numpy array)
    :param kernel_size: 커널 크기, 기본값은 5
    :param iterations: 팽창 연산 반복 횟수, 기본값은 1
    :return: 넓어진 세그멘테이션 마스크 (numpy array)
    """
    
    # 팽창 연산 커널
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return dilated_mask

def GroundedSAM(grounding_dino_model, sam_model, 
                image, source_image, raw_image, tags, device,
                box_threshold, text_threshold, iou_threshold, size_threshold=None, filt_db=None, filt_ds=None):
    
    while True:
        boxes_filt, pred_phrases, scores = get_grounding_output(grounding_dino_model, image, 
                                                                tags, box_threshold, text_threshold, device)
        if boxes_filt is not None:  # GroundedSAM 함수가 성공적으로 값을 반환하면 루프 종료
            break

    # run SAM
    sam_model.set_image(source_image)
    size = raw_image.size

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    scores = [scores[idx] for idx in nms_idx]
    
    if size_threshold is not None and len(boxes_filt) > 1:
        box_widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W # x_max - x_min
        box_heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H  # y_max - y_min

        # size_threshold의 각 값을 사용하여 조건에 맞는 인덱스를 찾음
        filt1_idx = torch.nonzero(box_widths < size_threshold[0]).squeeze(1)
        filt2_idx = torch.nonzero(box_heights < size_threshold[1]).squeeze(1)
        combined_indices = torch.cat((filt1_idx, filt2_idx))
        filt_size = torch.unique(combined_indices)

        if len(filt_size) != len(boxes_filt):
            boxes_filt = boxes_filt[filt_size]
            pred_phrases = [pred_phrases[i] for i in filt_size]
            scores = [scores[i] for i in filt_size]

    if filt_db != None:
        for i in range(boxes_filt.size(0)):
            x_min, y_min, x_max, y_max = boxes_filt[i].tolist()
            new_x_min, new_y_min, new_x_max, new_y_max = dilate_bounding_box(x_min, y_min, x_max, y_max, scale=filt_db)
            boxes_filt[i] = torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max])        
        
        boxes_filt[:, [0, 2]] = boxes_filt[:, [0, 2]].clamp(0, W)
        boxes_filt[:, [1, 3]] = boxes_filt[:, [1, 3]].clamp(0, H)
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)
    else:
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)

    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    
    if masks is None:
        masks = boxes_filt
    
    if filt_ds != None:
        for i in range(len(masks)):
            dil = dilate_segment_mask(masks[i][0].cpu().numpy().astype(np.uint8), kernel_size=filt_ds, iterations=1)
            masks[i][0] = torch.tensor(dil > 0)
    
    return masks, boxes_filt, pred_phrases, scores

def get_grounding_output_2(model, image_path, caption, box_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    caption = caption.replace(",", ".")

    prompts = dict(image=image_path, prompt=caption)

    with torch.no_grad():
        results = model.inference(prompts)

    # logits = torch.tensor(results["scores"]).cpu()
    scores = torch.tensor(results["scores"]).cpu().sigmoid()
    boxes = torch.tensor(results["boxes"]).cpu()
    categorys = results["categorys"]

    scores_filt = scores.clone()
    boxes_filt = boxes.clone()
    categorys_filt = categorys.copy()

    filt_mask = (scores_filt > box_threshold)
    scores_filt = scores_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    categorys_filt = list(np.array(categorys_filt)[np.array(filt_mask)])

    # print(f"boxes_filt shape: {boxes_filt.shape}")
    # print(f"boxes_filt content: {boxes_filt}")

    if boxes_filt.dim() == 1 and boxes_filt.numel() % 4 == 0:
        boxes_filt = boxes_filt.view(-1, 4)
    elif boxes_filt.dim() != 2 or boxes_filt.size(1) != 4:
        raise ValueError(f"Expected boxes_filt to be [num_boxes, 4], but got shape {boxes_filt.shape}")

    pred_phrases = []
    for logit, box, category in zip(scores_filt, boxes_filt, categorys_filt):
        
        if with_logits:
            pred_phrases.append(category + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(category)

    return boxes_filt, pred_phrases, scores_filt

def GroundedSAM_2(grounding_dino_model, sam_model, 
                  source_image, raw_image, image_path, 
                  box_threshold2, tags, device, iou_threshold, size_threshold=None, filt_db=None, filt_ds=None, filt_bb=1):
    
    boxes_filt, pred_phrases, scores = get_grounding_output_2(grounding_dino_model, image_path, tags, box_threshold2)
    print("GroundingDINO1.5 finished")

    # run SAM
    sam_model.set_image(source_image)
    size = raw_image.size

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    scores = [scores[idx] for idx in nms_idx]

    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    if size_threshold is not None and len(boxes_filt) > 1:
        box_widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W # x_max - x_min
        box_heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H  # y_max - y_min

        # size_threshold의 각 값을 사용하여 조건에 맞는 인덱스를 찾음
        filt1_idx = torch.nonzero(box_widths < size_threshold[0]).squeeze(1)
        filt2_idx = torch.nonzero(box_heights < size_threshold[1]).squeeze(1)
        combined_indices = torch.cat((filt1_idx, filt2_idx))
        filt_size = torch.unique(combined_indices)

        if len(filt_size) != len(boxes_filt):
            boxes_filt = boxes_filt[filt_size]
            pred_phrases = [pred_phrases[i] for i in filt_size]
            scores = [scores[i] for i in filt_size]

    if filt_db != None:
        for i in range(boxes_filt.size(0)):
            x_min, y_min, x_max, y_max = boxes_filt[i].tolist()
            new_x_min, new_y_min, new_x_max, new_y_max = dilate_bounding_box(x_min, y_min, x_max, y_max, scale=filt_db)
            boxes_filt[i] = torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max])        
        
        boxes_filt[:, [0, 2]] = boxes_filt[:, [0, 2]].clamp(0, W)
        boxes_filt[:, [1, 3]] = boxes_filt[:, [1, 3]].clamp(0, H)
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)
    else:
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)

    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    
    if masks is None:
        masks = boxes_filt
    
    if filt_ds != None:
        for i in range(len(masks)):
            dil = dilate_segment_mask(masks[i][0].cpu().numpy().astype(np.uint8), kernel_size=filt_ds, iterations=1)
            masks[i][0] = torch.tensor(dil > 0)

    return masks, boxes_filt, pred_phrases, scores

def remove_large_boxes(boxes, image_width, image_height):
    half_width, half_height = image_width / 2, image_height / 2

    mask = (boxes[:, 2] <= half_width) & (boxes[:, 3] <= half_height)
    filtered_boxes = boxes[mask]
    
    return filtered_boxes

def find_largest_box_size(grounding_dino_model, image, raw_image, tags,
                        box_threshold, text_threshold, iou_threshold, device):

    # boxes_filt, pred_phrases, scores = get_grounding_output(
    #     grounding_dino_model, image, tags, box_threshold, text_threshold, device)
    boxes_filt, boxes_score, pred_phrases = predict(grounding_dino_model, image, tags, box_threshold, text_threshold, device, remove_combined = True)

    size = raw_image.size
    H, W = size[1], size[0]

    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    scores = [scores[idx] for idx in nms_idx]

    widths = boxes_filt[:, 2] - boxes_filt[:, 0]
    heights = boxes_filt[:, 3] - boxes_filt[:, 1]

    normalized_widths = widths / W
    normalized_heights = heights / H
    
    largest_width = torch.max(normalized_widths)
    largest_height = torch.max(normalized_heights)
    
    return largest_width.item(), largest_height.item()

def eval_zsas(gt, pred_mask):
    if isinstance(gt, np.ndarray):
        gt_mask_np = gt
    else:
        gt_mask_np = gt.cpu().squeeze(0).numpy()
    
    if isinstance(pred_mask, np.ndarray):
        pred_mask_np = pred_mask
    else:
        pred_mask_np = pred_mask.cpu().squeeze(0).numpy()
    
    # Intersection over Union (IoU)
    intersection = np.logical_and(gt_mask_np, pred_mask_np)
    union = np.logical_or(gt_mask_np, pred_mask_np)
    iou = np.round(np.sum(intersection) / np.sum(union), 2)
    
    # Dice coefficient
    dice_coefficient = np.round((2 * np.sum(intersection)) / (np.sum(gt_mask_np) + np.sum(pred_mask_np)), 2)

    # Accuracy
    accuracy = np.sum(gt_mask_np == pred_mask_np) / gt_mask_np.size

    # Precision
    precision = np.sum(intersection) / np.sum(pred_mask_np)

    # Recall
    recall = np.sum(intersection) / np.sum(gt_mask_np)
    
    # F1 score
    f1_score = (2 * precision * recall) / (precision + recall)

    return iou * 100, dice_coefficient * 100, accuracy * 100, precision * 100, recall * 100, f1_score * 100

def compute_pro_with_padding(pred, gt, padding=0):
    if padding > 0:
        if pred.ndim == 2 and gt.ndim == 2:
            pred = pred[padding:-padding, padding:-padding]
            gt = gt[padding:-padding, padding:-padding]

    intersection = np.sum(pred * gt)
    union = np.sum(gt)
    pro = (intersection / union) * 100 if union > 0 else 0

    return pro

def calculate_max_f1(gt, pred):
    best_f1 = 0
    best_threshold = 0
    thresholds = np.linspace(0, 1, 100)

    for thres in thresholds:
        pred_binary = (pred >= thres).astype(int)
        tp = (gt * pred_binary).sum()
        fp = pred_binary.sum() - tp
        fn = gt.sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thres

    return best_f1, best_threshold

def eval_zsas_last_2(gt, pred, padding=0):   
    ap = average_precision_score(gt.ravel(), pred.ravel())
    f1m, thres = calculate_max_f1(gt, pred)

    pred_binary = (pred >= thres).astype(int)
    pro = compute_pro_with_padding(pred_binary, gt, padding)

    return ap * 100, f1m * 100, pro

def eval_zsas_last(gt, pred):
    # if len(set(gt)) > 1:
    #     # gt_not = 1-int(next(iter(set(gt))))
    #     # gt, pred = np.append(gt, gt_not), np.append(pred, gt_not)
        
    # else:
    #     roc_auc = None
    roc_auc = roc_auc_score(gt, pred)
    ap = average_precision_score(gt, pred)
    f1m, thres = calculate_max_f1(gt, pred)
    
    return roc_auc * 100 , ap * 100, f1m * 100

def paste_cropped_image(back_image, cropped_image, position):
    back_image.paste(cropped_image, position)
    return back_image

def add_word_to_each_item(word_list, word_to_add):
    words = word_list.split(',')
    
    new_words = [word + ' ' + word_to_add for word in words]
    
    result = ','.join(new_words)
    
    return result

def clean_string(s):
    if s is None:
        return ""
    s = s.strip('.,')
    s = s.replace("''", "").replace('""', "")

    s = s.replace("word", "").replace("Word", "")
    s = s.replace("none", "").replace("None", "")

    s = re.sub(r'\bof\b', '', s)
    s = re.sub(r'\bpart\b', '', s)
    
    s = s + ',' + 'abnormal, defect'

    parts = s.split(',')
    cleaned_parts = []
    
    for part in parts:
        words = part.split()
        unique_words = list(dict.fromkeys(words))
        cleaned_parts.append(' '.join(unique_words))

    cleaned_string = ', '.join(cleaned_parts)
    
    if cleaned_string.startswith(','):
        cleaned_string = cleaned_string[1:].strip()
       
    unique_tags = set(cleaned_string.split(', '))
    cleaned_string = ', '.join(unique_tags)

    return cleaned_string.replace('.', "").strip('.,').replace(',','.') #.replace(',','.')

def classification_adjectiveclause_llama(tokenizer, model, tags, main_name, sub_name, device):
    classification_messages = [{"role": "system", "content": "Assistant is always must be listed as words in lowercase format."},                 

    {"role": "user", "content": f"""
                                    Objects recognized in the image include: {tags}.

                                    I would like to divide them according to the degree to which they are related to {main_name} and {sub_name}.
                                    Please classify according to the information below.

                                    1. Please classify nouns related to {main_name} into the Nouns list. They should be listed in order of relevance.
                                    2. Please classify Nouns related to {sub_name} into the Adjectives list.
                                    3. Please delete objects that are not classified above.
                                    4. Outputs a list of each noun and adjective according to the system output format. noun: 'word, word'

                                    The noun list is
                                        Nouns: word,word
                                    The adjective list is
                                        Adjective: word,word
                                    Please save the results in the format
                                    """},]
    nouns_result = ""
    while not nouns_result :
        with torch.no_grad():
            input_ids = tokenizer.apply_chat_template(
                classification_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        classification_response = outputs[0][input_ids.shape[-1]:]

        classification_result = tokenizer.decode(classification_response, skip_special_tokens=True)
        
        nouns_match = re.search(rf'{re.escape("Nouns: ")}(.*)', classification_result)

        nouns_result = nouns_match.group(1).strip() if nouns_match else ""
        print('Rotate until the noun list is filled.')
        print("nouns: ",nouns_result)
        print('-'*100)

    print('Finally Result')
    print("nouns: ",nouns_result)
    nouns_cleaned_string = clean_string(nouns_result)
    if nouns_cleaned_string:
        if main_name not in nouns_cleaned_string:
            nouns_combined_string = nouns_cleaned_string + ',' + main_name
        else:
            nouns_combined_string = nouns_cleaned_string
    else:
        nouns_combined_string = main_name

    print("clean_nouns: ",nouns_combined_string)
    print('-'*100)
    
    llama_tags = ''
    for word in nouns_combined_string.split(',')[:3]:
        adjectives_messages = [{"role": "system", "content": """The assistant should always answer only by listing lowercase words in the following format: 'word, word'."""},
            {"role": "user", "content": f"""Objects recognized in the image include: {word}.
                                            I would like to create an adjective clause before the object tag to find anomaly parts of the recognized object in the image.
                                            
                                            Based on recognized object tags, adjectives or infinitives are converted to adjective clauses, creating a list that accurately specifies only the singular or unique part of the object.
                                            Additionally, adjective clauses must be converted into 5 non-redundant results."""},]
        with torch.no_grad():
            input_ids = tokenizer.apply_chat_template(
                adjectives_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        adjectives_response = outputs[0][input_ids.shape[-1]:]
        adjectives_result = tokenizer.decode(adjectives_response, skip_special_tokens=True)
        print(word, ':',adjectives_result)
        combination_result = add_word_to_each_item(adjectives_result, word)
        print(word, ':',combination_result)
        finally_result = clean_string(combination_result, word)
        print(word, ':', finally_result)

        llama_tags = llama_tags + ',' + finally_result

    if llama_tags.startswith(","):
        llama_tags = llama_tags[1:]

    return llama_tags

def process_object_output(grounding_dino_model, image, tags,
                             box_threshold, text_threshold, raw_img, iou_threshold, device):
    # # Grounding output 얻기
    boxes_filt, pred_phrases, boxes_score = get_grounding_output(
        grounding_dino_model, image, tags, box_threshold, text_threshold, device, with_logits=False)
    
    # boxes_filt, pred_phrases, boxes_score = get_grounding_output_2(
    #     grounding_dino_model, image, tags, box_threshold, text_threshold, device, with_logits=False)

    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    
    # Bounding box 좌표 조정
    boxes_filt = boxes_filt.to(device) 
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * scale_tensor
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # Bounding box를 CPU로 이동
    boxes_filt = boxes_filt.cpu()
    
    # NMS (Non-Maximum Suppression)
    nms_idx = torchvision.ops.nms(boxes_filt, boxes_score, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    boxes_score = [boxes_score[idx] for idx in nms_idx]
    
    # # bounding box 크기가 가장 큰 경우
    # widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W
    # heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H
    # max_value, idx = torch.max(torch.tensor(widths * heights), dim=0)
    
    # bounding box 점수가 가장 큰 경우
    max_value, idx = torch.max(torch.tensor(boxes_score), dim=0)
    
    boxes_filt = boxes_filt[idx].unsqueeze(0)
    pred_phrases = [pred_phrases[idx]]
    boxes_score = [boxes_score[idx]]
    
    widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W
    heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H
    object_size = torch.tensor(widths * heights)
    
    return boxes_filt, pred_phrases, boxes_score, object_size

def process_object_output_2(grounding_dino_model, image, tags,
                             box_threshold, text_threshold, raw_img, iou_threshold, device):

    boxes_filt, boxes_score, pred_phrases = predict(grounding_dino_model, image, tags, box_threshold, text_threshold, device, remove_combined = True)
 
    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    
    # Bounding box 좌표 조정
    boxes_filt = boxes_filt.to(device) 
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * scale_tensor
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # Bounding box를 CPU로 이동
    boxes_filt = boxes_filt.cpu()
    
    # NMS (Non-Maximum Suppression)
    nms_idx = torchvision.ops.nms(boxes_filt, boxes_score, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    boxes_score = [boxes_score[idx] for idx in nms_idx]  
    
    best_boxes_filt = boxes_filt[0].unsqueeze(0)
    widths = (best_boxes_filt[:, 2] - best_boxes_filt[:, 0])/W
    heights = (best_boxes_filt[:, 3] - best_boxes_filt[:, 1])/H
    object_size = torch.tensor(widths * heights)    
    
    return boxes_filt, pred_phrases, boxes_score, object_size

def process_object_output_3(grounding_dino_model, image, tags,
                             box_threshold, text_threshold, raw_img, iou_threshold, device):
    # # Grounding output 얻기
    boxes_filt, pred_phrases, boxes_score = get_grounding_output(
        grounding_dino_model, image, tags, box_threshold, text_threshold, device, with_logits=False)
    
    # boxes_filt, pred_phrases, boxes_score = get_grounding_output_2(
    #     grounding_dino_model, image, tags, box_threshold, text_threshold, device, with_logits=False)

    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    
    # Bounding box 좌표 조정
    boxes_filt = boxes_filt.to(device) 
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * scale_tensor
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # Bounding box를 CPU로 이동
    boxes_filt = boxes_filt.cpu()
    
    # NMS (Non-Maximum Suppression)
    nms_idx = torchvision.ops.nms(boxes_filt, boxes_score, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    boxes_score = [boxes_score[idx] for idx in nms_idx]
    
    # # bounding box 크기가 가장 큰 경우
    widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W
    heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H
    # max_value, idx = torch.max(torch.tensor(widths * heights), dim=0)
    
    # bounding box 점수가 가장 큰 경우
    # max_value, idx = torch.max(torch.tensor(boxes_score), dim=0)
    
    # boxes_filt = boxes_filt[idx].unsqueeze(0)
    # pred_phrases = [pred_phrases[idx]]
    # boxes_score = [boxes_score[idx]]
    
    # widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W
    # heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H
    object_size = torch.tensor(widths * heights)
    
    return boxes_filt, pred_phrases, boxes_score, object_size


def process_anomaly_tags(llama_model, messages, llama_tokenizer, tag):
    
    # messages = [{
    #     "role": "system",
    #     "content": """The assistant should always answer only by listing lowercase words."""
    # },{
    #     "role": "user",
    #     "content": f"""
    #     To identify the anomalous parts of the recognized objects in the image, I would like to extract words.
    #     Write down non-redundant anomalous nouns that can appear in the recognized objects."""
    # }]
    
    with torch.no_grad():
        input_ids = llama_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(llama_model.device)

        terminators = [
            llama_tokenizer.eos_token_id,
            llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = llama_model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=llama_tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    
    response = outputs[0][input_ids.shape[-1]:]
    result = llama_tokenizer.decode(response, skip_special_tokens=True)
    result = result.replace(".", " ")
    adjective_tags = clean_string(result)
    # anomaly_tags = add_word_to_each_item(adjective_tags, tag)
    anomaly_tags = adjective_tags + ', ' + tag
    
    return result, adjective_tags, anomaly_tags.strip('., ').replace(",", ".")

def process_anomaly_tags_2(model, tokenizer, tags):    
    # tags = list(OrderedDict.fromkeys(tags))
    # while len(tags) != 3:
    #     tags += [tags[0]]
    llama_tags = ''
    # print('anomaly generation tags :', tags)
    for object_tag in tags:
        # adjectives_messages = [{"role": "system", "content": """The assistant should always answer only by listing lowercase words in the following format: 'word, word'."""},
        #     {"role": "user", "content": f"""Objects recognized in the image include: {word}.
        #                                     I would like to create an adjective clause before the object tag to find anomaly parts of the recognized object in the image.
                                            
        #                                     Based on recognized object tags, adjectives or infinitives are converted to adjective clauses, creating a list that accurately specifies only the singular or unique part of the object.
        #                                     Additionally, adjective clauses must be converted into 5 non-redundant results."""},]
        adjectives_messages = [{"role": "system", "content": """The assistant should always answer only by listing lowercase words in the following format: 'word, word'"""},
            {"role": "user", "content": f"""I would like to extract words to enter into the object detection model to find abnormal parts of {object_tag}. 
                    Please write down non-redundant anomalous nouns that can appear in {object_tag} images."""},]
        with torch.no_grad():
            input_ids = tokenizer.apply_chat_template(
                adjectives_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        adjectives_response = outputs[0][input_ids.shape[-1]:]
        adjectives_result = tokenizer.decode(adjectives_response, skip_special_tokens=True)
        # print(word, 'Adjective result', ':', adjectives_result)
        clean_result = clean_string(adjectives_result)
        # print(object_tag, 'Clean adjective result ', ':', clean_result)
        # finally_result = add_word_to_each_item(clean_result, object_tag)
        
        # llama_tags = llama_tags + ' . ' + finally_result
        llama_tags = llama_tags + ' . ' + clean_result+ ', '+ object_tag

    # llama_tags = llama_tags.strip('.,').replace(",", ".")
    llama_tags = llama_tags + '.'
    # llama_tags = list(OrderedDict.fromkeys(llama_tags))
    return llama_tags


def process_anomaly_tags_2_sy(model, tokenizer, tags):    
    # tags = list(OrderedDict.fromkeys(tags))
    # while len(tags) != 3:
    #     tags += [tags[0]]
    llama_tags = ''
    # print('anomaly generation tags :', tags)
    for object_tag in tags:
        # adjectives_messages = [{"role": "system", "content": """The assistant should always answer only by listing lowercase words in the following format: 'word, word'."""},
        #     {"role": "user", "content": f"""Objects recognized in the image include: {word}.
        #                                     I would like to create an adjective clause before the object tag to find anomaly parts of the recognized object in the image.
                                            
        #                                     Based on recognized object tags, adjectives or infinitives are converted to adjective clauses, creating a list that accurately specifies only the singular or unique part of the object.
        #                                     Additionally, adjective clauses must be converted into 5 non-redundant results."""},]
        adjectives_messages = [{"role": "system", "content": """The assistant should always answer only by listing lowercase words in the following format: 'word, word'"""},
            {"role": "user", "content": f"""I would like to extract words to enter into the object detection model to find abnormal parts of {object_tag} for industrial anomaly detection. 
                    Please write down non-redundant industrial anomalous nouns that can appear in {object_tag} images."""},]
        with torch.no_grad():
            input_ids = tokenizer.apply_chat_template(
                adjectives_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        adjectives_response = outputs[0][input_ids.shape[-1]:]
        adjectives_result = tokenizer.decode(adjectives_response, skip_special_tokens=True)
        # print(word, 'Adjective result', ':', adjectives_result)
        clean_result = clean_string(adjectives_result)
        # print(object_tag, 'Clean adjective result ', ':', clean_result)
        # finally_result = add_word_to_each_item(clean_result, object_tag)
        
        # llama_tags = llama_tags + ' . ' + finally_result
        llama_tags = llama_tags + ' . ' + clean_result+ ', '+ object_tag

    # llama_tags = llama_tags.strip('.,').replace(",", ".")
    llama_tags = llama_tags + '.'
    # llama_tags = list(OrderedDict.fromkeys(llama_tags))
    return llama_tags

def process_box_output(grounding_dino_model, image, tags,
                             box_threshold, text_threshold, device, raw_img, iou_threshold):
    # # Grounding output 얻기
    boxes_filt, pred_phrases, boxes_score = get_grounding_output(
        grounding_dino_model, image, tags, box_threshold, text_threshold, device, with_logits=False)
    
    # boxes_filt, pred_phrases, boxes_score = get_grounding_output_2(
    #     grounding_dino_model, image, tags, box_threshold, text_threshold, device, with_logits=False)
    
    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    
    # Bounding box 좌표 조정
    boxes_filt = boxes_filt.to(device) 
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * scale_tensor
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # Bounding box를 CPU로 이동
    boxes_filt = boxes_filt.cpu()
    
    # NMS (Non-Maximum Suppression)
    nms_idx = torchvision.ops.nms(boxes_filt, boxes_score, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    boxes_score = [boxes_score[idx] for idx in nms_idx]
    
    return boxes_filt, pred_phrases, boxes_score

def process_box_output_plus(grounding_dino_model, image, tags,
                             box_threshold, text_threshold, device, raw_img, iou_threshold):
    # # Grounding output 얻기
    box_lst, pred_lst, score_lst = [], [], []
    for tag in tags:
        boxes_filt, pred_phrases, boxes_score = get_grounding_output(
            grounding_dino_model, image, tag, box_threshold, text_threshold, device, with_logits=False)
        box_lst.append(boxes_filt)
        pred_lst.append(pred_phrases)
        score_lst.append(boxes_score)

    boxes_filt = torch.cat(box_lst, dim=0)
    pred_phrases = list(itertools.chain(*pred_lst))
    boxes_score = torch.cat(score_lst)
    
    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    
    # Bounding box 좌표 조정
    boxes_filt = boxes_filt.to(device) 
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * scale_tensor
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # Bounding box를 CPU로 이동
    boxes_filt = boxes_filt.cpu()
    
    # NMS (Non-Maximum Suppression)
    nms_idx = torchvision.ops.nms(boxes_filt, boxes_score, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    boxes_score = [boxes_score[idx] for idx in nms_idx]
    
    return boxes_filt, pred_phrases, boxes_score

def process_box_output_2(grounding_dino_model, image, tags,
                             box_threshold, text_threshold, device, raw_img, iou_threshold):
    if isinstance(tags, list):
        tags = ''.join(tags)
    # Grounding output 얻기
    boxes_filt, boxes_score, pred_phrases = predict(grounding_dino_model, image, tags, box_threshold, text_threshold, device, remove_combined = True)
    
    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    
    # Bounding box 좌표 조정
    boxes_filt = boxes_filt.to(device) 
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * scale_tensor
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # Bounding box를 CPU로 이동
    boxes_filt = boxes_filt.cpu()
    
    # NMS (Non-Maximum Suppression)
    nms_idx = torchvision.ops.nms(boxes_filt, boxes_score, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    boxes_score = [boxes_score[idx] for idx in nms_idx]
    
    return boxes_filt, pred_phrases, boxes_score

def process_size_output(raw_img, object_size, boxes_filt, pred_phrases, boxes_score, threshold = 0.9):
    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W
    heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H
    
    filt_idx = torch.nonzero(widths*heights < (object_size * threshold)).squeeze(1)
    filt_size = torch.unique(filt_idx)
    
    if len(filt_size) != 0:
        boxes_filt = boxes_filt[filt_size]
        pred_phrases = [pred_phrases[i] for i in filt_size]
        boxes_score = [boxes_score[i] for i in filt_size]
    else:
        boxes_filt, pred_phrases, boxes_score = None, None, None
    
    return boxes_filt, pred_phrases, boxes_score

def process_anomaly_segmentation(raw_img, src_img, sam_model, boxes_filt, boxes_score, device):
    # 이미지 크기
    H, W = raw_img.size[1], raw_img.size[0]
    
    if boxes_filt is not None and len(boxes_filt) > 0:
        # Clamp box coordinates to image boundaries
        boxes_filt[:, [0, 2]] = boxes_filt[:, [0, 2]].clamp(0, W)
        boxes_filt[:, [1, 3]] = boxes_filt[:, [1, 3]].clamp(0, H)
        
        # Transform boxes to model input size
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)
        
        # Set image for the model
        sam_model.set_image(src_img)
        
        # Predict masks
        masks, masks_score, _ = sam_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # If no masks are predicted, return a default mask
        if masks is None or len(masks) == 0:
            masks = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)
            masks_score = [torch.tensor(0.0, dtype=torch.float32)]
            boxes_score = [torch.tensor(0.0, dtype=torch.float32)]
    else:
        # No boxes provided, return a default mask
        masks = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)
        masks_score = [torch.tensor(0.0, dtype=torch.float32)]
        boxes_score = [torch.tensor(0.0, dtype=torch.float32)]
    
    return masks, masks_score, boxes_score
  
def process_draw_boxes(raw_image, boxes_filt, pred_phrases):
    box_image = raw_image.copy()
    if boxes_filt is not None:
        box_draw = ImageDraw.Draw(box_image)
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, box_draw, label)
    box_image_result = np.array(box_image)
    
    return box_image_result

def process_draw_masks(raw_image, masks):
    H, W = raw_image.size[1], raw_image.size[0]
    mask_image = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
    mask_image_result = np.array(mask_image)
    
    return mask_image_result

def process_extract_object_nouns(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    object_nouns = list(set([token.text for token in doc if token.pos_ == 'NOUN']))
    return ", ".join(object_nouns)
          
def process_specify_resolution(test_imgs, test_scores, test_masks, gt_mask_list, resolution: tuple=(400,400)):
    resize_image = []
    resize_score = []
    resize_mask = []
    resize_gt = []
    for image, score, mask, gt in zip(test_imgs, test_scores, test_masks, gt_mask_list):
        if mask.dtype == np.bool_:
            mask = mask.astype(np.uint8) * 255
        image = cv2.resize(image, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        score = cv2.resize(score, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
        resize_image += [image]
        resize_score += [score]
        resize_mask += [mask]
        resize_gt += [gt]
    return resize_image, resize_score, resize_mask, resize_gt

def process_inpainting(image, image_path, device,
               boxes_filt, scores_filt, pred_phrases, masks, 
               main_name, sub_name, sub_number, 
               inpainting_diff_threshold, filt_ds=None):

    # Set Pipe
    if device.type == 'cpu':
        float_type = torch.float32
    else:
        float_type = torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=float_type,
        ).to(device)

    inpainting_mask = sum(masks[i][0] for i in range(len(masks)))
    inpainting_mask = inpainting_mask > 0

    annotated_frame = annotate(image_source=image, boxes=boxes_filt, logits=scores_filt, phrases=pred_phrases)
    annotated_frame = annotated_frame[..., ::-1]

    image_mask = inpainting_mask.cpu().numpy()
    image_source_pil = Image.fromarray(image)
    image_mask_pil = Image.fromarray(image_mask)

    # annotated_frame_pil = Image.fromarray(annotated_frame)
    # annotated_frame_with_mask_pil = Image.fromarray(show_mask(inpainting_mask, annotated_frame))

    image_source_for_inpaint = image_source_pil.resize((512, 512))
    image_mask_for_inpaint = image_mask_pil.resize((512, 512))

    inpainting_image = pipe(prompt='', image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]   # prompt=main_name 제외
    inpainting_image = inpainting_image.resize((image_source_pil.size[0], image_source_pil.size[1]))

    ipa_path = "./results_image_sy/inpainting/ipa_{}_{}_{}.png".format(main_name, sub_name, sub_number)
    inpainting_image.save(ipa_path)

    diff_raw_image = cv2.imread(image_path)
    diff_inpainted_image = cv2.imread(ipa_path)

    diff_image = cv2.absdiff(diff_raw_image, diff_inpainted_image)
    diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    anomaly_map_1 = np.where(diff_gray > inpainting_diff_threshold, 255, 0)
    anomaly_map_2 = np.where(image_mask, anomaly_map_1, 0)

    return diff_inpainted_image, anomaly_map_1, anomaly_map_2

def get_anomaly_number(data, main_name):
    for record in data:
        if record['main_name'] == main_name:
            return record['anomaly_number']
    return None

def convert_bmp_to_png(bmp_path):
    # BMP 이미지 열기
    with Image.open(bmp_path) as img:
        # PNG 파일 경로 만들기
        png_path = os.path.splitext(bmp_path)[0] + '.png'
        # PNG 형식으로 저장
        img.save(png_path, 'PNG')
    return png_path


def get_main_names(dataset_name):
    folder_path = f'./datasets/{dataset_name}/'
    if dataset_name == 'mvtec':
        folder_mvtec_path = f'./datasets/{dataset_name}_anomaly_detection/'
        return folder_mvtec_path, sorted([item for item in os.listdir(folder_mvtec_path) if os.path.isdir(os.path.join(folder_mvtec_path, item))])
    elif dataset_name == 'VisA':
        return folder_path,  sorted([item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item)) and not item.startswith('split')])
    elif dataset_name == 'mtd':
        return folder_path,  ['Magnetic']
    elif dataset_name == 'KSDD':
        folder_ksdd_path = './datasets/kolektaorsdd/'
        return folder_ksdd_path,  sorted([item for item in os.listdir(folder_ksdd_path) if os.path.isdir(os.path.join(folder_ksdd_path, item))])
    elif dataset_name == 'KSDD2':
        return folder_path,  ['test']
    elif dataset_name in ['WFDD', 'MPDD', 'DTD-Synthetic']:
        return folder_path,  sorted([item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))])
    elif dataset_name == 'DAGM':
        folder_dagm_path = f'./datasets/{dataset_name}_KaggleUpload/'
        return folder_dagm_path,  sorted([item for item in os.listdir(folder_dagm_path) if os.path.isdir(os.path.join(folder_dagm_path, item))])
    elif dataset_name == 'btad':
        folder_btad_path = f'./datasets/{dataset_name}/BTech_Dataset_transformed/'
        return folder_btad_path, sorted([item for item in os.listdir(folder_btad_path) if os.path.isdir(os.path.join(folder_btad_path, item))])    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_paths(dataset_name, main_name):
    base_path = './datasets'
    good_folder_path = None
    folder_path = None
    sub_names = None

    if dataset_name == 'mvtec':
        good_folder_path = f'{base_path}/{dataset_name}_anomaly_detection/{main_name}/test/good'
        folder_path = f'{base_path}/{dataset_name}_anomaly_detection/{main_name}/test'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'VisA':
        good_folder_path = f'{base_path}/{dataset_name}/{main_name}/Data/Images/Normal'
        folder_path = f'{base_path}/{dataset_name}/{main_name}/Data/Images'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'mtd':
        good_folder_path = f'{base_path}/Magnetic-tile-defect-datasets./Magnetic/MT_Free/Imgs'
        folder_path = f'{base_path}/Magnetic-tile-defect-datasets./{main_name}'
        sub_names = sorted([item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item)) and item != '.git'])
    elif dataset_name == 'KSDD':
        folder_path = f'{base_path}/kolektaorsdd/{main_name}'
        sub_names = sorted([os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.lower().endswith('.jpg')])
    elif dataset_name == 'KSDD2':
        folder_path = f'{base_path}/KolektorSDD2/{main_name}'
        sub_names = sorted(list(set(f[:5] for f in os.listdir(folder_path))))
    elif dataset_name in ['WFDD', 'MPDD', 'DTD-Synthetic']:
        good_folder_path = f'{base_path}/{dataset_name}/{main_name}/train/good'
        folder_path = f'{base_path}/{dataset_name}/{main_name}/test'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'DAGM':
        good_folder_path = f'{base_path}/{dataset_name}_KaggleUpload/{main_name}/'
        folder_path = f'{base_path}/{dataset_name}_KaggleUpload/{main_name}/'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'btad':
        good_folder_path = f'{base_path}/{dataset_name}/BTech_Dataset_transformed/{main_name}/train/ok'
        folder_path = f'{base_path}/{dataset_name}/BTech_Dataset_transformed/{main_name}/test'
        sub_names = ['ko']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return good_folder_path, folder_path, sub_names

def process_good_phrases(dataset_name, good_folder_path, sub_names, random_num, main_name, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE):
    """
    Processes good images for different datasets and returns phrases and scores.

    Args:
        dataset_name (str): Name of the dataset.
        good_folder_path (str): Path to the folder containing good images.
        sub_names (list): List of subdirectories or file names.
        random_num (int): Number of samples to process.
        ram_model: RAM model instance.
        grounding_dino_model: Grounding Dino model instance.
        box_threshold (float): Box threshold.
        text_threshold (float): Text threshold.
        iou_threshold (float): IOU threshold.
        DEVICE (str): Device to use for inference.

    Returns:
        tuple: good_phrases, good_scores
    """
    good_phrases, good_scores = [], []

    if dataset_name in ['mvtec', 'VisA', 'WFDD', 'MPDD', 'DTD-Synthetic', 'btad']:
        file_extension = '.png' if dataset_name in ['mvtec', 'WFDD', 'MPDD', 'DTD-Synthetic', 'btad'] else '.JPG'
        files = sorted([file for file in os.listdir(good_folder_path) if file.endswith(file_extension)])
        for sub_number in random.sample(files, min(random_num, len(files))):
            good_path = os.path.join(good_folder_path, sub_number)
            process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    elif dataset_name == 'mtd':
        files = sorted([file for file in os.listdir(good_folder_path) if file.endswith('.jpg')])
        for sub_number in random.sample(files, min(random_num, len(files))):
            good_path = os.path.join(good_folder_path, sub_number)
            process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    elif dataset_name == 'KSDD':
        with open(f'./datasets/kolektaorsdd/kolektaorsdd_anomaly.json', 'r') as json_file:
            number_data = json.load(json_file)  
            anomaly_number = get_anomaly_number(number_data, main_name)
        for sub_name in random.sample(sub_names, min(random_num, len(sub_names))):
            if sub_name not in anomaly_number:
                good_path = os.path.join(f'./datasets/kolektaorsdd/{main_name}/{sub_name}.jpg')
                process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    elif dataset_name == 'KSDD2':
        idx_ksdd2 = 0
        for sub_name in random.sample(sub_names, len(sub_names)):
            test_path = f'./datasets/KolektorSDD2/{main_name}/{sub_name}_GT.png'
            test_gt = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            if test_gt.sum() <= 0:
                good_path = f'./datasets/KolektorSDD2/{main_name}/{sub_name}.png'
                process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)
                idx_ksdd2 += 1
            if idx_ksdd2 == random_num:
                break

    elif dataset_name == 'DAGM':
        for sub_name in sub_names:
            label_folder = os.path.join(good_folder_path, sub_name, 'Label')
            files = sorted([file for file in os.listdir(label_folder) if file.endswith('.PNG')])
            for sub_number in random.sample(files, min(random_num, len(files))):
                sub_number = sub_number.split('_')[0] + '.PNG'
                good_path = os.path.join(good_folder_path, sub_name, sub_number)
                process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    return good_phrases, good_scores


def process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores):
    """
    Processes a single image and appends results to good_phrases and good_scores.

    Args:
        good_path (str): Path to the good image.
        ram_model: RAM model instance.
        grounding_dino_model: Grounding Dino model instance.
        box_threshold (float): Box threshold.
        text_threshold (float): Text threshold.
        iou_threshold (float): IOU threshold.
        DEVICE (str): Device to use for inference.
        good_phrases (list): List to append phrases.
        good_scores (list): List to append scores.
    """
    img, _, raw_img, ram_img, _, _, _ = load_image(good_path, good_path)
    res = inference_ram(ram_img.to(DEVICE), ram_model)
    img_tags = res[0].strip(' ').replace('  ', ' ').replace(' |', '.').replace('close-up', '').replace('number. ', '')
    _, good_phrase, good_score, _ = process_object_output(grounding_dino_model, img, img_tags, box_threshold, text_threshold, raw_img, iou_threshold, DEVICE)
    good_phrases += good_phrase
    good_scores += good_score

def get_image_and_gt_paths(dataset_name, main_name, sub_name, sub_number):
    """
    Returns the image and ground truth paths based on the dataset, sub_name, and sub_number.

    Args:
        dataset_name (str): The name of the dataset.
        main_name (str): Main name for the dataset.
        sub_name (str): Subfolder name.
        sub_number (str): The identifier of the file.

    Returns:
        tuple: (img_path, gt_path)
    """
    if dataset_name == 'mvtec':
        img_path = f'./datasets/{dataset_name}_anomaly_detection/{main_name}/test/{sub_name}/{sub_number}.png'
        gt_path = img_path if sub_name == 'good' else f'./datasets/{dataset_name}_anomaly_detection/{main_name}/ground_truth/{sub_name}/{sub_number}_mask.png'
    elif dataset_name == 'VisA':
        img_path = f'./datasets/{dataset_name}/{main_name}/Data/Images/{sub_name}/{sub_number}.JPG'
        gt_path = img_path if sub_name == 'Normal' else f'./datasets/{dataset_name}/{main_name}/Data/Masks/{sub_name}/{sub_number}.png'
    elif dataset_name == 'mtd':
        img_path = f'./datasets/Magnetic-tile-defect-datasets./{main_name}/{sub_name}/Imgs/{sub_number}.jpg'
        gt_path = f'./datasets/Magnetic-tile-defect-datasets./{main_name}/{sub_name}/Imgs/{sub_number}.png'
    elif dataset_name == 'KSDD':
        img_path = f'./datasets/kolektaorsdd/{main_name}/{sub_name}.jpg'
        gt_path = f'./datasets/kolektaorsdd/{main_name}/{sub_name}_label.png'
    elif dataset_name == 'KSDD2':
        img_path = f'./datasets/KolektorSDD2/{main_name}/{sub_name}.png'
        gt_path = f'./datasets/KolektorSDD2/{main_name}/{sub_name}_GT.png'
    elif dataset_name in ['WFDD', 'MPDD', 'DTD-Synthetic']:
        img_path = f'./datasets/{dataset_name}/{main_name}/test/{sub_name}/{sub_number}.png'
        gt_path = f'./datasets/{dataset_name}/{main_name}/ground_truth/{sub_name}/{sub_number}_mask.png'
    elif dataset_name == 'DAGM':
        img_path = f'./datasets/{dataset_name}_KaggleUpload/{main_name}/{sub_name}/{sub_number}.PNG'
        gt_path = f'./datasets/{dataset_name}_KaggleUpload/{main_name}/{sub_name}/Label/{sub_number}_label.PNG'
        if not os.path.exists(gt_path):
            gt_path = None
    elif dataset_name == 'btad':
        img_path = f'./datasets/{dataset_name}/BTech_Dataset_transformed/{main_name}/test/{sub_name}/{sub_number}.png'
        gt_path = f'./datasets/{dataset_name}/BTech_Dataset_transformed/{main_name}/ground_truth/{sub_name}/{sub_number}.png'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return img_path, gt_path

def unique_words_list(strings):
    unique_strings = []

    for s in strings:
        words = s.split(' ')
        unique_words = list(set(words))
        unique_strings += unique_words

    return list(set(unique_strings))

def get_main_names(dataset_name):
    folder_path = f'../IAP-AS/datasets/{dataset_name}/'
    if dataset_name == 'mvtec':
        folder_mvtec_path = f'../IAP-AS/datasets/{dataset_name}_anomaly_detection/'
        return folder_mvtec_path, sorted([item for item in os.listdir(folder_mvtec_path) if os.path.isdir(os.path.join(folder_mvtec_path, item))])
    elif dataset_name == 'VisA':
        return folder_path,  sorted([item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item)) and not item.startswith('split')])
    elif dataset_name == 'mtd':
        return folder_path,  ['Magnetic']
    elif dataset_name == 'KSDD':
        folder_ksdd_path = '../IAP-AS/datasets/kolektaorsdd/'
        return folder_ksdd_path,  sorted([item for item in os.listdir(folder_ksdd_path) if os.path.isdir(os.path.join(folder_ksdd_path, item))])
    elif dataset_name == 'KSDD2':
        return folder_path,  ['test']
    elif dataset_name in ['WFDD', 'MPDD', 'DTD-Synthetic']:
        return folder_path,  sorted([item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))])
    elif dataset_name == 'DAGM':
        folder_dagm_path = f'../IAP-AS/datasets/{dataset_name}_KaggleUpload/'
        return folder_dagm_path,  sorted([item for item in os.listdir(folder_dagm_path) if os.path.isdir(os.path.join(folder_dagm_path, item))])
    elif dataset_name == 'btad':
        folder_btad_path = f'../IAP-AS/datasets/{dataset_name}/BTech_Dataset_transformed/'
        return folder_btad_path, sorted([item for item in os.listdir(folder_btad_path) if os.path.isdir(os.path.join(folder_btad_path, item))])    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_paths(dataset_name, main_name):
    base_path = '../IAP-AS/datasets'
    good_folder_path = None
    folder_path = None
    sub_names = None

    if dataset_name == 'mvtec':
        good_folder_path = f'{base_path}/{dataset_name}_anomaly_detection/{main_name}/test/good'
        folder_path = f'{base_path}/{dataset_name}_anomaly_detection/{main_name}/test'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'VisA':
        good_folder_path = f'{base_path}/{dataset_name}/{main_name}/Data/Images/Normal'
        folder_path = f'{base_path}/{dataset_name}/{main_name}/Data/Images'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'mtd':
        good_folder_path = f'{base_path}/Magnetic-tile-defect-datasets./Magnetic/MT_Free/Imgs'
        folder_path = f'{base_path}/Magnetic-tile-defect-datasets./{main_name}'
        sub_names = sorted([item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item)) and item != '.git'])
    elif dataset_name == 'KSDD':
        folder_path = f'{base_path}/kolektaorsdd/{main_name}'
        sub_names = sorted([os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.lower().endswith('.jpg')])
    elif dataset_name == 'KSDD2':
        folder_path = f'{base_path}/KolektorSDD2/{main_name}'
        sub_names = sorted(list(set(f[:5] for f in os.listdir(folder_path))))
    elif dataset_name in ['WFDD', 'MPDD', 'DTD-Synthetic']:
        good_folder_path = f'{base_path}/{dataset_name}/{main_name}/train/good'
        folder_path = f'{base_path}/{dataset_name}/{main_name}/test'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'DAGM':
        good_folder_path = f'{base_path}/{dataset_name}_KaggleUpload/{main_name}/'
        folder_path = f'{base_path}/{dataset_name}_KaggleUpload/{main_name}/'
        sub_names = os.listdir(folder_path)
    elif dataset_name == 'btad':
        good_folder_path = f'{base_path}/{dataset_name}/BTech_Dataset_transformed/{main_name}/train/ok'
        folder_path = f'{base_path}/{dataset_name}/BTech_Dataset_transformed/{main_name}/test'
        sub_names = ['ko']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return good_folder_path, folder_path, sub_names

def process_good_phrases(dataset_name, good_folder_path, sub_names, random_num, main_name, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE):
    """
    Processes good images for different datasets and returns phrases and scores.

    Args:
        dataset_name (str): Name of the dataset.
        good_folder_path (str): Path to the folder containing good images.
        sub_names (list): List of subdirectories or file names.
        random_num (int): Number of samples to process.
        ram_model: RAM model instance.
        grounding_dino_model: Grounding Dino model instance.
        box_threshold (float): Box threshold.
        text_threshold (float): Text threshold.
        iou_threshold (float): IOU threshold.
        DEVICE (str): Device to use for inference.

    Returns:
        tuple: good_phrases, good_scores
    """
    good_phrases, good_scores = [], []

    if dataset_name in ['mvtec', 'VisA', 'WFDD', 'MPDD', 'DTD-Synthetic', 'btad']:
        file_extension = '.png' if dataset_name in ['mvtec', 'WFDD', 'MPDD', 'DTD-Synthetic', 'btad'] else '.JPG'
        files = sorted([file for file in os.listdir(good_folder_path) if file.endswith(file_extension)])
        for sub_number in random.sample(files, min(random_num, len(files))):
            good_path = os.path.join(good_folder_path, sub_number)
            process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    elif dataset_name == 'mtd':
        files = sorted([file for file in os.listdir(good_folder_path) if file.endswith('.jpg')])
        for sub_number in random.sample(files, min(random_num, len(files))):
            good_path = os.path.join(good_folder_path, sub_number)
            process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    elif dataset_name == 'KSDD':
        with open(f'../IAP-AS/datasets/kolektaorsdd/kolektaorsdd_anomaly.json', 'r') as json_file:
            number_data = json.load(json_file)  
            anomaly_number = get_anomaly_number(number_data, main_name)
        for sub_name in random.sample(sub_names, min(random_num, len(sub_names))):
            if sub_name not in anomaly_number:
                good_path = os.path.join(f'../IAP-AS/datasets/kolektaorsdd/{main_name}/{sub_name}.jpg')
                process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    elif dataset_name == 'KSDD2':
        idx_ksdd2 = 0
        for sub_name in random.sample(sub_names, len(sub_names)):
            test_path = f'../IAP-AS/datasets/KolektorSDD2/{main_name}/{sub_name}_GT.png'
            test_gt = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            if test_gt.sum() <= 0:
                good_path = f'../IAP-AS/datasets/KolektorSDD2/{main_name}/{sub_name}.png'
                process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)
                idx_ksdd2 += 1
            if idx_ksdd2 == random_num:
                break

    elif dataset_name == 'DAGM':
        for sub_name in sub_names:
            label_folder = os.path.join(good_folder_path, sub_name, 'Label')
            files = sorted([file for file in os.listdir(label_folder) if file.endswith('.PNG')])
            for sub_number in random.sample(files, min(random_num, len(files))):
                sub_number = sub_number.split('_')[0] + '.PNG'
                good_path = os.path.join(good_folder_path, sub_name, sub_number)
                process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores)

    return good_phrases, good_scores


def process_single_image(good_path, ram_model, grounding_dino_model, box_threshold, text_threshold, iou_threshold, DEVICE, good_phrases, good_scores):
    """
    Processes a single image and appends results to good_phrases and good_scores.

    Args:
        good_path (str): Path to the good image.
        ram_model: RAM model instance.
        grounding_dino_model: Grounding Dino model instance.
        box_threshold (float): Box threshold.
        text_threshold (float): Text threshold.
        iou_threshold (float): IOU threshold.
        DEVICE (str): Device to use for inference.
        good_phrases (list): List to append phrases.
        good_scores (list): List to append scores.
    """
    img, _, raw_img, ram_img, _, _, _ = load_image(good_path, good_path)
    res = inference_ram(ram_img.to(DEVICE), ram_model)
    img_tags = res[0].strip(' ').replace('  ', ' ').replace(' |', '.').replace('close-up', '').replace('number. ', '')
    _, good_phrase, good_score, _ = process_object_output(grounding_dino_model, img, img_tags, box_threshold, text_threshold, raw_img, iou_threshold, DEVICE)
    good_phrases += good_phrase
    good_scores += good_score

def get_image_and_gt_paths(dataset_name, main_name, sub_name, sub_number):
    """
    Returns the image and ground truth paths based on the dataset, sub_name, and sub_number.

    Args:
        dataset_name (str): The name of the dataset.
        main_name (str): Main name for the dataset.
        sub_name (str): Subfolder name.
        sub_number (str): The identifier of the file.

    Returns:
        tuple: (img_path, gt_path)
    """
    base_path = '../IAP-AS/datasets'
    if dataset_name == 'mvtec':
        img_path = f'{base_path}/{dataset_name}_anomaly_detection/{main_name}/test/{sub_name}/{sub_number}.png'
        gt_path = img_path if sub_name == 'good' else f'{base_path}/{dataset_name}_anomaly_detection/{main_name}/ground_truth/{sub_name}/{sub_number}_mask.png'
    elif dataset_name == 'VisA':
        img_path = f'{base_path}/{dataset_name}/{main_name}/Data/Images/{sub_name}/{sub_number}.JPG'
        gt_path = img_path if sub_name == 'Normal' else f'{base_path}/{dataset_name}/{main_name}/Data/Masks/{sub_name}/{sub_number}.png'
    elif dataset_name == 'mtd':
        img_path = f'{base_path}/Magnetic-tile-defect-datasets./{main_name}/{sub_name}/Imgs/{sub_number}.jpg'
        gt_path = f'{base_path}/Magnetic-tile-defect-datasets./{main_name}/{sub_name}/Imgs/{sub_number}.png'
    elif dataset_name == 'KSDD':
        img_path = f'{base_path}/kolektaorsdd/{main_name}/{sub_name}.jpg'
        gt_path = f'{base_path}/kolektaorsdd/{main_name}/{sub_name}_label.png'
    elif dataset_name == 'KSDD2':
        img_path = f'{base_path}/KolektorSDD2/{main_name}/{sub_name}.png'
        gt_path = f'{base_path}/KolektorSDD2/{main_name}/{sub_name}_GT.png'
    elif dataset_name in ['WFDD', 'MPDD', 'DTD-Synthetic']:
        img_path = f'{base_path}/{dataset_name}/{main_name}/test/{sub_name}/{sub_number}.png'
        gt_path = f'{base_path}/{dataset_name}/{main_name}/ground_truth/{sub_name}/{sub_number}_mask.png'
    elif dataset_name == 'DAGM':
        img_path = f'{base_path}/{dataset_name}_KaggleUpload/{main_name}/{sub_name}/{sub_number}.PNG'
        gt_path = f'{base_path}/{dataset_name}_KaggleUpload/{main_name}/{sub_name}/Label/{sub_number}_label.PNG'
        if not os.path.exists(gt_path):
            gt_path = None
    elif dataset_name == 'btad':
        img_path = f'{base_path}/{dataset_name}/BTech_Dataset_transformed/{main_name}/test/{sub_name}/{sub_number}.png'
        gt_path = f'{base_path}/{dataset_name}/BTech_Dataset_transformed/{main_name}/ground_truth/{sub_name}/{sub_number}.png'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return img_path, gt_path