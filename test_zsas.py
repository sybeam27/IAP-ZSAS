# setting
# library
import pdb
import sys
import json
import os
sys.path.append('./SegmentAnything/GroundingDINO')
sys.path.append('./SegmentAnything/SAM')
sys.path.append('./SegmentAnything')
sys.path.append('./Llama3')
sys.path.append('./utils')

import random
import argparse
from typing import List
import argparse
import cv2
import numpy as np
import pandas as pd
import requests
import stringprep
import torch
import torchvision
import torchvision.transforms as TS
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionInpaintPipeline
from io import BytesIO
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.ops import box_convert
import torchvision.ops as ops
from ram import inference_ram
from ram.models import ram
import supervision as sv
import spacy
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from segment_anything import SamPredictor, build_sam, build_sam_hq
import SegmentAnything.SAA as SegmentAnyAnomaly
from SegmentAnything.datasets import *
from SegmentAnything.utils.csv_utils import *
from SegmentAnything.utils.eval_utils import *
from SegmentAnything.utils.metrics import *
from SegmentAnything.utils.training_utils import *
from utils.function import load_image, load_model, normalize, setup_seed, eval_zsas_last_2, \
    process_object_output, process_box_output, process_size_output, \
    process_anomaly_segmentation, process_draw_boxes, process_draw_masks, \
    process_specify_resolution, process_anomaly_tags_2, get_anomaly_number, convert_bmp_to_png, \
    get_paths, get_main_names, process_good_phrases, get_image_and_gt_paths
    
# ArgumentParser 
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--gpu', type=str, default="0", help='gpu_number')
parser.add_argument('--dataset', type=str, default="mvtec", help='dataset_name')
parser.add_argument('--model', type=str, default="iap_zsas", help='model_name')
parser.add_argument('--box_threshold', type=float, default=0.1, help='GroundingSAM box threshold')
parser.add_argument('--text_threshold', type=float, default=0.1, help='GroundingSAM text threshold')
parser.add_argument('--size_threshold', type=float, default=0.8, help='Bounding-box size threshold')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IOU threshold')
parser.add_argument('--random_img_num', type=int, default=10, help='random image extraction number')
parser.add_argument('--eval_resolution', type=int, default=400, help='Description of evaluation resolution')
parser.add_argument('--exp_idx', type=str, default='random', help='Description of experiment index')
parser.add_argument('--version', type=int, default=1, help='Description of evaluation version')

args = parser.parse_args()
gpu_number = args.gpu
box_threshold = args.box_threshold
text_threshold = args.text_threshold
threshold = args.size_threshold
iou_threshold = args.iou_threshold
random_num = args.random_img_num
dataset_name = args.dataset
model_name = args.model
experiment_index = args.exp_idx
version = args.version
eval_resolution = args.eval_resolution


print("-" * 50, 'MODEL LOAD START', "-" * 50)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
DEVICE = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else 'cpu')
SELECT_SAM_HQ = False

dino_config_file = "./SegmentAnything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" 
dino_checkpoint = "./checkpoints/groundingdino_swint_ogc.pth"  
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
sam_hq_checkpoint = "./checkpoints/sam_hq_vit_h.pth"
ram_checkpoint = "./checkpoints/ram_swin_large_14m.pth"
llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_api_token = "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"

# Get GroundingDINO Model
grounding_dino_model = load_model(dino_config_file, dino_checkpoint, DEVICE)

# Get SAM Model
if SELECT_SAM_HQ:
    sam_model = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(DEVICE))
else:
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(DEVICE))

# Get RAM Model
ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
ram_model.eval()
ram_model = ram_model.to(DEVICE)

login(llama_api_token)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": f"cuda:{gpu_number}"},
)

nlp = spacy.load("en_core_web_md") 

# llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id)
print("-" * 50, 'MODEL LOAD COMPLETE', "-" * 50)


print("-" * 54, 'TEST START ', "-" * 54)
setup_seed(111)

# mvtec_t_list = ['carpet','leather','grid','tile','wood']
# mvtec_so_list = ['bottle','hazelnut','cable','capsule','metal_nut','pill','screw','toothbrush','transistor','zipper']

# if dataset_name == 'mvtec':
#     main_names = mvtec_t_list + mvtec_so_list
# elif dataset_name == "mtd":
#     folder_path = './datasets/Magnetic-tile-defect-datasets./'  
#     main_names = ['Magnetic'] 
# elif dataset_name == "ksdd":
#     folder_path = './datasets/kolektaorsdd/'
#     main_names = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
#     main_names = sorted(main_names)

folder_path, main_names = get_main_names(dataset_name)

print(f'main_names of {dataset_name} :', main_names)

root_dir = f"./result_{model_name}"
csv_dir = os.path.join(root_dir, 'csv')
os.makedirs(csv_dir, exist_ok=True)
result_dir = os.path.join(root_dir, dataset_name, 'result')
os.makedirs(result_dir, exist_ok=True)


print('-' * 40, f'{model_name} Model : {dataset_name} test is starting... ', '-' * 40)
for main_name in main_names:
    print('-' * 45, f'{model_name} Model : {main_name} image test is starting...', '-' * 45) 
    
    test_imgs, gt_list, gt_mask_list, names, test_scores, test_masks = [], [], [], [], [], []

    # if dataset_name == 'mvtec':
    #     good_folder_path = f'./datasets/{dataset_name}_anomaly_detection/{main_name}/test/good'
    #     folder_path = f'./datasets/{dataset_name}_anomaly_detection/{main_name}/test'
    #     sub_names = os.listdir(folder_path)
    # elif dataset_name == 'mtd':
    #     good_folder_path = f'./datasets/Magnetic-tile-defect-datasets./Magnetic/MT_Free/Imgs' 
    #     folder_path = f'./datasets/Magnetic-tile-defect-datasets./{main_name}'
    #     sub_names = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item)) and item != '.git']
    #     sub_names = sorted(sub_names)   
    # elif dataset_name == 'ksdd':
    #     folder_path = f'./datasets/kolektaorsdd/{main_name}'
    #     sub_names = [os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.lower().endswith('.jpg')]      
    #     sub_names = sorted(sub_names)  
    #     with open(f'./datasets/kolektaorsdd/kolektaorsdd_anomaly.json', 'r') as json_file:
    #         number_data = json.load(json_file)  
    #         anomaly_number = get_anomaly_number(number_data, main_name)

    good_folder_path, folder_path, sub_names = get_paths(dataset_name, main_name)
    
    if dataset_name == 'KSDD':
        with open(f'./datasets/kolektaorsdd/kolektaorsdd_anomaly.json', 'r') as json_file:
            number_data = json.load(json_file)  
            anomaly_number = get_anomaly_number(number_data, main_name)
            
    if model_name == "iap_zsas":
        # if dataset_name == 'mvtec':        
        #     good_phrases, good_scores = [], []
        #     if len(sub_names) < random_num:
        #         random_num = len(sub_names)
        #     for sub_number in random.sample(sorted(os.listdir(good_folder_path)), random_num):
        #         sub_number = sub_number.split(".")[0]    
        #         good_path = f'./datasets/{dataset_name}_anomaly_detection/{main_name}/test/good/{sub_number}.png'
        #         img, _, raw_img, ram_img, _, _, _ = load_image(good_path, good_path)
        #         res = inference_ram(ram_img.to(DEVICE), ram_model)
        #         img_tags = res[0].strip(' ').replace('  ', ' ').replace(' |', '.').replace('close-up', '').replace('number. ', '')
        #         _, good_phrase, good_score, _ = process_object_output(grounding_dino_model, img, img_tags, box_threshold, text_threshold, raw_img, iou_threshold, DEVICE)
        #         good_phrases += good_phrase
        #         good_scores += good_score    
        # elif dataset_name == 'mtd':        
        #     good_phrases, good_scores = [], []
        #     for sub_number in random.sample(sorted([file for file in os.listdir(good_folder_path) if file.endswith('.jpg')]), random_num):
        #         good_path = os.path.join(good_folder_path, sub_number)
        #         img, _, raw_img, ram_img, _, _, _ = load_image(good_path, good_path)
        #         res = inference_ram(ram_img.to(DEVICE), ram_model)
        #         img_tags = res[0].strip(' ').replace('  ', ' ').replace(' |', '.').replace('close-up', '').replace('number. ', '')
        #         _, good_phrase, good_score, _ = process_object_output(grounding_dino_model, img, img_tags, box_threshold, text_threshold, raw_img, iou_threshold, DEVICE)
        #         good_phrases += good_phrase
        #         good_scores += good_score             
        # elif dataset_name == 'ksdd':
        #     good_phrases, good_scores = [], []
        #     if len(sub_names) < random_num:
        #         random_num = len(sub_names)
        #     for sub_name in random.sample(sub_names, random_num):
        #         if sub_name not in anomaly_number:
        #             good_path = os.path.join(f'./datasets/kolektaorsdd/{main_name}/{sub_name}.jpg')           
        #             img, _, raw_img, ram_img, _, _, _ = load_image(good_path, good_path)
        #             res = inference_ram(ram_img.to(DEVICE), ram_model)
        #             img_tags = res[0].strip(' ').replace('  ', ' ').replace(' |', '.').replace('close-up', '').replace('number. ', '')
        #             _, good_phrase, good_score, _ = process_object_output(grounding_dino_model, img, img_tags, box_threshold, text_threshold, raw_img, iou_threshold, DEVICE)
        #             good_phrases += good_phrase
        #             good_scores += good_score
        
        good_phrases, good_scores = process_good_phrases(dataset_name,
                                                good_folder_path,
                                                sub_names,
                                                random_num,
                                                main_name,
                                                ram_model,
                                                grounding_dino_model,
                                                box_threshold,
                                                text_threshold,
                                                iou_threshold,
                                                DEVICE)    
        top_k = 1
        good_df = pd.DataFrame({'Phrase': good_phrases, 'Score': [score.item() for score in good_scores]})
        top_df = good_df.groupby('Phrase')['Score'].max().nlargest(top_k).reset_index()
        top_phrases = top_df['Phrase'].tolist()
        top_scores = top_df['Score'].tolist()
        object_tag = top_phrases[0]
        
        # 추가한 부분
        # top_phrases_lst = top_phrases[0].split()
        # doc = nlp(' '.join(top_phrases_lst))
        # object_tag = [token.text for token in doc if token.pos_ == "NOUN" and token.dep_ != "compound"]

        anomaly_tags = process_anomaly_tags_2(llama_model, llama_tokenizer, top_phrases)
        
        # 산업용 프롬프트
        # anomaly_tags = process_anomaly_tags_2_sy(llama_model, llama_tokenizer, top_phrases)
        
        # 추가한 부분
        # anomaly_tags = refine_anomaly_tags(object_tag, anomaly_tags) 
    
    for sub_name in tqdm(sub_names, desc="Processing"):
        # if dataset_name == 'ksdd':
        #     sub_numbers = ['0.KSDD2']
        # elif dataset_name == 'mtd':
        #     sub_numbers = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(folder_path, sub_name,'Imgs')) if file.lower().endswith('.jpg')]
        # else:            
        #     sub_folder_path = os.path.join(folder_path, sub_name)
        #     sub_numbers= sorted(os.listdir(sub_folder_path))
        if dataset_name == 'mtd':
            sub_numbers = [os.path.splitext(file)[0]  for file in os.listdir(os.path.join(folder_path, sub_name, 'Imgs')) if file.lower().endswith('.jpg')]
        elif dataset_name in ['KSDD', 'KSDD2']:
            sub_numbers = [f'0.{dataset_name}']
        else:            
            sub_numbers = sorted(os.listdir(os.path.join(folder_path, sub_name)))

        for sub_number in sub_numbers:
            sub_number = sub_number.split(".")[0]
            img_path, gt_path = get_image_and_gt_paths(dataset_name, main_name, sub_name, sub_number)
            if gt_path == None or (sub_name == 'good' and dataset_name != 'mvtec'):
                continue
            # if dataset_name == 'mvtec':
            #     img_path = f'./datasets/{dataset_name}_anomaly_detection/{main_name}/test/{sub_name}/{sub_number}.png'
            #     gt_path = img_path if sub_name == 'good' else f'./datasets/{dataset_name}_anomaly_detection/{main_name}/ground_truth/{sub_name}/{sub_number}_mask.png'
            # elif dataset_name == 'mtd':
            #     img_path = f'./datasets/Magnetic-tile-defect-datasets./{main_name}/{sub_name}/Imgs/{sub_number}.jpg'
            #     gt_path = f'./datasets/Magnetic-tile-defect-datasets./{main_name}/{sub_name}/Imgs/{sub_number}.png'
            # elif dataset_name == 'ksdd':
            #     img_path = f'./datasets/kolektaorsdd/{main_name}/{sub_name}.jpg'
            #     gt_path = f'./datasets/kolektaorsdd/{main_name}/{sub_name}_label.bmp'
            #     gt_path = convert_bmp_to_png(gt_path)

            img, src_img, raw_img, ram_img, gt_img, gt_bn, gt_mask = load_image(img_path, gt_path)
            
            test_imgs += [np.array(src_img)]
            # if dataset_name == 'mtd':
            #     gt_list += [0 if sub_name in ['MT_Free'] else 1]
            # elif dataset_name == 'ksdd':
            #     if sub_name not in anomaly_number: 
            #         gt_list += [0]
            #     else:
            #         gt_list += [1]
            # else:
            #     gt_list += [0 if sub_name in ['good', 'Normal'] else 1]
            if dataset_name == 'KSDD':
                gt_list = [1 if sub_name in anomaly_number else 0]
            elif dataset_name == 'KSDD2':
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_list = [1 if gt.sum() > 0 else 0]
            else:
                gt_list += [0 if sub_name in ['good', 'Normal', 'MT_Free'] else 1]  
                              
            # gt_img[gt_img > 0] = 1  # 255 -> 1로 변경
            # gt_mask_list += [np.array(gt_img)]
            gt_mask_list += [(gt_bn == 255).astype(int)]
            names += [f'{main_name}_{sub_name}_{sub_number}']
            
            # model start
            if model_name == 'iap_zsas':
                object_boxes_filt, _, _, object_size = process_object_output(grounding_dino_model, img, object_tag, box_threshold, text_threshold, raw_img, iou_threshold, device=DEVICE)
                obj_box_image = process_draw_boxes(raw_img, object_boxes_filt, object_tag)
                
                boxes_filt, pred_phrases, boxes_score = process_box_output(
                    grounding_dino_model, img, anomaly_tags, box_threshold, text_threshold, DEVICE, raw_img, iou_threshold)
                bf_th_box_image = process_draw_boxes(raw_img, boxes_filt, pred_phrases)
                
                boxes_filt, pred_phrases, boxes_score = process_size_output(raw_img, object_size, boxes_filt, pred_phrases, boxes_score, threshold)
                af_th_box_image = process_draw_boxes(raw_img, boxes_filt, pred_phrases)
        
                anomaly_masks, masks_score, boxes_score = process_anomaly_segmentation(raw_img, src_img, sam_model, boxes_filt, boxes_score, DEVICE)
                anomlay_mask_image = process_draw_masks(raw_img, anomaly_masks)
                
                scores = [anomaly_masks[i].cpu().numpy() * boxes_score[i].item() for i in range(len(anomaly_masks))]
                score = np.sum(scores, axis=0)[0]
                score = normalize(score)
                test_scores += [np.array(score)]
                
                masks = sum(anomaly_masks[i][0] for i in range(len(anomaly_masks)))
                mask = masks.cpu().numpy()
                mask[mask > 0] = 1
                test_masks += [mask]
                
            elif model_name == 'base':
                anomaly_tags = 'defect, abnormal'

                boxes_filt, pred_phrases, boxes_score = process_box_output(
                    grounding_dino_model, img, anomaly_tags, box_threshold, text_threshold, DEVICE, raw_img, iou_threshold)
                
                boxes_filt, pred_phrases, boxes_score = process_box_output(
                    grounding_dino_model, img, anomaly_tags, box_threshold, text_threshold, DEVICE, raw_img, iou_threshold)
                bf_th_box_image = process_draw_boxes(raw_img, boxes_filt, pred_phrases)

                anomaly_masks, boxes_score = process_anomaly_segmentation(raw_img, src_img, sam_model, boxes_filt, boxes_score, DEVICE)
                anomlay_mask_image = process_draw_masks(raw_img, anomaly_masks)

                scores = [anomaly_masks[i].cpu().numpy() * boxes_score[i].item() for i in range(len(boxes_score))]
                score = np.sum(scores, axis=0)[0]
                test_scores += [score]
                masks = sum(anomaly_masks[i][0] for i in range(len(anomaly_masks)))
                mask = masks.cpu().numpy()
                mask[mask > 0] = 1
                test_masks += [mask]

    print('-' * 30, f'{model_name} Model : {main_name} image test is ended...', '-' * 30) 
    df = pd.DataFrame({
    'names' : names,
    'test_imgs' : test_imgs,
    'gt_list'  : gt_list, 
    'gt_mask_list' : gt_mask_list,
    'test_scores' : test_scores,
    'test_masks' : test_masks
    })
        
    result_path = os.path.join(result_dir, f"{main_name}_test_results_indx_{experiment_index}.pkl")
    df.to_pickle(result_path)

print('-' * 40, f'{model_name} Model : {dataset_name} test is ended...', '-' * 40)
print("-" * 50, 'TEST COMPLETE', "-" * 50)


print("-" * 50, 'EVALUATION START', "-" * 50)
root_dir = f"./result_{model_name}"
csv_dir = os.path.join(root_dir, 'csv')
result_dir = os.path.join(root_dir, dataset_name, 'result')
idx,  is_pro_lst, is_ap_lst, is_f1m_lst, ps_pro_lst, ps_ap_lst, ps_f1m_lst = [], [], [], [], [], [], []

for main_name in main_names:
    print('-' * 30, f'{model_name} Model : {main_name} anomaly score evaluation is starting...', '-' * 30) 
    
    result_path = os.path.join(result_dir, f"{main_name}_test_results_indx_{experiment_index}.pkl")
    result_df = pd.read_pickle(result_path)
    
    test_imgs = result_df['test_imgs'].tolist()
    gt_list = result_df['gt_list'].tolist()
    gt_mask_list = result_df['gt_mask_list'].tolist()
    test_scores = result_df['test_scores'].tolist()
    test_masks = result_df['test_masks'].tolist()

    eval_resolution = 400
    test_imgs, test_scores, test_masks, gt_mask_list = process_specify_resolution(
        test_imgs, test_scores, test_masks, gt_mask_list,
        resolution=(eval_resolution, eval_resolution))

    n_test_scores = normalize(test_scores)
    n_test_masks = normalize(test_masks)
    np_scores = np.array(n_test_scores) 
    np_masks = np.array(n_test_masks) 
    img_scores = np_scores.reshape(np_scores.shape[0], -1).max(axis=1) 
    img_masks = np_masks.reshape(np_masks.shape[0], -1).max(axis=1) 

    gt_list = np.stack(gt_list, axis=0)
    gt_list = np.asarray(gt_list, dtype=int)
    gt_masks = np.asarray(gt_mask_list, dtype=int)  

    is_ap, is_f1m, is_pro = eval_zsas_last_2(gt_list, img_scores)
    print('image-level score f1-max :', round(is_f1m, 2))
    ps_ap, ps_f1m, ps_pro = eval_zsas_last_2(gt_masks.flatten(), np_scores.flatten())
    print('pixel-level score f1-max :', round(ps_f1m, 2))
    
    idx.append(f'{main_name}_indx_{experiment_index}')
    is_ap_lst.append(round(is_ap, 2))
    is_f1m_lst.append(round(is_f1m, 2))
    is_pro_lst.append(round(is_pro, 2))
    ps_ap_lst.append(round(ps_ap, 2))
    ps_f1m_lst.append(round(ps_f1m, 2))
    ps_pro_lst.append(round(ps_pro, 2))
    print('-' * 30, f'{model_name} Model : {main_name} image evaluation is ended...', '-' * 30) 

evaluate_df = pd.DataFrame({
'idx'   : idx,
'AS_ap'  : ps_ap_lst, 'AS_f1-max'  : ps_f1m_lst, 'AS_s_pro' : ps_pro_lst,
'AC_ap'  : is_ap_lst, 'AC_f1-max'  : is_f1m_lst, 'AC_s_pro' : is_pro_lst
})

numeric_cols = evaluate_df.select_dtypes(include=[float, int]).columns
column_means_without_nan = evaluate_df[numeric_cols].mean(skipna=True)
column_means_without_nan['idx'] = 'mean'

evaluate_df.loc[-1] = column_means_without_nan
evaluate_df.index = evaluate_df.index + 1  
evaluate_df = evaluate_df.sort_index() 

csv_path = os.path.join(csv_dir, f"{model_name}_{dataset_name}_score_indx_{experiment_index}_{version}.csv")
evaluate_df.to_csv(csv_path, header=True, float_format='%.2f')

print("-" * 50, 'EVALUATION COMPLETE', "-" * 50)
