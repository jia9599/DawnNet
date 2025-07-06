# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
from skimage import filters
import random
from skimage.segmentation import slic
from torch import nn
from main.dataset.dataset_MMIS import MMIS_dataset
from main.loss import BinaryMaskLoss
from main.metrics import dice_coefficient, batch_hausdorff_95
from main.unet.unet_model_uncertainty_V2 import UNet_Uncertainty_V2
join = os.path.join
from torchvision.transforms import InterpolationMode
from segment_anything import sam_model_registry
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Tuple, Dict
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from skimage.filters import laplace, gaussian


def parse_args():
    parser = argparse.ArgumentParser(description='pytorch sam testing')
    parser.add_argument("--base-dir", default='./data/MMIS/TC')
    parser.add_argument("--list-dir", default='./lists/lists_MMIS')
    parser.add_argument("--output-dir", default='./out_dir')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="1", type=int)
    parser.add_argument("--debug", default=False, type=bool, 
                      help='whether only use the first batch')
    parser.add_argument('--is-savenii', type=bool, default=False)
    parser.add_argument('--seed', default=42, type=int, help='Random Number Seed ')
    parser.add_argument('--pre-weights',
                      default=r'./pre_weights/AdaMS_SAM.pth',
                      help='pre_weights')
    parser.add_argument('--pre-weights-unet',
                      default=r'./pre_weights/aux_model.pth',
                      help='pre_weights')
    parser.add_argument('--isAutoPrompt',
                      default=False,
                      help='Whether to enable the auto-prompt function  Note: Prompts will increase inference time')
    args = parser.parse_args()
    return args

args = parse_args()
class Model(nn.Module):
    def __init__(self, sam, unet, unet_uncertainty):
        super(Model, self).__init__()
        self.sam = sam
        self.unet = unet
        self.unet_uncertainty = unet_uncertainty
        self.resizer = ResizeLongestSide(1024)
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, x_unet, origin,isAutoPrompt):

        x_unet, uncertainty_map = predict_with_uncertainty(self.unet, x_unet, self.unet_uncertainty, 10)
        img_embedding, spgen1, spgen2, spgen3, spgen4, cnn_out, main_uncertainty_map = self.sam.image_encoder(x, origin, apply_drop=True)
        spgen1 = self.up1(spgen1)
        spgen2 = self.up2(spgen2)
        spgen3 = self.up3(spgen3)
        prompt = postprocess_masks(spgen1 + spgen2 + spgen3 + spgen4)
        autoMask = prompt
        if isAutoPrompt:
            # Auxiliary mask
            aux_mask = torch.sigmoid((0.7 * x_unet + 0.3 * cnn_out))
            # Main mask
            mask = torch.sigmoid(prompt)
            # Edge map
            final_edge_map = generate_edge_map(origin)
            # Superpixel map
            segments = generate_superpixels(origin, n_segments=250)
            # High-confidence prompt points
            high_confidence_points = select_points_grid(mask, uncertainty_map, main_uncertainty_map, aux_mask)
            # High-uncertainty prompt points
            high_uncertainty_points = select_points_mask_superpixels(
                mask=mask,
                aux_uncertainty_map=uncertainty_map,
                main_uncertainty_map=main_uncertainty_map,
                x_aux_mask=aux_mask,
                segments=segments,
                sobel_map=final_edge_map
            )
        else:
            high_confidence_points = []
            high_uncertainty_points = []
            segments = None
            final_edge_map = None

        points = high_confidence_points + high_uncertainty_points
        use_points = []
        if (points):
            points_prompt = convert_to_tuple(points)
            coords = points_prompt[0]
            labels = points_prompt[1]
            apply_coords = self.resizer.apply_coords_torch(coords, (512, 512))
            use_points = (apply_coords, labels)
        outputs_mask = []

        if use_points:
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=use_points,
                boxes=None,
                masks=None
            )
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=img_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
        else:
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=img_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
        masks = postprocess_masks(low_res_masks)

        outputs_mask.append(masks)
        return masks, autoMask, points, uncertainty_map, x_unet, segments, cnn_out,final_edge_map,main_uncertainty_map
    
def select_points_mask_superpixels(mask, aux_uncertainty_map, main_uncertainty_map, x_aux_mask,segments, sobel_map,
                                   foreground_threshold=0.3,
                                   background_threshold=0.5,
                                   min_foreground_ratio=0.2,
                                   nms_radius=8):
    high_uncertainty_candidates = []

    if isinstance(segments, torch.Tensor):
        segments_np = segments.cpu().numpy()
    else:
        segments_np = segments

    unique_regions = np.unique(segments_np)


    for region_id in unique_regions:
        region_mask = (segments_np == region_id)
        region_mask_tensor = torch.from_numpy(region_mask).to(mask.device)

        mask_grid = mask[0, 0][region_mask_tensor]
        x_unet_mask_grid = x_aux_mask[0, 0][region_mask_tensor]
        sobel_map_grid = sobel_map[0, 0][region_mask_tensor]
        aux_uncertainty_grid = aux_uncertainty_map[0, 0][region_mask_tensor]
        main_uncertainty_grid = main_uncertainty_map[0, 0][region_mask_tensor]


        mask_foreground_ratio = (mask_grid > foreground_threshold).float().mean().item()
        unet_foreground_ratio = (x_unet_mask_grid > foreground_threshold).float().mean().item()
        
        if mask_foreground_ratio > min_foreground_ratio and unet_foreground_ratio > min_foreground_ratio:
            # Calculate uncertainty map and edge map Dynamic thresholds
            percentile_90 = torch.quantile(aux_uncertainty_grid, 0.90).item()
            main_unc_percentile_90 = torch.quantile(main_uncertainty_grid, 0.90).item()
            percentile_sobel_50 = torch.quantile(sobel_map_grid, 0.50).item()

            # candidate prompting points
            high_uncertainty_mask = aux_uncertainty_grid >= percentile_90
            high_main_uncertainty_mask = main_uncertainty_grid >= main_unc_percentile_90
            edge_condition = sobel_map_grid >= percentile_sobel_50
            combined_mask = edge_condition & high_uncertainty_mask&high_main_uncertainty_mask
            candidate_indices = torch.where(combined_mask)[0]

            for idx in candidate_indices:
                coord = np.argwhere(region_mask)[idx]
                x, y = coord[1], coord[0]
             
                vote_pos = int(mask[0, 0, y, x] < foreground_threshold) + \
                           int(x_aux_mask[0, 0, y, x] < foreground_threshold)

                point_type = "uncertainty-negative" if vote_pos >= 2 else "uncertainty-positive"

                point = {
                    "coord": torch.tensor([x, y], device=mask.device),
                    "type": point_type,
                    "value": mask_grid[idx],
                    "unet_value": x_unet_mask_grid[idx],
                    "uncertainty": aux_uncertainty_grid[idx],
                    "main_uncertainty": main_uncertainty_grid[idx],
                    "edge_value": sobel_map_grid[idx],
                    "score":main_uncertainty_grid[idx] + aux_uncertainty_grid[idx] + 3*sobel_map_grid[idx]
                }
                high_uncertainty_candidates.append(point)

    # Apply global NMS to control spatial distribution
    high_uncertainty_candidates = global_uncertainty_nms(high_uncertainty_candidates, radius=nms_radius)
    return high_uncertainty_candidates

def global_uncertainty_nms(points, radius=12):
    if len(points) == 0:
        return []

    points = sorted(points, key=lambda x: -x["score"])
    keep = []
    occupied = torch.zeros((512, 512), dtype=torch.bool).to(points[0]["coord"].device)

    for pt in points:
        x, y = pt["coord"].tolist()
        y1, y2 = max(y - radius, 0), min(y + radius + 1, 512)
        x1, x2 = max(x - radius, 0), min(x + radius + 1, 512)

        if not occupied[y1:y2, x1:x2].any():
            keep.append(pt)
            occupied[y1:y2, x1:x2] = True

    return keep

def get_grid_type(mask_ratio, unet_ratio, diff_thresh=0.3):
    diff = abs(mask_ratio - unet_ratio)
    ratio = 0.2
    if mask_ratio > ratio and unet_ratio > ratio:
        return 'both'
    elif mask_ratio > ratio and unet_ratio < ratio:
        if diff >= diff_thresh:
            return 'mask_only'
        else:
            return 'both'
    elif mask_ratio < ratio and unet_ratio > ratio:
        if diff >= diff_thresh:
            return 'unet_only'
        else:
            return 'both'
    else:
        return 'background'
    
def add_points(indices, scores, point_type, i, j, grid_size, mask_data, unet_data, uncertainty_data, device):
    selected = []
    if len(scores) == 0:
        return selected

    # Sort scores in descending order and get indices
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    num_candidates = len(sorted_scores)

    # Calculate top 10% and median
    idx_90 = int(0.1 * num_candidates)
    idx_50 = int(0.5 * num_candidates)

    # Select two quantiles`
    for idx in [idx_90, idx_50]:
        if idx >= num_candidates:
            continue  # Prevent out-of-bounds`
        y, x = indices[0][sorted_indices[idx]].item(), indices[1][sorted_indices[idx]].item()
        selected.append({
            "coord": torch.tensor([j * grid_size + x, i * grid_size + y], device=device),
            "type": "confidence-positive",
            "score": scores[sorted_indices[idx]].item(),
            "mask_value": mask_data[y, x].item(),
            "unet_value": unet_data[y, x].item(),
            "uncertainty": uncertainty_data[y, x].item()
        })

    return selected

def select_points_grid(mask, uncertainty_map, main_uncertainty_map, x_unet_mask, grid_size=32):
    device = mask.device
    _, _, height, width = uncertainty_map.shape
    high_confidence_points = []

    for i in range(height // grid_size):
        for j in range(width // grid_size):
            grid_slice = (slice(0, 1), slice(0, 1),
                          slice(i * grid_size, (i + 1) * grid_size),
                          slice(j * grid_size, (j + 1) * grid_size))
            mask_grid = mask[grid_slice].squeeze()
            x_unet_grid = x_unet_mask[grid_slice].squeeze()
            main_uncertainty_grid = main_uncertainty_map[grid_slice].squeeze()
            uncertainty_grid = uncertainty_map[0, 0, i * grid_size:(i + 1) * grid_size,
                                               j * grid_size:(j + 1) * grid_size]

            mask_ratio = (mask_grid > 0.5).float().mean()
            unet_ratio = (x_unet_grid > 0.5).float().mean()
            
            grid_type = get_grid_type(mask_ratio, unet_ratio)

            if grid_type == 'background':
                continue

            if grid_type in ['both', 'both_soft']:
                dynamic_threshold_mask_50 = max(mask_grid.flatten().quantile(0.5).item(), 0.5)
                dynamic_threshold_mask_95 = max(mask_grid.flatten().quantile(0.90).item(), 0.5)
                dynamic_threshold_unet_50 = max(x_unet_grid.flatten().quantile(0.5).item(), 0.5)
                dynamic_threshold_unet_95 = max(x_unet_grid.flatten().quantile(0.90).item(), 0.5)
                mask_condition_1 = (mask_grid > dynamic_threshold_mask_95) & (x_unet_grid > dynamic_threshold_unet_50)
                mask_condition_2 =  (mask_grid > dynamic_threshold_mask_50) & (x_unet_grid > dynamic_threshold_unet_95)
                condition = mask_condition_1|mask_condition_2
                if torch.any(condition):
                    indices = torch.where(condition)
                    scores = 0.5 * mask_grid[indices] + 0.5 * x_unet_grid[indices]
                    high_confidence_points += add_points(indices, scores, "both", i, j, grid_size,
                                                         mask_grid, x_unet_grid, main_uncertainty_grid, device)

            elif grid_type == 'mask_only':
                condition = ((mask_grid > 0.8) &                             # Confident main mask
                                (main_uncertainty_grid < 0.2) 
                             & (x_unet_grid > 0.3) & (x_unet_grid < 0.5)          # Weak opposition from auxiliary mask
                            )
                if torch.any(condition):
                    indices = torch.where(condition)
                    scores = 0.7 * mask_grid[indices] + 0.3 * x_unet_grid[indices]
                    high_confidence_points += add_points(indices, scores, "mask_only", i, j, grid_size,
                                                         mask_grid, x_unet_grid, main_uncertainty_grid, device)

            elif grid_type == 'unet_only':
                condition = (
                                (x_unet_grid > 0.8) &                             # Confident auxiliary mask
                                (uncertainty_grid < 0.1) 
                                &(mask_grid > 0.3) & (mask_grid < 0.5)            # Weak opposition from main mask
                            )
                if torch.any(condition):
                    indices = torch.where(condition)
                    scores = 0.7 * x_unet_grid[indices] + 0.3 * mask_grid[indices]
                    high_confidence_points += add_points(indices, scores, "unet_only", i, j, grid_size,
                                                         mask_grid, x_unet_grid, main_uncertainty_grid, device)
    # Apply global NMS to control spatial distribution
    high_confidence_points = global_nms(high_confidence_points, radius=grid_size // 4)
    return high_confidence_points

def global_nms(points, radius=8):
    if not points:
        return []

    coords = torch.stack([p["coord"] for p in points], dim=0).float()  # [N, 2]
    scores = torch.tensor([p["score"] for p in points])  # [N]

    keep = grid_nms(coords, scores, radius=radius, device=coords.device)
    return [points[i] for i in keep]



def inference(args, model,  device='cuda',isAutoPrompt=False):

    db_test = MMIS_dataset(base_dir=args.base_dir, list_dir=args.list_dir, split='test')
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
                            collate_fn=collate_fn)
    resizer = ResizeLongestSide(1024)

    criterion = BinaryMaskLoss(0.8).to(device)

    logger.info("{} test iterations per epoch".format(len(testloader)))
    model.unet_uncertainty.eval()
    model.unet.eval()
    model.sam.eval()
    with torch.no_grad():
        dice_list = []
        hausdorff_list = []
        for step, sampled_batch in enumerate(testloader):
            input, label, case_name = getInput(sampled_batch)

            input = input.to(device)
            label = label.to(device)
            origin = input
            input_unet = input

            input = resizer.apply_image_torch(input)

            # test one batch
            outputs, hint, loss1, loss2, loss, points, uncertainty_map, unet_out, segments, sobel_map,main_uncertainty_map = test_one_epoch(
                input, input_unet, label, model, criterion, origin,isAutoPrompt)

            # Obtain predicted binary mask
            outputs_binary = torch.sigmoid(outputs) > 0.5

            # Calculate evaluation metrics
            dice = dice_coefficient(outputs_binary.float(), label).item()
            dice_list.append(dice)
            hausdorff = batch_hausdorff_95(outputs, label)
            hausdorff_list.append(hausdorff)

            case_list = []
            for i in range(len(sampled_batch)):
                slice_name = sampled_batch[i]['case_name']
                case_list.append(slice_name)
                logger.info(f"slice_name:{slice_name}, dice:{dice:.4f}, hausdorff:{hausdorff:.4f}")

            name = case_name[0]
        average_dice = sum(dice_list)/len(dice_list)
        average_hausdorff95 = sum(hausdorff_list)/len(hausdorff_list)

        logger.info(f" average_dice:average_hausdorff95 {average_dice:.4f} {average_hausdorff95:.4f}")
    return "Testing Finished!"


def test_one_epoch(input,input_unet, label, model, criterion,origin,isAutoPrompt):
    outputs, hint, points, uncertainty_map, unet_out, segments,cnn_out,sobel_map,main_uncertainty_map = model(input, input_unet,origin,isAutoPrompt)
    pred_mask = torch.sigmoid(outputs)
    pred_hint = torch.sigmoid(hint)
    pred_cnn_mask = torch.sigmoid(cnn_out)
    loss1 = criterion(pred_mask, label)
    loss3 = criterion(pred_hint, label)
    loss2 = criterion(pred_cnn_mask, label)


    loss = loss1 + 0.5 * loss2 + loss3
    return outputs, hint, loss1, loss2, loss, points, uncertainty_map, unet_out, segments,sobel_map,main_uncertainty_map

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Create network model
    model_type = "vit_b"
    logger.info("test_MMIS")
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    sam = sam_model_registry[model_type]()
    unet = UNet_Uncertainty_V2(n_channels=3, n_classes=1)
    unet_uncertainty = UNet_Uncertainty_V2(n_channels=3, n_classes=1)

    # Load pre-trained weights
    weight_name = args.pre_weights.split("\\")[-1]
    weight_name_unet = args.pre_weights_unet.split("\\")[-1]

    if args.pre_weights is not None:
        logger.info(f"Load pre-trained weights：{weight_name}")
        model_pth = torch.load(args.pre_weights)
        missing_key, unexpected_key = sam.load_state_dict(model_pth, strict=False)


    if args.pre_weights_unet is not None:
        logger.info(f"Load pre-trained weights：{weight_name_unet}")
        unet_pth = torch.load(args.pre_weights_unet)
        unet_missing_key, unet_unexpected_key = unet.load_state_dict(unet_pth, strict=False)
        unet_missing_key, unet_unexpected_key = unet_uncertainty.load_state_dict(unet_pth, strict=False)


    model = Model(sam, unet, unet_uncertainty).to(device)

    inference(args, model, device=device,isAutoPrompt=args.isAutoPrompt)


def preprocess(x):
    # resize
    transform_resize = transforms.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR)
    x = transform_resize(x)
    return x


def postprocess_masks(
        masks,
        input_size=(256, 256),
        original_size=(512, 512),
):
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


def postprocess_promptmasks(
        masks,
        input_size=(512, 512),
        original_size=(1024, 1024),
):
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


def getInput(sampled_batch):
    img_batch = []
    case_names = []
    origin_batch = []
    for imgs in sampled_batch:
        img_single = imgs['image']
        img_batch.append(img_single)
        case_names.append(imgs['case_name'])
    input = torch.stack(img_batch, dim=0)
    label_batch = []
    for labels in sampled_batch:
        label_single = labels['label']
        label_batch.append(label_single)
    label = torch.stack(label_batch, dim=0)
    return input, label, case_names




def collate_fn(batch):
    batch_size = len(batch)
    return batch



def convert_to_tuple(
        points: List[Dict[str, torch.Tensor]]

) -> Tuple[torch.Tensor, torch.Tensor]:

    coords = [point["coord"] for point in points]
    # `# Set labels according to the type of prompting points`
    labels = [torch.tensor(1) if "positive" in point["type"] else torch.tensor(0) for point in points]

    coord_tensor = torch.stack(coords) if coords else torch.empty(0)
    label_tensor = torch.stack(labels) if labels else torch.empty(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coord_tensor = coord_tensor.unsqueeze(0).to(device)
    label_tensor = label_tensor.unsqueeze(0).to(device)

    return coord_tensor, label_tensor

def generate_edge_map(origin,  device='cuda'):
    image_np = origin[0, 0].cpu().numpy()
    sobel_map = filters.sobel(image_np)
    sobel_map = torch.from_numpy(sobel_map).to(device)
    sobel_min, sobel_max = sobel_map.min(), sobel_map.max()
    sobel_map = (sobel_map - sobel_min) / (sobel_max - sobel_min + 1e-6)
    log_map = laplace(gaussian(image_np, sigma=1.5))
    log_map = torch.from_numpy(log_map).to(device)
    log_min, log_max = log_map.min(), log_map.max()
    log_map = (log_map - log_min) / (log_max - log_min + 1e-6)
    final_edge_map = (0.5 * sobel_map + 0.5 * log_map).unsqueeze(0).unsqueeze(0)
    return final_edge_map



def generate_superpixels(image, n_segments=100):
    if image.ndim == 4:
        image = image[0].permute(1, 2, 0).detach().cpu().numpy()
    elif image.ndim == 3:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    segments = slic(image, n_segments=n_segments, compactness=5)
    return segments

def grid_nms(coords, scores, radius=8, device='cuda'):

    if coords.size(0) == 0:
        return []

    # Sort by score in descending order`
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    sorted_coords = coords[sorted_indices]

    # Calculate distance matrix`
    y = sorted_coords[:, 0].unsqueeze(1)
    x = sorted_coords[:, 1].unsqueeze(1)
    dist_matrix = torch.sqrt((y - y.t()) ** 2 + (x - x.t()) ** 2)

    # Initialize retention markers`
    keep = torch.ones(len(sorted_coords), dtype=torch.bool).to(device)

    for i in range(len(sorted_coords)):
        if not keep[i]:
            continue
        # Suppress neighboring points within suppression radius`
        suppress = (dist_matrix[i] < radius) & (keep)
        suppress[:i + 1] = False
        keep[suppress] = False

    return sorted_indices[keep.to(sorted_indices.device)]


def normalize_uncertainty(uncertainty, method='log'):
    if method == 'log':
        uncertainty = torch.log(uncertainty + 1e-10)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    elif method == 'zscore':
        mean = uncertainty.mean()
        std = uncertainty.std()
        uncertainty = (uncertainty - mean) / std
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    else:
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        uncertainty = torch.clamp(uncertainty, min=0.0, max=1.0)
    return uncertainty

def predict_with_uncertainty(unet, x, unet_uncertainty, n_samples=10):
    unet_uncertainty.eval()
    # Normal prediction`
    mask = unet(x, apply_drop=False)
    predictions = []
    # Dropout prediction
    with torch.no_grad():
        for i in range(n_samples):
            output = unet_uncertainty(x, apply_drop=True)
            output = torch.sigmoid(output).detach()
            predictions.append(output)
    predictions = torch.stack(predictions)
    uncertainty = torch.var(predictions, dim=0)
    uncertainty = normalize_uncertainty(uncertainty, method=' ').to('cuda')
    return mask, uncertainty





if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=args.output_dir + '/testLog.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    main(args)