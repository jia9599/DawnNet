# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import random
from torch import nn
from main.dataset.dataset_MMIS import MMIS_dataset
from main.loss import BinaryMaskLoss
from main.metrics import dice_coefficient, batch_hausdorff_95
from torch.optim import lr_scheduler
join = os.path.join
from torchvision.transforms import InterpolationMode
from segment_anything import sam_model_registry
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Tuple, Dict
from segment_anything.utils.transforms import ResizeLongestSide

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='pytorch sam training')
    parser.add_argument("--base-dir", default='./data/MMIS/TC')
    parser.add_argument("--list-dir", default='./lists/lists_MMIS')
    parser.add_argument("--output-dir", default='./out_dir')
    parser.add_argument('--fold', default=3, type=int, help='5-fold, default is none')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="1", type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--scheduler-type", default="StepLR", help="Default attenuation once every 10 rounds")
    parser.add_argument("--weight-decay", default="1e-4", type=float)
    parser.add_argument("--eval-interval", default=1, type=int, help="validation interval default 10 Epochs")
    parser.add_argument("--lr", default='0.001', type=float, help='initial learning rate')
    parser.add_argument("--debug", default=False, type=bool, help='whether only use the first batch')
    parser.add_argument('--freeze-layers', default=True, type=bool)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--seed', default=42, type=int, help='Random Number Seed ')
    parser.add_argument('--pre-weights',
                      default='./pre_weights/sam_vit_b_01ec64.pth',
                      help='pre_weights')
    args = parser.parse_args()
    return args

args = parse_args()

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Create network model
    model_type = "vit_b"
    logger.info("train_MMIS")
    sam = sam_model_registry[model_type]()
    resizer = ResizeLongestSide(1024)
    # Load pre-trained weights
    weight_name = args.pre_weights.split("\\")[-1]

    if args.pre_weights is not None:
        logger.info(f"Load pre-trained weights：{weight_name}")
        model_pth = torch.load(args.pre_weights)
        missing_key, unexpected_key = sam.load_state_dict(model_pth, strict=False)


    model = Model(sam).to(device)

    # Freeze weights
    num_param = 0
    train_list = []
    train_para = []
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "image_encoder.specific" in name:
                para.requires_grad_(True)
                num_param += para.numel()
                train_list.append(name)
                train_para.append(para)
            else:
                para.requires_grad_(False)
    logger.info(f"train param：{num_param}")


    if args.debug:
        args.batch_size = 1
        args.epochs = 500


    train_data = MMIS_dataset(base_dir=args.base_dir, list_dir=args.list_dir, split='train',fold=args.fold)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False,
                              collate_fn=collate_fn)
    n_train = len(train_data)
    val_data = MMIS_dataset(base_dir=args.base_dir, list_dir=args.list_dir, split='val', fold=args.fold)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False,
                            collate_fn=collate_fn)
    n_val = len(val_data)

    if args.debug:
        train_loader = [next(iter(train_loader))]
        val_loader = [next(iter(val_loader))]

    # Define loss functions and optimizers
    criterion = BinaryMaskLoss(0.8).to(device)
    optimizer = torch.optim.AdamW(train_para, lr=args.lr, weight_decay=args.weight_decay)

    # Define learning rate decay strategy
    if args.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    elif args.scheduler_type == 'ReduceLR':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif args.scheduler_type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0001)
    else:
        scheduler = None

    # train stage
    model.train()
    logger.info(f'''Starting training:
           Fold:            {args.fold}
           Pre_weights:     {weight_name}
           Epochs:          {args.epochs}
           Batch size:      {args.batch_size}
           Learning rate:   {args.lr}
           Training size:   {n_train}
           Validation size: {n_val}
           Device:          {device}
       ''')
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    train_hausdorffs = []
    val_hausdorffs = []
    num_train_batches = 0
    best_dice = 0.0
    for epoch in (range(args.start_epoch, args.epochs)):
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_dice = 0.0
        total_hausdorff = 0.0
        numPreAllZero = 0

        for step, sampled_batch in enumerate(train_loader):

            input, label, case_name = getInput(sampled_batch)

            input = input.to(device)
            label = label.to(device)
            origin = input

            input = preprocess(input)

            # Train for one epoch (accurately, one batch)
            outputs, hint, loss1, loss2, loss= train_one_epoch(input, label, model, optimizer, criterion,origin,epoch)
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

            # Obtain predicted binary mask
            outputs_binary = torch.sigmoid(outputs) > 0.5

            # Calculate evaluation metrics
            total_dice += dice_coefficient(outputs_binary.float(), label).item()
            hausdorff = batch_hausdorff_95(outputs, label)
            if np.isnan(hausdorff):
                numPreAllZero += 1
            else:
                total_hausdorff += hausdorff
                num_train_batches += 1

        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            if (epoch + 1) % 10 == 0 :
                lr = optimizer.param_groups[0]['lr']
                logger.info(f"current epoch ：{epoch}，current learning rate：{lr}")

        # Average metrics per epoch
        if num_train_batches > 0:
            epoch_hausdorff = total_hausdorff / num_train_batches
            train_hausdorffs.append(epoch_hausdorff)
        else:
            epoch_hausdorff = float('nan')
            logger.warning(f"Epoch {epoch}: All batches predicted zeros for Hausdorff calculation!")


        # dice
        epoch_dice = total_dice / (len(train_loader))
        train_dices.append(epoch_dice)

        # loss
        epoch_loss = total_loss / (len(train_loader))
        train_losses.append(epoch_loss)

        # Validation stage
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                model.eval()
                total_val_loss = 0
                total_val_loss1 = 0
                total_val_loss2 = 0
                total_val_dice = 0.0
                total_val_hausdorff = 0.0
                val_numPreAllZero = 0
                num_valid_batches = 0
                for step, sampled_batch in enumerate(val_loader):

                    input, label, case_name = getInput(sampled_batch)
                    input = input.to(device)
                    label = label.to(device)
                    origin = input
                    input = resizer.apply_image_torch(input)

                    # val for one epoch (accurately, one batch)
                    outputs, hint, loss1, loss2, loss = val_one_epoch(
                        input, label, model, criterion, origin, epoch)
                    total_val_loss += loss.item()
                    total_val_loss1 += loss1.item()
                    total_val_loss2 += loss2.item()

                    # Obtain predicted binary mask
                    outputs_binary = torch.sigmoid(outputs) > 0.5

                    # Calculate evaluation metrics
                    total_val_dice = total_val_dice + dice_coefficient(outputs_binary.float(), label).item()
                    hausdorff = batch_hausdorff_95(outputs, label)
                    if np.isnan(hausdorff):
                        val_numPreAllZero += 1
                    else:
                        total_val_hausdorff += hausdorff
                        num_valid_batches += 1

                # Average metrics per epoch
                if num_valid_batches > 0:
                    epoch_val_hausdorff = total_val_hausdorff / num_valid_batches
                    val_hausdorffs.append(epoch_val_hausdorff)
                else:
                    epoch_val_hausdorff = float('nan')
                    logger.warning(f"Epoch {epoch}: All batches predicted zeros for Hausdorff calculation!")

                # dice
                epoch_val_dice = total_val_dice / (len(val_loader))
                val_dices.append(epoch_val_dice)

                # loss
                epoch_val_loss = total_val_loss / len(val_loader)
                val_losses.append(epoch_val_loss)

        # Save the best model
        if epoch >= 5:
            if epoch_val_dice > best_dice:
                best_dice = epoch_val_dice
                save_mode_path = os.path.join(args.output_dir, 'bestModel')
                torch.save(model.sam.state_dict(), save_mode_path)
                logger.info(f"New best model saved to {save_mode_path} with Dice: {best_dice:.4f} epoch: {epoch}")
        # print log
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            logger.info(f"EPOCH:{epoch}  " +
                        f"train_loss:{epoch_loss:.4f}   train_dice:{epoch_dice:.4f}  " +
                        f"train_hd95:{epoch_hausdorff:.4f}  val_loss:{epoch_val_loss:.4f} " +
                        f"val_dice:{epoch_val_dice:.4f}  val_hd95:{epoch_val_hausdorff:.4f} ")
        else:
            logger.info(
                f"EPOCH:{epoch:.4f}  train_loss:{epoch_loss:.4f}  train_dice:{epoch_dice:.4f}  train_hd95:{epoch_hausdorff:.4f}")


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


def getInput(sampled_batch):
    img_batch = []
    case_names = []
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



def train_one_epoch(input,  label, model, optimizer, criterion, origin, epoch):
    outputs, promptMask, cnn_out = model(input, origin)

    pred_mask = torch.sigmoid(outputs)
    pred_hint = torch.sigmoid(promptMask)
    pred_cnn_mask = torch.sigmoid(cnn_out)

    # Calculate loss
    loss1 = criterion(pred_mask, label)
    loss2 = criterion(pred_cnn_mask, label)
    loss3 = criterion(pred_hint, label)
    loss = loss1 + 0.5*loss2 + loss3

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return outputs, promptMask, loss1, loss2, loss


def val_one_epoch(input,  label, model, criterion, origin, epoch):
    outputs, promptMask, cnn_out = model(input,origin)
    pred_mask = torch.sigmoid(outputs)
    pred_hint = torch.sigmoid(promptMask)
    pred_cnn_mask = torch.sigmoid(cnn_out)
    loss1 = criterion(pred_mask, label)
    loss2 = criterion(pred_cnn_mask, label)
    loss3 = criterion(pred_hint, label)
    loss = loss1 + 0.5 * loss2 + loss3
    return outputs, promptMask, loss1, loss2, loss


class Model(nn.Module):
    def __init__(self, sam):
        super(Model, self).__init__()
        self.sam = sam
        self.resizer = ResizeLongestSide(1024)
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, origin):
        img_embedding, spgen1, spgen2, spgen3, spgen4, cnn_out = self.sam.image_encoder(x, origin)

        spgen1 = self.up1(spgen1)
        spgen2 = self.up2(spgen2)
        spgen3 = self.up3(spgen3)

        autoMask = postprocess_masks(spgen1 + spgen2 + spgen3 + spgen4)

        outputs_mask = []
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
        return masks, autoMask, cnn_out


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=args.output_dir + '/trainLog.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    args = parse_args()
    main(args)

