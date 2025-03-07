# train.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from monai.losses import DiceCELoss
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import (
    get_train_data_loader,
    get_data_loader,
    dice,
)
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.data import DataLoader  # Use MONAI's DataLoader

# Import the models
from networks.Unet import UNet3D  # Assuming this is your UNet3D implementation
from networks.Vit import ViT3DSegmentation
from networks.transunet import TransUNet3D
from networks.segresnet import SegResNet
from networks.resunet import ResUnet



def main():
    parser = argparse.ArgumentParser(description='Train model on BTCV dataset.')
    parser.add_argument('--data_path', type=str,
                        default='/data2/Chongyu/WholeBrainSeg/BTCV/data/train/images',
                        help='Path to training image data directory.')
    parser.add_argument('--label_path', type=str,
                        default='/data2/Chongyu/WholeBrainSeg/BTCV/data/train/labels',
                        help='Path to training label data directory.')
    parser.add_argument('--val_data_path', type=str,
                        default='/data2/Chongyu/WholeBrainSeg/BTCV/data/test/images',
                        help='Path to validation image data directory.')
    parser.add_argument('--val_label_path', type=str,
                        default='/data2/Chongyu/WholeBrainSeg/BTCV/data/test/labels',
                        help='Path to validation label data directory.')
    parser.add_argument('--output_dir', type=str, default='./output_segresnet',
                        help='Directory to save models and logs.')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate.')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Number of epochs between saving model checkpoints.')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Number of epochs between evaluations.')
    parser.add_argument('--roi_size', type=int, default=96,
                        help='ROI size for training and evaluation.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--num_classes', type=int, default=14,
                        help='Number of segmentation classes.')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to resume checkpoint file.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet3d', 'vit', 'transunet','segresnet','resunet'],
                        help='Model type to train: unet3d, vit, or transunet.')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for ViT and TransUNet models.')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension for ViT model.')
    parser.add_argument('--depth', type=int, default=12,
                        help='Depth of Transformer encoder for ViT model.')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads in ViT model.')
    parser.add_argument('--mlp_dim', type=int, default=3072,
                        help='MLP dimension in ViT model.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in ViT model.')
    # Add new arguments for data loader parameters
    parser.add_argument('--a_min', type=float, default=-175.0,
                        help='Minimum intensity value for scaling.')
    parser.add_argument('--a_max', type=float, default=250.0,
                        help='Maximum intensity value for scaling.')
    parser.add_argument('--b_min', type=float, default=0.0,
                        help='Minimum output intensity value after scaling.')
    parser.add_argument('--b_max', type=float, default=1.0,
                        help='Maximum output intensity value after scaling.')
    parser.add_argument('--num_samples_per_image', type=int, default=2,
                        help='Number of samples to extract per image.')
    args = parser.parse_args()

    # Set up distributed training if specified
    if args.distributed:
        # Initialize the process group
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"Distributed training with {world_size} GPUs. "
              f"Current process rank: {rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        rank = 0  # Main process

    # Create output directory if it doesn't exist (only on main process)
    if rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # TensorBoard writer (only on main process)
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    else:
        writer = None

    # Check roi_size compatibility
    if args.model in ['vit', 'transunet']:
        if args.roi_size % args.patch_size != 0:
            raise ValueError(
                f"For {args.model} model, roi_size must be divisible by patch_size."
            )

    # Instantiate the model based on the selected type
    if args.model == 'unet3d':
        model = UNet3D(
            n_channels=1,
            n_classes=args.num_classes,
            base_features=16
        ).to(device)
    elif args.model == 'resunet':
        model = ResUnet(
            n_channels=1,
            n_classes=args.num_classes,
            base_features=16
        ).to(device)
    elif args.model == 'vit':
        model = ViT3DSegmentation(
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            in_channels=1,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            num_classes=14,
            dropout=0.1
        ).to(device)
    elif args.model == 'transunet':
    
        model = TransUNet3D(
            img_size=(args.roi_size, args.roi_size, args.roi_size),
            in_channels=1,
            num_classes=args.num_classes,
            base_channels=args.patch_size,
            embed_dim=args.embed_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            num_layers=args.depth,
            dropout=args.dropout
        ).to(device)
    elif args.model == 'segresnet':
        model = SegResNet(
            in_channels=1,
            num_classes=args.num_classes,
            init_features=16  
        ).to(device)

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    # Wrap the model with DDP if distributed training
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"Model wrapped with DistributedDataParallel on device {local_rank}")

    # Define the loss function and optimizer
    criterion = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, batch=True
    )

    if args.model == 'unet3d' or args.model == 'segresnet':
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-5
        )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_checkpoint is not None:
        if os.path.isfile(args.resume_checkpoint):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(
                args.resume_checkpoint, map_location=map_location
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Process {rank}: Resumed training from checkpoint at epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume_checkpoint}'")

    # Determine shuffle value based on whether distributed training is enabled
    if args.distributed:
        shuffle = False  # Disable shuffle; shuffling is handled by DistributedSampler
    else:
        shuffle = True  # Enable shuffle when not using distributed training

    # Prepare training data loader using utils.py
    train_loader = get_train_data_loader(
        data_path=args.data_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        roi_depth=args.roi_size,
        roi_height=args.roi_size,
        roi_width=args.roi_size,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        num_samples_per_image=args.num_samples_per_image,
        transforms=None  # Use default transforms from utils.py
    )

    # Apply DistributedSampler if distributed training
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        # Create new DataLoader with the DistributedSampler
        train_loader = DataLoader(
            dataset=train_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Shuffle is handled by the sampler
            num_workers=args.num_workers,
            sampler=train_sampler,
            pin_memory=True,
        )
    else:
        train_sampler = None

    # Prepare validation data loader using utils.py (only on main process)
    if rank == 0:
        val_loader = get_data_loader(
            data_path=args.val_data_path,
            label_path=args.val_label_path,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            roi_depth=args.roi_size,
            roi_height=args.roi_size,
            roi_width=args.roi_size,
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            transforms=None  # Use default transforms from utils.py
        )
    else:
        val_loader = None

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0

        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(total=len(train_loader),
                        desc=f"Epoch [{epoch+1}/{args.num_epochs}]",
                        ncols=100)
        else:
            pbar = None

        for batch_data in train_loader:
            images = batch_data['image'].to(device, non_blocking=True)
            labels = batch_data['label'].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar
            if rank == 0:
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)

        # Close progress bar
        if rank == 0:
            pbar.close()

        # Average loss over epoch
        epoch_loss /= len(train_loader)

        # Reduce loss across all processes if distributed
        if args.distributed:
            # Create a tensor of the epoch_loss
            loss_tensor = torch.tensor(epoch_loss).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor.item() / world_size

        # Write loss to TensorBoard (only main process)
        if rank == 0 and writer is not None:
            writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Print loss
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], "
                  f"Average Loss: {epoch_loss:.4f}")

        # Evaluate every eval_interval epochs
        if ((epoch + 1) % args.eval_interval == 0) and rank == 0:
            model.eval()
            dice_list_case = []
            with torch.no_grad():
                val_pbar = tqdm(total=len(val_loader),
                                desc="Validation", ncols=100)
                for val_data in val_loader:
                    val_images = val_data['image'].to(device,
                                                      non_blocking=True)
                    val_labels = val_data['label'].to(device,
                                                      non_blocking=True)
                    # Use sliding window inference
                    val_outputs = sliding_window_inference(
                        val_images,
                        roi_size=(args.roi_size, args.roi_size,
                                  args.roi_size),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        mode='gaussian'
                    )

                    # Compute dice
                    outputs_seg = torch.softmax(val_outputs, dim=1
                                                ).cpu().numpy()
                    outputs_seg = np.argmax(outputs_seg, axis=1
                                            ).astype(np.uint8)
                    labels = val_labels.cpu().numpy()[:, 0, :, :, :]
                    dice_list_sub = []
                    num_classes = args.num_classes
                    for organ_idx in range(1, num_classes):
                        organ_Dice = dice(
                            outputs_seg[0] == organ_idx,
                            labels[0] == organ_idx
                        )
                        dice_list_sub.append(organ_Dice)
                    mean_dice = np.mean(dice_list_sub)
                    dice_list_case.append(mean_dice)

                    # Update validation progress bar
                    val_pbar.set_postfix({'Mean Dice': mean_dice})
                    val_pbar.update(1)

                val_pbar.close()
                mean_dice_all = np.mean(dice_list_case)
                if writer is not None:
                    writer.add_scalar('Dice/validation', mean_dice_all, epoch)
                print(f"Epoch [{epoch+1}/{args.num_epochs}], "
                      f"Validation Mean Dice: {mean_dice_all:.4f}")

        # Save model checkpoint every save_interval epochs
        if ((epoch + 1) % args.save_interval == 0) and rank == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'{args.model}_checkpoint_epoch_{epoch+1}.pth'
            )
            state_dict = (
                model.module.state_dict() if args.distributed
                else model.state_dict()
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved model checkpoint at {checkpoint_path}")

    if rank == 0 and writer is not None:
        writer.close()

    # Clean up
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
