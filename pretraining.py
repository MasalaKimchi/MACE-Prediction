import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from dataset import MonaiSurvivalDataset
from network import build_network
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain 3D ResNet to predict features from DICOM images.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with DICOM paths and features')
    parser.add_argument('--resnet', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='ResNet architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--feature_cols', type=str, nargs='+', required=True, help='List of feature columns to predict')
    parser.add_argument('--image_size', type=int, nargs=3, default=[256, 256, 64], help='Image size (D, H, W)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output', type=str, default='pretrain_logs/pretrained_resnet.pth', help='Path to save the pretrained model')
    parser.add_argument('--log_dir', type=str, default='pretrain_logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for model')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor')
    parser.add_argument('--persistent_workers', action='store_true', help='Use persistent workers in DataLoader')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory in DataLoader')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.log_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.log_dir, 'checkpoint.pth')
    writer = SummaryWriter(log_dir=args.log_dir)

    # Precautions for large datasets (100,000 images):
    # - Set use_cache=False or cache_rate < 1.0 in MonaiSurvivalDataset to avoid OOM
    # - Use high num_workers for DataLoader, but not more than CPU cores
    # - Use pin_memory=True for DataLoader if using CUDA
    # - Save checkpoints and logs frequently to avoid data loss
    # - Monitor disk space for logs and checkpoints
    # - Consider distributed training for even better scaling

    dataset = MonaiSurvivalDataset(
        csv_path=args.csv_path,
        feature_cols=args.feature_cols,
        image_size=tuple(args.image_size),
        use_cache=False,  # For large datasets, avoid caching all in RAM
        augment=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    x0, _, _, _, _ = dataset[0]
    feature_dim = x0.shape[0]

    model = build_network(resnet_type=args.resnet, in_channels=1, num_classes=feature_dim)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = DataParallel(model)
    model = model.to(device)

    if args.compile:
        try:
            model = torch.compile(model, mode='max-autotune')
            print('Enabled torch.compile')
        except Exception as e:
            print(f'torch.compile not available: {e}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_loss = float('inf')

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in loader:
            # Dataset returns: features, time, event, image, dicom_path
            features, _, _, images, _ = batch
            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, features)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)

        # Save checkpoint if best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model checkpoint to {checkpoint_path}")

    # Save final model
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), args.output)
    else:
        torch.save(model.state_dict(), args.output)
    print(f"Pretrained model saved to {args.output}")
    writer.close()

if __name__ == "__main__":
    main()
