import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders import MonaiSurvivalDataset
from architectures import build_network
from optimizers import create_optimizer_and_scheduler
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain 3D ResNet to predict features from DICOM images. By default, uses all available features.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with DICOM paths and features')
    parser.add_argument('--nifti_dir', type=str, required=True, help='Directory containing NIFTI files')
    parser.add_argument('--fold_column', type=str, default='Fold_1', help='Column name for fold split (default: Fold_1)')
    parser.add_argument('--train_fold', type=str, default='train', help='Value in fold column for training data (default: train)')
    parser.add_argument('--val_fold', type=str, default='val', help='Value in fold column for validation data (default: val)')
    parser.add_argument('--resnet', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='ResNet architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--feature_cols', type=str, nargs='+', default=None, help='List of feature columns to predict (default: all columns except NIFTI path, time, event, and fold columns)')
    parser.add_argument('--image_size', type=int, nargs=3, default=[256, 256, 64], help='Image size (D, H, W)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output', type=str, default='pretrain_logs/pretrained_resnet.pth', help='Path to save the pretrained model')
    parser.add_argument('--log_dir', type=str, default='pretrain_logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for model')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor')
    parser.add_argument('--persistent_workers', action='store_true', help='Use persistent workers in DataLoader')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory in DataLoader')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adamw', 'adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'cosine_warm_restarts', 'onecycle', 'none'], 
                       help='Learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                       help='Maximum gradient norm for clipping (0 to disable)')
    parser.add_argument('--eta_min', type=float, default=1e-7, 
                       help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
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

    # Create temporary CSV with full NIFTI paths
    import pandas as pd
    import tempfile
    
    # Load the original CSV
    if args.csv_path.endswith('.xlsx') or args.csv_path.endswith('.xls'):
        df = pd.read_excel(args.csv_path)
    else:
        df = pd.read_csv(args.csv_path)
    
    # Update NIFTI paths to full paths
    df['NIFTI Path'] = df['NIFTI Path'].apply(lambda x: os.path.join(args.nifti_dir, x))
    
    # Create temporary CSV file
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    try:
        # Create training dataset
        train_dataset = MonaiSurvivalDataset(
            csv_path=temp_csv.name,
            feature_cols=args.feature_cols,
            image_size=tuple(args.image_size),
            use_cache=False,  # For large datasets, avoid caching all in RAM
            augment=True,
            fold_column=args.fold_column,
            fold_value=args.train_fold
        )
        
        # Create validation dataset
        val_dataset = MonaiSurvivalDataset(
            csv_path=temp_csv.name,
            feature_cols=args.feature_cols,
            image_size=tuple(args.image_size),
            use_cache=False,
            augment=False,  # No augmentation for validation
            fold_column=args.fold_column,
            fold_value=args.val_fold
        )
    finally:
        # Clean up temporary file
        os.unlink(temp_csv.name)
    
    # Get the fitted feature scaler for saving (from training dataset)
    feature_scaler = train_dataset.get_feature_scaler()
    
    # Print which features are being used
    if args.feature_cols is None:
        print(f"Using all available features (auto-detected): {len(train_dataset.data[0]['features'])} features")
    else:
        print(f"Using specified features: {args.feature_cols}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    x0, _, _, _, _ = train_dataset[0]
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

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model.parameters(),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        eta_min=args.eta_min
    )
    
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Dataset returns: features, time, event, image, dicom_path
            features, _, _, images, _ = batch
            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, features)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features, _, _, images, _ = batch
                images = images.to(device, non_blocking=True)
                features = features.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(images)
                    loss = criterion(outputs, features)
                val_loss += loss.item() * images.size(0)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/val', avg_val_loss, epoch + 1)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

        # Save checkpoint if best validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint_data = {
                'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                'feature_scaler': feature_scaler,
                'feature_columns': args.feature_cols,
                'resnet_type': args.resnet,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved new best model checkpoint to {checkpoint_path} (Val Loss: {avg_val_loss:.4f})")

    # Save final model
    final_model_data = {
        'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
        'feature_scaler': feature_scaler,
        'feature_columns': args.feature_cols,
        'resnet_type': args.resnet,
        'epoch': args.epochs,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    }
    torch.save(final_model_data, args.output)
    print(f"Pretrained model saved to {args.output}")
    writer.close()

if __name__ == "__main__":
    main()
