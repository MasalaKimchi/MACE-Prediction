#!/usr/bin/env python3
"""
Run pretraining experiment with structured output in experiments folder.
Uses YAML config and organizes results with proper directory structure.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from experiments.experiment_utils import ExperimentManager, create_experiment_name
from pretrain_scripts.pretrain_features_distributed import main as pretrain_main


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_from_config(config: dict) -> ExperimentManager:
    """Create experiment manager from config."""
    experiment_name = config['experiment']['name']
    return ExperimentManager(experiment_name)


def run_pretraining_experiment(config_path: str, experiment_name: str = None):
    """
    Run pretraining experiment with structured output.
    
    Args:
        config_path: Path to YAML config file
        experiment_name: Override experiment name (optional)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override experiment name if provided
    if experiment_name:
        config['experiment']['name'] = experiment_name
    
    # Create experiment manager
    exp_manager = ExperimentManager(config['experiment']['name'])
    
    # Save configuration
    config_path_saved = exp_manager.save_config(config)
    
    # Update config paths to use experiment directory
    config['logging']['log_dir'] = exp_manager.get_log_dir()
    config['output']['final_model_path'] = str(exp_manager.experiment_dir / "checkpoints" / "final_model.pth")
    
    print(f"🚀 Starting pretraining experiment: {config['experiment']['name']}")
    print(f"📁 Experiment directory: {exp_manager.experiment_dir}")
    print(f"📝 Config saved to: {config_path_saved}")
    
    try:
        # Convert config to command line arguments for pretraining script
        args = config_to_args(config)
        
        # Set sys.argv to pass arguments to pretraining script
        import sys
        original_argv = sys.argv.copy()
        
        # Build command line arguments
        cmd_args = [
            'pretrain_features_distributed.py',
            '--csv_path', args.csv_path,
            '--fold_column', args.fold_column,
            '--train_fold', args.train_fold,
            '--val_fold', args.val_fold,
            '--resnet', args.resnet,
            '--feature_cols'] + args.feature_cols + [
            '--image_size'] + [str(x) for x in args.image_size] + [
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay),
            '--log_dir', args.log_dir,
            '--output', args.output,
            '--backend', args.backend
        ]
        
        # Add optional arguments
        if args.num_workers:
            cmd_args.extend(['--num_workers', str(args.num_workers)])
        if args.amp:
            cmd_args.append('--amp')
        if args.compile:
            cmd_args.append('--compile')
        if args.pin_memory:
            cmd_args.append('--pin_memory')
        if args.use_smart_cache:
            cmd_args.append('--use_smart_cache')
        if args.cache_rate != 0.5:
            cmd_args.extend(['--cache_rate', str(args.cache_rate)])
        if args.replace_rate != 0.5:
            cmd_args.extend(['--replace_rate', str(args.replace_rate)])
        if args.max_grad_norm != 1.0:
            cmd_args.extend(['--max_grad_norm', str(args.max_grad_norm)])
        if args.eta_min != 1e-7:
            cmd_args.extend(['--eta_min', str(args.eta_min)])
        
        # Set sys.argv for the pretraining script
        sys.argv = cmd_args
        
        # Run pretraining
        pretrain_main()
        
        # Restore original sys.argv
        sys.argv = original_argv
        
        # Mark experiment as completed
        exp_manager.mark_completed()
        print(f"✅ Pretraining experiment completed successfully!")
        
    except Exception as e:
        # Restore original sys.argv in case of error
        if 'original_argv' in locals():
            sys.argv = original_argv
        # Mark experiment as failed
        exp_manager.mark_failed(str(e))
        print(f"❌ Pretraining experiment failed: {e}")
        raise


def config_to_args(config: dict) -> argparse.Namespace:
    """Convert config dictionary to argparse.Namespace for compatibility."""
    args = argparse.Namespace()
    
    # Dataset arguments
    args.csv_path = config['dataset']['csv_path']
    args.fold_column = config['dataset']['fold_column']
    args.train_fold = config['dataset']['train_fold']
    args.val_fold = config['dataset']['val_fold']
    # Handle feature_cols - if null, we'll let the dataset auto-detect
    feature_cols = config['dataset']['feature_cols']
    if feature_cols is None:
        # For auto-detection, we'll pass an empty list and let the dataset handle it
        args.feature_cols = []
    else:
        args.feature_cols = feature_cols
    args.image_size = config['dataset']['image_size']
    
    # Model arguments
    args.resnet = config['model']['architecture']
    
    # Training arguments
    args.batch_size = config['training']['batch_size']
    args.epochs = config['training']['epochs']
    args.lr = config['training']['learning_rate']
    args.weight_decay = config['training']['weight_decay']
    args.optimizer = config['training']['optimizer']
    args.scheduler = config['training']['scheduler']
    args.eta_min = config['training']['eta_min']
    args.max_grad_norm = config['training']['max_grad_norm']
    args.amp = config['training']['amp']
    args.compile = config['training']['compile']
    
    # Output arguments
    args.log_dir = config['logging']['log_dir']
    args.output = config['output']['final_model_path']
    
    # Multi-GPU arguments
    args.world_size = config['distributed']['world_size']
    args.backend = config['distributed']['backend']
    args.num_workers = config['distributed']['num_workers']
    args.use_smart_cache = config['dataset']['use_smart_cache']
    args.cache_rate = config['dataset']['cache_rate']
    args.replace_rate = config['dataset']['replace_rate']
    args.pin_memory = config['dataloader']['pin_memory']
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Run pretraining experiment with structured output")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to YAML config file')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Override experiment name')
    parser.add_argument('--auto_name', action='store_true',
                       help='Generate automatic experiment name')
    
    args = parser.parse_args()
    
    # Generate automatic experiment name if requested
    if args.auto_name:
        config = load_config(args.config)
        model_name = config['model']['architecture']
        dataset_name = Path(config['dataset']['csv_path']).stem
        args.experiment_name = create_experiment_name('pretrain', model_name, dataset_name)
        print(f"🤖 Generated experiment name: {args.experiment_name}")
    
    run_pretraining_experiment(args.config, args.experiment_name)


if __name__ == '__main__':
    main()
