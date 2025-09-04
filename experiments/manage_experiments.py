#!/usr/bin/env python3
"""
Experiment management script for listing, viewing, and cleaning up experiments.
"""

import argparse
import yaml
from pathlib import Path
from experiments.experiment_utils import list_experiments, load_experiment_config, cleanup_experiment


def list_all_experiments():
    """List all experiments with their status and metadata."""
    experiments = list_experiments()
    
    if not experiments:
        print("📭 No experiments found in experiments/ directory")
        return
    
    print(f"📋 Found {len(experiments)} experiments:")
    print("=" * 80)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i:2d}. {exp['name']}")
        print(f"    📁 Path: {exp['path']}")
        print(f"    📅 Created: {exp['created_at']}")
        print(f"    📝 Type: {exp['config'].get('experiment', {}).get('type', 'unknown')}")
        print(f"    🏗️  Model: {exp['config'].get('model', {}).get('architecture', 'unknown')}")
        print()


def show_experiment_details(experiment_name: str):
    """Show detailed information about a specific experiment."""
    experiments = list_experiments()
    exp = next((e for e in experiments if e['name'] == experiment_name), None)
    
    if not exp:
        print(f"❌ Experiment '{experiment_name}' not found")
        return
    
    print(f"🔍 Experiment Details: {exp['name']}")
    print("=" * 50)
    print(f"📁 Path: {exp['path']}")
    print(f"📅 Created: {exp['created_at']}")
    print()
    
    # Show config
    print("📝 Configuration:")
    print("-" * 20)
    config = exp['config']
    
    # Experiment info
    exp_info = config.get('experiment', {})
    print(f"  Type: {exp_info.get('type', 'unknown')}")
    print(f"  Description: {exp_info.get('description', 'N/A')}")
    
    # Model info
    model_info = config.get('model', {})
    print(f"  Model: {model_info.get('architecture', 'unknown')}")
    print(f"  Init Mode: {model_info.get('init_mode', 'unknown')}")
    if model_info.get('pretrained_path'):
        print(f"  Pretrained: {model_info['pretrained_path']}")
    
    # Training info
    training_info = config.get('training', {})
    print(f"  Epochs: {training_info.get('epochs', 'unknown')}")
    print(f"  Batch Size: {training_info.get('batch_size', 'unknown')}")
    print(f"  Learning Rate: {training_info.get('learning_rate', 'unknown')}")
    
    # Dataset info
    dataset_info = config.get('dataset', {})
    print(f"  Dataset: {dataset_info.get('csv_path', 'unknown')}")
    if dataset_info.get('feature_cols'):
        print(f"  Features: {', '.join(dataset_info['feature_cols'])}")
    
    print()
    
    # Show directory structure
    exp_path = Path(exp['path'])
    print("📂 Directory Structure:")
    print("-" * 20)
    
    for subdir in ['configs', 'checkpoints', 'logs', 'results', 'artifacts']:
        subdir_path = exp_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.iterdir())
            print(f"  {subdir}/: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"    - {file.name}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        else:
            print(f"  {subdir}/: (empty)")


def compare_experiments(exp1_name: str, exp2_name: str):
    """Compare two experiments."""
    experiments = list_experiments()
    exp1 = next((e for e in experiments if e['name'] == exp1_name), None)
    exp2 = next((e for e in experiments if e['name'] == exp2_name), None)
    
    if not exp1:
        print(f"❌ Experiment '{exp1_name}' not found")
        return
    if not exp2:
        print(f"❌ Experiment '{exp2_name}' not found")
        return
    
    print(f"🔄 Comparing Experiments:")
    print(f"  1. {exp1_name}")
    print(f"  2. {exp2_name}")
    print("=" * 60)
    
    # Compare key parameters
    config1 = exp1['config']
    config2 = exp2['config']
    
    comparisons = [
        ('Model', 'model.architecture'),
        ('Epochs', 'training.epochs'),
        ('Batch Size', 'training.batch_size'),
        ('Learning Rate', 'training.learning_rate'),
        ('Optimizer', 'training.optimizer'),
        ('Dataset', 'dataset.csv_path'),
    ]
    
    for name, key_path in comparisons:
        keys = key_path.split('.')
        val1 = config1
        val2 = config2
        
        try:
            for key in keys:
                val1 = val1[key]
                val2 = val2[key]
            
            if val1 == val2:
                print(f"  {name}: {val1} (same)")
            else:
                print(f"  {name}: {val1} vs {val2} (different)")
        except KeyError:
            print(f"  {name}: N/A vs N/A")


def cleanup_experiments(experiment_names: list, keep_checkpoints: bool = True):
    """Clean up specified experiments."""
    for exp_name in experiment_names:
        experiments = list_experiments()
        exp = next((e for e in experiments if e['name'] == exp_name), None)
        
        if not exp:
            print(f"❌ Experiment '{exp_name}' not found")
            continue
        
        print(f"🧹 Cleaning up experiment: {exp_name}")
        cleanup_experiment(exp['path'], keep_checkpoints=keep_checkpoints)


def main():
    parser = argparse.ArgumentParser(description="Manage experiments")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('experiment_name', help='Name of experiment to show')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two experiments')
    compare_parser.add_argument('exp1', help='First experiment name')
    compare_parser.add_argument('exp2', help='Second experiment name')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up experiments')
    cleanup_parser.add_argument('experiments', nargs='+', help='Experiment names to cleanup')
    cleanup_parser.add_argument('--remove-checkpoints', action='store_true',
                               help='Also remove checkpoint files')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_all_experiments()
    elif args.command == 'show':
        show_experiment_details(args.experiment_name)
    elif args.command == 'compare':
        compare_experiments(args.exp1, args.exp2)
    elif args.command == 'cleanup':
        cleanup_experiments(args.experiments, keep_checkpoints=not args.remove_checkpoints)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
