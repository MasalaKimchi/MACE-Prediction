"""
Experiment management utilities for organizing results, configs, and checkpoints.
Provides structured storage in the experiments folder with YAML configs and organized weights.
"""

import os
import yaml
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch


class ExperimentManager:
    """Manages experiment organization with structured output in experiments folder."""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Initialize experiment manager.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments (default: "experiments")
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        
        # Create experiment directory structure
        self._create_directory_structure()
        
        # Initialize metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'status': 'running'
        }
    
    def _create_directory_structure(self):
        """Create the standard experiment directory structure."""
        directories = [
            'configs',           # YAML configuration files
            'checkpoints',       # Model checkpoints
            'logs',             # Training logs and tensorboard
            'results',          # Final results and metrics
            'artifacts'         # Additional artifacts (plots, etc.)
        ]
        
        for dir_name in directories:
            (self.experiment_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], config_name: str = "config.yaml") -> str:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of config file
            
        Returns:
            Path to saved config file
        """
        config_path = self.experiment_dir / "configs" / config_name
        
        # Add experiment metadata to config
        config_with_metadata = {
            'experiment': {
                'name': self.experiment_name,
                'created_at': self.metadata['created_at'],
                'config_path': str(config_path)
            },
            **config
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_with_metadata, f, default_flow_style=False, indent=2)
        
        print(f"📝 Configuration saved to: {config_path}")
        return str(config_path)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float], 
                       checkpoint_name: str = None, is_best: bool = False) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Dictionary of metrics
            checkpoint_name: Name for checkpoint file
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Determine checkpoint name
        if checkpoint_name is None:
            if is_best:
                checkpoint_name = "checkpoint_best.pth"
            else:
                checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pth"
        
        checkpoint_path = self.experiment_dir / "checkpoints" / checkpoint_name
        
        # Get model state dict (handle DataParallel/DDP)
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        # Create comprehensive checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'experiment_metadata': self.metadata,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        status = "🏆 BEST" if is_best else f"Epoch {epoch}"
        print(f"💾 Checkpoint saved: {checkpoint_path} ({status})")
        return str(checkpoint_path)
    
    def save_final_model(self, model: torch.nn.Module, metrics: Dict[str, float],
                        model_name: str = "final_model.pth") -> str:
        """
        Save final trained model.
        
        Args:
            model: Final model to save
            metrics: Final metrics
            model_name: Name for final model file
            
        Returns:
            Path to saved model
        """
        model_path = self.experiment_dir / "checkpoints" / model_name
        
        # Get model state dict (handle DataParallel/DDP)
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        # Create final model checkpoint
        final_model = {
            'model_state_dict': model_state_dict,
            'final_metrics': metrics,
            'experiment_metadata': self.metadata,
            'timestamp': datetime.now().isoformat(),
            'is_final': True
        }
        
        torch.save(final_model, model_path)
        print(f"🎯 Final model saved: {model_path}")
        return str(model_path)
    
    def save_results(self, results: Dict[str, Any], results_name: str = "results.json") -> str:
        """
        Save experiment results.
        
        Args:
            results: Results dictionary
            results_name: Name for results file
            
        Returns:
            Path to saved results
        """
        results_path = self.experiment_dir / "results" / results_name
        
        # Add experiment metadata
        results_with_metadata = {
            'experiment_metadata': self.metadata,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"📊 Results saved: {results_path}")
        return str(results_path)
    
    def get_log_dir(self) -> str:
        """Get path to logs directory."""
        return str(self.experiment_dir / "logs")
    
    def get_checkpoint_dir(self) -> str:
        """Get path to checkpoints directory."""
        return str(self.experiment_dir / "checkpoints")
    
    def get_results_dir(self) -> str:
        """Get path to results directory."""
        return str(self.experiment_dir / "results")
    
    def get_config_dir(self) -> str:
        """Get path to configs directory."""
        return str(self.experiment_dir / "configs")
    
    def get_artifacts_dir(self) -> str:
        """Get path to artifacts directory."""
        return str(self.experiment_dir / "artifacts")
    
    def update_metadata(self, **kwargs):
        """Update experiment metadata."""
        self.metadata.update(kwargs)
    
    def mark_completed(self, final_metrics: Dict[str, float] = None):
        """Mark experiment as completed."""
        self.metadata['status'] = 'completed'
        self.metadata['completed_at'] = datetime.now().isoformat()
        if final_metrics:
            self.metadata['final_metrics'] = final_metrics
    
    def mark_failed(self, error_message: str = None):
        """Mark experiment as failed."""
        self.metadata['status'] = 'failed'
        self.metadata['failed_at'] = datetime.now().isoformat()
        if error_message:
            self.metadata['error_message'] = error_message
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            'experiment_dir': str(self.experiment_dir),
            'metadata': self.metadata,
            'structure': {
                'configs': list((self.experiment_dir / "configs").glob("*.yaml")),
                'checkpoints': list((self.experiment_dir / "checkpoints").glob("*.pth")),
                'logs': list((self.experiment_dir / "logs").glob("*")),
                'results': list((self.experiment_dir / "results").glob("*.json")),
                'artifacts': list((self.experiment_dir / "artifacts").glob("*"))
            }
        }


def create_experiment_name(experiment_type: str, model_name: str, 
                          dataset_name: str = None, timestamp: bool = True) -> str:
    """
    Create a standardized experiment name.
    
    Args:
        experiment_type: Type of experiment (e.g., 'pretrain', 'finetune')
        model_name: Name of model (e.g., 'resnet50')
        dataset_name: Name of dataset (optional)
        timestamp: Whether to include timestamp
        
    Returns:
        Standardized experiment name
    """
    name_parts = [experiment_type, model_name]
    
    if dataset_name:
        name_parts.append(dataset_name)
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts.append(timestamp_str)
    
    return "_".join(name_parts)


def load_experiment_config(experiment_path: str) -> Dict[str, Any]:
    """
    Load configuration from an experiment.
    
    Args:
        experiment_path: Path to experiment directory
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(experiment_path) / "configs" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def list_experiments(base_dir: str = "experiments") -> list:
    """
    List all experiments in the base directory.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        List of experiment directories
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return []
    
    experiments = []
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            config_path = exp_dir / "configs" / "config.yaml"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    experiments.append({
                        'name': exp_dir.name,
                        'path': str(exp_dir),
                        'config': config.get('experiment', {}),
                        'created_at': config.get('experiment', {}).get('created_at', 'unknown')
                    })
                except Exception as e:
                    print(f"Warning: Could not load config for {exp_dir.name}: {e}")
    
    # Sort by creation time (newest first)
    experiments.sort(key=lambda x: x['created_at'], reverse=True)
    return experiments


def cleanup_experiment(experiment_path: str, keep_checkpoints: bool = True):
    """
    Clean up experiment directory (remove logs, keep checkpoints).
    
    Args:
        experiment_path: Path to experiment directory
        keep_checkpoints: Whether to keep checkpoint files
    """
    exp_path = Path(experiment_path)
    
    if not exp_path.exists():
        print(f"Experiment directory not found: {experiment_path}")
        return
    
    # Remove logs directory
    logs_dir = exp_path / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
        print(f"Removed logs directory: {logs_dir}")
    
    # Optionally remove checkpoints
    if not keep_checkpoints:
        checkpoints_dir = exp_path / "checkpoints"
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
            print(f"Removed checkpoints directory: {checkpoints_dir}")
    
    print(f"Cleanup completed for experiment: {experiment_path}")
