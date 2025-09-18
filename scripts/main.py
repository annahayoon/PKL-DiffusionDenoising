#!/usr/bin/env python3
"""
Main Entry Point for PKL Diffusion Denoising Scripts

This script provides a unified command-line interface for all PKL diffusion
denoising operations including training, evaluation, and preprocessing.

Usage:
    python scripts/main.py <command> [args]

Commands:
    train       - Train diffusion models
    evaluate    - Evaluate model performance (includes inference + metrics)
    preprocess  - Preprocess data
    baseline    - Run baseline methods
    util        - Utility operations
    sid         - SID dataset operations (download, evaluate)

Examples:
    python scripts/main.py train --config configs/training/microscopy.yaml
    python scripts/main.py evaluate --checkpoint checkpoints/best_model.pt --real-dir data/real/
    python scripts/main.py baseline --input data/test/ --output baseline_results/
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
try:
    from pkl_dg.utils import (
        load_config, 
        print_config_summary,
        validate_and_complete_config,
        setup_logging
    )
    from pkl_dg.utils.visualization import create_comparison_previews
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("⚠️ PKL-DG utils not available, running in basic mode")


class ScriptRunner:
    """Utility class for running scripts with proper error handling."""
    
    def __init__(self, scripts_dir: Path):
        self.scripts_dir = scripts_dir
        
    def run_script(
        self, 
        script_path: Path, 
        args: List[str], 
        check_exists: bool = True
    ) -> int:
        """Run a script with given arguments.
        
        Args:
            script_path: Path to the script
            args: Arguments to pass to the script
            check_exists: Whether to check if script exists
            
        Returns:
            Exit code from the script
        """
        if check_exists and not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return 1
        
        # Build command
        cmd = [sys.executable, str(script_path)] + args
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        try:
            # Run script
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Error running script: {e}")
            return 1
    
    def list_scripts(self, subdir: str) -> List[Path]:
        """List available scripts in a subdirectory."""
        script_dir = self.scripts_dir / subdir
        if not script_dir.exists():
            return []
        
        scripts = []
        for script_path in script_dir.glob("*.py"):
            if script_path.name != "__init__.py":
                scripts.append(script_path)
        
        return sorted(scripts)


def create_training_parser(subparsers):
    """Create training command parser."""
    parser = subparsers.add_parser(
        'train', 
        help='Train diffusion models',
        description='Train diffusion models on various datasets'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        choices=['microscopy', 'imagenet', 'mnist'],
        default='microscopy',
        help='Dataset to train on'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--progressive',
        action='store_true',
        help='Enable progressive training'
    )
    
    parser.add_argument(
        '--gpu', '-g',
        type=int,
        help='GPU ID to use'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser




def create_evaluation_parser(subparsers):
    """Create evaluation command parser."""
    parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate model performance',
        description='Evaluate models using various metrics'
    )
    
    parser.add_argument(
        '--real-dir',
        type=str,
        help='Directory containing real images'
    )
    
    parser.add_argument(
        '--fake-dir',
        type=str,
        help='Directory containing generated images'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Model checkpoint to evaluate'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['fid', 'is', 'psnr', 'ssim', 'lpips'],
        default=['fid', 'is'],
        help='Metrics to compute'
    )
    
    parser.add_argument(
        '--baseline',
        choices=['rl', 'rcan', 'all'],
        help='Run baseline comparison'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )
    
    return parser


def create_preprocess_parser(subparsers):
    """Create preprocessing command parser."""
    parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess data',
        description='Preprocess and prepare data for training'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input data directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--task',
        choices=['prepare', 'process', 'visualize'],
        default='prepare',
        help='Preprocessing task'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='Target image size'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize images'
    )
    
    return parser


def create_baseline_parser(subparsers):
    """Create baseline command parser."""
    parser = subparsers.add_parser(
        'baseline',
        help='Run baseline methods',
        description='Run baseline methods for comparison'
    )
    
    parser.add_argument(
        '--method',
        choices=['rl', 'richardson-lucy'],
        default='rl',
        help='Baseline method to run'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory or file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='baseline_outputs/',
        help='Output directory'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup baseline method'
    )
    
    return parser


def create_util_parser(subparsers):
    """Create utility command parser."""
    parser = subparsers.add_parser(
        'util',
        help='Utility operations',
        description='Various utility operations'
    )
    
    parser.add_argument(
        '--task',
        choices=['preview', 'interpolate', 'config'],
        required=True,
        help='Utility task to run'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input file or directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file or directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file'
    )
    
    return parser


def create_sid_parser(subparsers):
    """Create SID (See-in-the-Dark) command parser."""
    parser = subparsers.add_parser(
        'sid',
        help='SID dataset operations',
        description='Download, evaluate, and manage SID (See-in-the-Dark) dataset for cross-domain generalization'
    )
    
    parser.add_argument(
        '--task',
        choices=['download', 'evaluate', 'check'],
        required=True,
        help='SID task to perform'
    )
    
    # Common arguments
    parser.add_argument('--data-dir', default='data/SID', help='SID dataset directory')
    parser.add_argument('--camera', choices=['Sony', 'Fuji'], default='Sony', help='Camera type')
    
    # Evaluation-specific arguments
    parser.add_argument('--checkpoint', help='Model checkpoint for evaluation')
    parser.add_argument('--guidance-types', nargs='+', choices=['pkl', 'l2', 'anscombe'],
                       default=['pkl'], help='Guidance strategies for evaluation')
    parser.add_argument('--max-images', type=int, help='Maximum images to evaluate')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    # Download-specific arguments
    parser.add_argument('--force', action='store_true', help='Force re-download')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser


def handle_training_command(args, runner: ScriptRunner) -> int:
    """Handle training command."""
    # Route to appropriate run* script based on dataset
    if args.dataset in ['microscopy', 'real']:
        script_name = "run_microscopy.py"
        script_args = ['--mode', 'train']
        
        if args.config:
            script_args.extend(['--config', args.config])
    else:
        # Natural images (imagenet, mnist, cifar, etc.)
        script_name = "run_natural.py"
        script_args = ['--mode', 'train', '--dataset', args.dataset]
        
        if args.config:
            script_args.extend(['--config', args.config])
    
    script_path = runner.scripts_dir / script_name
    
    # Add common arguments
    if args.resume:
        script_args.extend(['--resume', args.resume])
    
    if args.progressive:
        script_args.append('--progressive')
    
    if args.gpu is not None:
        script_args.extend(['--gpu', str(args.gpu)])
    
    if args.debug:
        script_args.append('--debug')
    
    return runner.run_script(script_path, script_args)




def handle_evaluation_command(args, runner: ScriptRunner) -> int:
    """Handle evaluation command."""
    if args.baseline:
        # Run baseline comparison
        if args.baseline == 'all':
            script_name = "compare_all_methods.py"
        else:
            script_name = "run_baseline_comparison.py"
        
        script_path = runner.scripts_dir / "evaluation" / script_name
        script_args = []
        
        if args.real_dir:
            script_args.extend(['--real-dir', args.real_dir])
        if args.fake_dir:
            script_args.extend(['--fake-dir', args.fake_dir])
        
    else:
        # Regular evaluation
        script_path = runner.scripts_dir / "evaluation" / "evaluate.py"
        script_args = []
        
        if args.checkpoint:
            script_args.extend(['--checkpoint', args.checkpoint])
        if args.real_dir:
            script_args.extend(['--real-dir', args.real_dir])
        if args.fake_dir:
            script_args.extend(['--fake-dir', args.fake_dir])
        if args.metrics:
            script_args.extend(['--metrics'] + args.metrics)
        if args.output:
            script_args.extend(['--output', args.output])
    
    return runner.run_script(script_path, script_args)


def handle_preprocess_command(args, runner: ScriptRunner) -> int:
    """Handle preprocessing command."""
    # Choose preprocessing script based on task
    script_map = {
        'prepare': 'prepare_images.py',
        'process': 'process_microscopy_data.py',
        'visualize': 'visualize_real_data.py'
    }
    
    script_name = script_map.get(args.task, 'preprocess_all.py')
    script_path = runner.scripts_dir / "preprocessing" / script_name
    
    # Build arguments
    script_args = ['--input-dir', args.input_dir, '--output-dir', args.output_dir]
    
    if args.image_size != 256:
        script_args.extend(['--image-size', str(args.image_size)])
    
    if args.normalize:
        script_args.append('--normalize')
    
    return runner.run_script(script_path, script_args)


def handle_baseline_command(args, runner: ScriptRunner) -> int:
    """Handle baseline command."""
    # Use the comprehensive Richardson-Lucy baseline from pkl_dg.baselines
    import subprocess
    import sys
    
    # Build arguments for the comprehensive script
    script_args = [
        sys.executable, '-m', 'pkl_dg.baselines.richardson_lucy',
        '--input-dir', args.input,
        '--output-dir', args.output
    ]
    
    # Add GT directory if available
    if hasattr(args, 'gt_dir') and args.gt_dir:
        script_args.extend(['--gt-dir', args.gt_dir])
    
    try:
        result = subprocess.run(script_args, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Baseline command failed: {e}")
        return e.returncode


def handle_util_command(args, runner: ScriptRunner) -> int:
    """Handle utility command."""
    if args.task == 'preview':
        print("🔧 Creating comparison previews")
        if not UTILS_AVAILABLE:
            print("❌ PKL-DG utils required for preview generation")
            return 1
        
        if not args.input:
            print("❌ Input directory required for preview generation")
            print("Usage: --input <wf_dir>,<pred_dir>,<gt_dir> --output <out_dir>")
            return 1
        
        # Parse input directories (expecting comma-separated: wf_dir,pred_dir,gt_dir)
        input_dirs = args.input.split(',')
        if len(input_dirs) != 3:
            print("❌ Input must be three comma-separated directories: wf_dir,pred_dir,gt_dir")
            return 1
        
        wf_dir, pred_dir, gt_dir = input_dirs
        out_dir = args.output if args.output else 'preview_outputs'
        
        try:
            count = create_comparison_previews(
                wf_dir=wf_dir,
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                out_dir=out_dir,
                max_n=24  # Default from original script
            )
            if count > 0:
                print(f"✅ Successfully generated {count} preview images")
                return 0
            else:
                print("⚠️ No preview images generated")
                return 1
        except Exception as e:
            print(f"❌ Error generating previews: {e}")
            return 1
    
    elif args.task == 'interpolate':
        print("🔧 Interpolation utility - creating interpolation sequence")
        if not UTILS_AVAILABLE:
            print("❌ PKL-DG utils required for interpolation")
            return 1
        
        # TODO: Implement interpolation utility script
        print("⚠️ Interpolation utility not yet implemented")
        return 1
    
    elif args.task == 'config':
        print("🔧 Configuration utility")
        if args.config and UTILS_AVAILABLE:
            try:
                config = load_config(args.config)
                print_config_summary(config, "Configuration Summary")
                return 0
            except Exception as e:
                print(f"❌ Error loading config: {e}")
                return 1
        else:
            print("❌ Config file required and utils must be available")
            return 1
    
    else:
        print(f"❌ Unknown utility task: {args.task}")
        return 1


def handle_sid_command(args, runner: ScriptRunner) -> int:
    """Handle SID (See-in-the-Dark) dataset command."""
    if args.task == 'download':
        print("📥 Downloading SID dataset")
        # Use the dataset module to download SID data
        try:
            from pkl_dg.data import download_sid_dataset
            data_dir = args.data_dir or 'data/SID'
            camera = args.camera or 'Sony'
            
            success = download_sid_dataset(data_dir, camera, force=args.force)
            if success:
                print(f"✅ {camera} SID dataset downloaded successfully to {data_dir}")
                return 0
            else:
                print(f"❌ Failed to download {camera} SID dataset")
                return 1
        except ImportError as e:
            print(f"❌ Error importing download function: {e}")
            return 1
    
    elif args.task == 'evaluate':
        print("🔬 Evaluating on SID dataset for cross-domain generalization")
        
        # Required arguments
        if not args.checkpoint:
            print("❌ Checkpoint required for evaluation")
            return 1
        
        # Use the unified evaluation system as a Python module
        cmd = [sys.executable, '-m', 'pkl_dg.evaluation']
        cmd.extend([
            '--config-name', 'evaluation/sid_evaluation',
            f'model.checkpoint_path={args.checkpoint}',
            f'processing.sid_camera_type={args.camera}',
            f'processing.sid_data_dir={args.data_dir}'
        ])
        
        # Optional arguments
        if args.guidance_types:
            guidance_str = '[' + ','.join(args.guidance_types) + ']'
            cmd.append(f'processing.sid_guidance_types={guidance_str}')
        if args.max_images:
            cmd.append(f'processing.max_images={args.max_images}')
        if args.output_dir:
            cmd.append(f'inference.output_dir={args.output_dir}')
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        try:
            import subprocess
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except Exception as e:
            print(f"❌ Error running evaluation: {e}")
            return 1
    
    elif args.task == 'check':
        print("✅ Checking SID dataset")
        # Use the dataset module to check SID data
        try:
            from pkl_dg.data import SIDDataset
            data_dir = args.data_dir or 'data/SID'
            camera = args.camera or 'Sony'
            
            try:
                # Try to create dataset to check if data exists
                dataset = SIDDataset(data_dir=data_dir, camera_type=camera)
                print(f"✅ {camera} SID dataset found in {data_dir}")
                print(f"   Found {len(dataset)} image pairs")
                return 0
            except (FileNotFoundError, RuntimeError, Exception) as e:
                print(f"❌ {camera} SID dataset not found in {data_dir}")
                print(f"   Error: {e}")
                print(f"💡 To download: python scripts/main.py sid --task download --camera {camera}")
                return 1
        except ImportError as e:
            print(f"❌ Error importing SID dataset: {e}")
            return 1
    
    else:
        print(f"❌ Unknown SID task: {args.task}")
        print("Available tasks: download, evaluate, check")
        return 1


def list_available_scripts(runner: ScriptRunner):
    """List all available scripts."""
    print("📋 Available Scripts:")
    print("=" * 50)
    
    categories = [
        ('training', 'Training Scripts'),
        ('evaluation', 'Evaluation Scripts'),
        ('preprocessing', 'Preprocessing Scripts'),
        ('baselines', 'Baseline Scripts'),
        ('utilities', 'Utility Scripts')
    ]
    
    for category, title in categories:
        scripts = runner.list_scripts(category)
        if scripts:
            print(f"\n{title}:")
            for script in scripts:
                print(f"  • {script.name}")
        else:
            print(f"\n{title}: No scripts found")


def main():
    """Main entry point."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='PKL Diffusion Denoising - Unified Script Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--list-scripts',
        action='store_true',
        help='List all available scripts'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='PKL Diffusion Denoising v1.0.0'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Add command parsers
    create_training_parser(subparsers)
    create_evaluation_parser(subparsers)
    create_preprocess_parser(subparsers)
    create_baseline_parser(subparsers)
    create_util_parser(subparsers)
    create_sid_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize script runner
    scripts_dir = Path(__file__).parent
    runner = ScriptRunner(scripts_dir)
    
    # Handle list scripts
    if args.list_scripts:
        list_available_scripts(runner)
        return 0
    
    # Require command
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging if available
    if UTILS_AVAILABLE:
        try:
            setup_logging("logs", log_level="INFO")
        except Exception:
            pass  # Continue without logging
    
    # Route to appropriate handler
    try:
        if args.command == 'train':
            return handle_training_command(args, runner)
        elif args.command == 'evaluate':
            return handle_evaluation_command(args, runner)
        elif args.command == 'preprocess':
            return handle_preprocess_command(args, runner)
        elif args.command == 'baseline':
            return handle_baseline_command(args, runner)
        elif args.command == 'util':
            return handle_util_command(args, runner)
        elif args.command == 'sid':
            return handle_sid_command(args, runner)
        else:
            print(f"❌ Unknown command: {args.command}")
            print("Available commands: train, evaluate, preprocess, baseline, util, sid")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug if hasattr(args, 'debug') else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
