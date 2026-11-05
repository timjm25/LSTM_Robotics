"""
Visualise predicted vs actual action trajectories from a saved checkpoint.

Usage:
    python scripts/visualise_predictions.py --checkpoint checkpoints/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lstm_robotics.evaluate import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/demo")
    parser.add_argument("--data_csv", default="", help="trajectory log CSV")
    parser.add_argument("--demo", action="store_true", help="synthetic data if no real data")
    parser.add_argument("--output_dir", default="eval_output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        data_csv=args.data_csv,
        use_synthetic=args.demo,
        output_dir=args.output_dir,
        device_str=args.device,
    )
