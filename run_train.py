"""
Entry point for training the VisuoMotor LSTM model (alias of `python -m lstm_robotics train`).

Synthetic demo data (no robot required):
    python run_train.py --demo

Real robot trajectory log (CSV; schema in lstm_robotics/logs.py, template in data/templates/):
    python run_train.py --data_csv /path/to/trajectory_log.csv --epochs 100

Episode directory or HDF5:
    python run_train.py --data_dir /path/to/episodes --epochs 100

With pretrained MobileNetV2 backbone:
    python run_train.py --data_csv logs.csv --backbone mobilenetv2 --epochs 50

Evaluate a trained checkpoint:
    python run_train.py --eval --checkpoint checkpoints/best.pt --data_csv logs.csv
"""
import sys

from lstm_robotics.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
