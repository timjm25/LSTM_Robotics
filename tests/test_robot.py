"""Tests for the robot interface adapters and the closed control loop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from lstm_robotics.bridge import DryRunBridge
from lstm_robotics.robot import (
    ActionLimiter,
    RobotError,
    RobotInterface,
    RobotUnavailableError,
    SimulatedRobot,
    TCPBridgeRobot,
    max_delta_from_env,
    robot_from_env,
    run_policy,
)


class ConstPredictor:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step(self, frame):
        assert frame.ndim == 3 and frame.shape[2] == 3
        return self.action


class FailingPredictor(ConstPredictor):
    def step(self, frame):
        raise RuntimeError("model blew up")


def _frame():
    return np.zeros((16, 16, 3), dtype=np.uint8)


@pytest.fixture
def bridge():
    srv = DryRunBridge().start_background()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_sim_is_default_and_satisfies_interface():
    r = robot_from_env({})
    assert isinstance(r, SimulatedRobot) and isinstance(r, RobotInterface)
    r.connect()
    assert r.get_frame().shape == (224, 224, 3)


def test_limiter_clips_and_counts():
    lim = ActionLimiter([0.01] * 3 + [0.1] * 3)
    d, g = lim(np.array([0.5, -0.5, 0.001, 1.0, 0.0, -1.0]), 1.7)
    np.testing.assert_allclose(d, [0.01, -0.01, 0.001, 0.1, 0.0, -0.1])
    assert g == 1.0 and lim.n_clipped == 1
    with pytest.raises(ValueError, match="non-finite"):
        lim(np.array([np.nan] * 6), 0.0)


def test_max_delta_env_parsing():
    assert max_delta_from_env({}).tolist() == [0.01] * 6
    assert max_delta_from_env({"LSTM_ROBOT_MAX_DELTA": "0.02"}).tolist() == [0.02] * 6
    with pytest.raises(ValueError, match="one value or six"):
        max_delta_from_env({"LSTM_ROBOT_MAX_DELTA": "1,2"})


def test_run_policy_on_sim_denormalises_and_limits():
    robot = SimulatedRobot(image_size=32)
    pred = ConstPredictor([1.0, -1.0, 0.5, 0, 0, 0, 0.9])
    sent = run_policy(pred, robot, steps=4, action_scale=[0.002, 0.05, 0.004, 1, 1, 1],
                      limiter=ActionLimiter(0.01))
    np.testing.assert_allclose(sent[0], [0.002, -0.01, 0.002, 0, 0, 0, 0.9])
    assert len(robot.sent) == 4 and pred.resets == 1
    assert robot.pose[0] == pytest.approx(0.008)


def test_tcp_requires_env_vars():
    with pytest.raises(RobotUnavailableError, match="LSTM_ROBOT_HOST"):
        robot_from_env({"LSTM_ROBOT_DRIVER": "tcp"})
    with pytest.raises(ValueError, match="sim' or 'tcp"):
        robot_from_env({"LSTM_ROBOT_DRIVER": "ros"})


def test_tcp_unreachable_bridge_fails_gracefully():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens on this port now
    robot = robot_from_env({"LSTM_ROBOT_DRIVER": "tcp", "LSTM_ROBOT_HOST": "127.0.0.1",
                            "LSTM_ROBOT_PORT": str(port)}, frame_source=_frame)
    with pytest.raises(RobotUnavailableError, match="cannot reach robot bridge"):
        robot.connect()


def test_tcp_bridge_round_trip(bridge):
    robot = robot_from_env({"LSTM_ROBOT_DRIVER": "tcp", "LSTM_ROBOT_HOST": "127.0.0.1",
                            "LSTM_ROBOT_PORT": str(bridge.port)}, frame_source=_frame)
    assert isinstance(robot, TCPBridgeRobot)
    sent = run_policy(ConstPredictor([0.5] * 6 + [1.0]), robot, steps=3,
                      action_scale=[0.01] * 6)
    assert len(bridge.received) == 3
    np.testing.assert_allclose(bridge.received[0], [0.005] * 6 + [1.0])
    np.testing.assert_allclose(sent, np.array(bridge.received))


def test_bridge_rejection_raises(bridge):
    robot = TCPBridgeRobot("127.0.0.1", bridge.port, _frame)
    robot.connect()
    assert robot.name == "dry-run-bridge"
    with pytest.raises(RobotError, match="rejected"):
        robot._request({"cmd": "dance"})
    robot.close()


def test_error_mid_run_sends_stop(bridge):
    robot = TCPBridgeRobot("127.0.0.1", bridge.port, _frame)
    with pytest.raises(RuntimeError, match="blew up"):
        run_policy(FailingPredictor([0] * 7), robot, steps=3, action_scale=[1] * 6)
    assert bridge.stopped == 1


def test_bad_camera_frame_rejected(bridge):
    robot = TCPBridgeRobot("127.0.0.1", bridge.port, lambda: np.zeros((4, 4)))
    robot.connect()
    with pytest.raises(RobotError, match="H, W, 3"):
        robot.get_frame()
    robot.close()
