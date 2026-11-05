"""
Dry-run reference implementation of the robot bridge protocol (see lstm_robotics/robot.py).

It accepts the JSON-lines commands a real bridge must handle, records them, and moves
nothing. Use it to test the policy loop end to end, and as the skeleton for the real bridge
that calls your vendor SDK / ROS 2 node.

    python -m lstm_robotics.bridge --port 5555
"""
import argparse
import json
import socketserver
import threading
from typing import List


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        for raw in self.rfile:
            try:
                msg = json.loads(raw)
                cmd = msg.get("cmd")
                if cmd == "hello":
                    reply = {"ok": True, "name": server.robot_name}
                elif cmd == "action":
                    delta = msg["delta"]
                    if len(delta) != 6:
                        raise ValueError("delta needs 6 values")
                    grip = float(msg["gripper"])
                    server.received.append([float(v) for v in delta] + [grip])
                    if server.verbose:
                        print(f"action delta={delta} gripper={grip:.2f}")
                    reply = {"ok": True}
                elif cmd == "stop":
                    server.stopped += 1
                    reply = {"ok": True}
                else:
                    reply = {"ok": False, "error": f"unknown cmd {cmd!r}"}
            except (ValueError, KeyError, TypeError) as exc:
                reply = {"ok": False, "error": str(exc)}
            self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))


class DryRunBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str = "127.0.0.1", port: int = 0, verbose: bool = False,
                 robot_name: str = "dry-run-bridge"):
        super().__init__((host, port), _Handler)
        self.received: List[List[float]] = []
        self.stopped = 0
        self.verbose = verbose
        self.robot_name = robot_name

    @property
    def port(self) -> int:
        return self.server_address[1]

    def start_background(self) -> "DryRunBridge":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dry-run robot bridge (moves nothing)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()
    srv = DryRunBridge(args.host, args.port, verbose=True)
    print(f"dry-run bridge listening on {args.host}:{srv.port} (Ctrl-C to quit)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
