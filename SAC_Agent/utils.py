from pathlib import Path
import time
import sys
from termcolor import colored
import cv2 as cv
import atexit


class Tee:
    """Duplicate stdout/stderr to also write into a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        primary = self.streams[0]
        return primary.isatty() if hasattr(primary, "isatty") else False


def setup_run_logging():
    """Mirror console output into a timestamped log file under logs/."""

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{timestamp}.log"
    log_handle = open(log_path, "a", buffering=1, encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    atexit.register(log_handle.close)
    print(
        colored(
            f"Logging output to {log_path}",
            color="green",
            attrs=["bold"],
        )
    )
    return log_path


def save_scene_snapshot(controller, run_tag, cameras=None, width=1280, height=960):
    """Capture and save snapshots of the current scene from one or more cameras."""

    snapshots_dir = Path("renders")
    snapshots_dir.mkdir(exist_ok=True)

    if cameras is None:
        cameras = ["main1"]
    elif isinstance(cameras, str):
        cameras = [cameras]

    saved_paths = []
    controller.sim.forward()
    for camera_name in cameras:
        rgb, _ = controller.get_image_data(
            width=width, height=height, camera=camera_name
        )
        snapshot_path = snapshots_dir / f"{run_tag}_{camera_name}_scene.png"
        success = cv.imwrite(str(snapshot_path), cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
        if success:
            saved_paths.append(snapshot_path)
            print(
                colored(
                    f"Saved scene snapshot to {snapshot_path}",
                    color="green",
                    attrs=["bold"],
                )
            )
        else:
            print(
                colored(
                    f"Failed to write snapshot for camera '{camera_name}'",
                    color="red",
                    attrs=["bold"],
                )
            )
    return saved_paths
