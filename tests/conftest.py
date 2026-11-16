"""Every test that takes `device` runs on the CPU and on the GPU, if there is one."""
import torch


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        metafunc.parametrize("device", available_devices())


def pytest_report_header(config):
    return f"fastlinkage test devices: {', '.join(available_devices())}"
