"""Finish the requested acquisition without changing Windows script policy."""
import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--download-pid", type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
handle = kernel.OpenProcess(0x00100000, False, args.download_pid)
try:
    metadata = subprocess.run([sys.executable, str(root / "scripts/fetch_lasco_annotations.py")], cwd=root)
    print(f"metadata exit={metadata.returncode}; waiting for archive process {args.download_pid}", flush=True)
    if handle:
        kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
finally:
    if handle:
        kernel.CloseHandle(handle)
result = subprocess.run([sys.executable, str(root / "scripts/validate_additional_data.py")], cwd=root)
print(f"full coverage gate exit={result.returncode}; see additional_data_coverage.json", flush=True)
raise SystemExit(result.returncode)
