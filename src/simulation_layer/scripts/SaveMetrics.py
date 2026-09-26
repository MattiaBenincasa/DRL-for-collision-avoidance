#!/usr/bin/env python3
"""SaveMetrics: stores DDQN metrics in memory and writes them to CSV at the end of each episode.

This version creates a timestamped CSV filename of the form
  ddqn_metrics_episodes_YYYYmmdd_HHMMSS.csv
and writes configuration / environment metadata as commented JSON at the top
of the CSV file (lines starting with '#').
"""

import os
import csv
import threading
from typing import List, Dict, Optional, Any
import datetime
import json
import platform
import socket
import getpass
import sys
import subprocess


class SaveMetrics:
    """Container for DDQN metrics.

    Example usage:
        sm = SaveMetrics(csv_path='metrics/', config={'lr':0.001})
        sm.add(step=1, reward=0.1, loss=0.02)
        sm.write_episode(episode=0)
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        fieldnames: Optional[List[str]] = None,
        append: bool = True,
        decimals: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Keep the originally requested path; the final timestamped filename
        # will be generated on first write so the timestamp reflects creation time.
        self._requested_csv_path = csv_path
        self.csv_path: Optional[str] = None
        self.append = append
        self.decimals = int(decimals)
        self._memory: List[Dict[str, Any]] = []
        self._fieldnames: Optional[List[str]] = list(fieldnames) if fieldnames else None
        self._lock = threading.Lock()
        self._created_at: Optional[str] = None
        self.config: Dict[str, Any] = dict(config) if config else {}
        self._metadata_written = False

    def _round_value(self, v: Any) -> Any:
        """Round numeric values to `self.decimals` decimals (best-effort)."""
        if v is None:
            return None

        # Keep bool as-is (bool is a subclass of int in Python)
        if isinstance(v, bool):
            return v

        # Round floats
        if isinstance(v, float):
            return round(v, self.decimals)

        # Keep ints as-is
        if isinstance(v, int):
            return v

        # Best-effort support for NumPy scalar types without a hard dependency on NumPy
        # (e.g., np.float32, np.float64, np.int32, ...)
        try:
            import numpy as np  # type: ignore

            if isinstance(v, np.generic):  # NumPy scalar
                py_v = v.item()
                if isinstance(py_v, float):
                    return round(py_v, self.decimals)
                return py_v

            # If it is array-like, do NOT round/serialize it here
            if isinstance(v, np.ndarray):
                return v
        except Exception:
            pass

        return v

    def add(self, **metrics: Any) -> None:
        """Add one metrics row to memory (does not write to disk)."""
        if not metrics:
            return

        # Round values before storing them
        rounded_metrics = {k: self._round_value(v) for k, v in metrics.items()}

        with self._lock:
            if self._fieldnames is None:
                self._fieldnames = list(rounded_metrics.keys())
            else:
                for k in rounded_metrics.keys():
                    if k not in self._fieldnames:
                        self._fieldnames.append(k)

            # Store a copy to avoid external side effects
            self._memory.append(dict(rounded_metrics))

    def extend(self, metrics_list: List[Dict[str, Any]]) -> None:
        """Add multiple rows to memory."""
        for m in metrics_list:
            # Use add() to preserve fieldnames order
            self.add(**m)

    def _ensure_csv_path(self) -> None:
        """Create the final timestamped CSV path on first write."""
        if self.csv_path is not None:
            return
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        req = self._requested_csv_path
        if not req:
            dirpath = os.getcwd()
        else:
            # If a directory was provided or the path ends with a separator, treat as directory
            if str(req).endswith(os.sep) or os.path.isdir(req):
                dirpath = req
            else:
                dirpath = os.path.dirname(req) or os.getcwd()

        os.makedirs(dirpath or ".", exist_ok=True)
        filename = f"ddqn_metrics_episodes_{timestamp}.csv"
        self.csv_path = os.path.join(dirpath, filename)
        self._created_at = now.isoformat(sep=' ', timespec='seconds')

    def _get_git_info(self) -> Dict[str, Optional[str]]:
        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
            return {"branch": branch, "commit": commit}
        except Exception:
            return {"branch": None, "commit": None}

    def _compose_metadata_lines(self, header: List[str]) -> List[str]:
        meta = {
            "created_at": self._created_at or datetime.datetime.now().isoformat(sep=' ', timespec='seconds'),
            "csv_path": self.csv_path,
            "decimals": self.decimals,
            "append": self.append,
            "fieldnames": header,
            "config": self.config,
            "environment": {
                "python": sys.version.replace('\n', ' '),
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "user": getpass.getuser(),
                "cwd": os.getcwd(),
                "pid": os.getpid(),
            },
            "git": self._get_git_info(),
        }
        try:
            meta_json = json.dumps(meta, indent=2, ensure_ascii=False)
        except Exception:
            meta_json = str(meta)

        lines = ["# SaveMetrics metadata"]
        for ln in meta_json.splitlines():
            lines.append("# " + ln)
        return lines

    def _existing_header(self) -> Optional[List[str]]:
        """Read the existing CSV header (if any)."""
        if self.csv_path is None:
            return None
        if not os.path.exists(self.csv_path):
            return None
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                # Skip initial commented metadata lines
                first = None
                for row in reader:
                    if not row:
                        continue
                    # If the line starts with '#' it's metadata; skip it
                    # csv.reader returns a list of columns, but commenting was written as a full line
                    # so if the first cell starts with '#', skip until we find the header
                    if row[0].startswith("#"):
                        continue
                    first = row
                    break
                return first
        except Exception:
            return None

    def write_episode(self, episode: Optional[int] = None) -> None:
        """Write all in-memory metrics to CSV and then clear them.

        If `episode` is provided, each row will be tagged with the 'episode' column.
        """
        with self._lock:
            if not self._memory:
                return

            # Determine the header, preferring explicit fieldnames
            if self._fieldnames:
                header = list(self._fieldnames)
            else:
                seen: List[str] = []
                for row in self._memory:
                    for k in row.keys():
                        if k not in seen:
                            seen.append(k)
                header = seen

            if episode is not None and "episode" not in header:
                header = ["episode"] + header

            # Ensure we have a final CSV path (timestamped) before inspecting filesystem
            self._ensure_csv_path()

            file_exists = os.path.exists(self.csv_path)
            mode = "a" if file_exists and self.append else "w"

            # If the file exists and has a header, respect it when appending
            if file_exists:
                existing = self._existing_header()
                header_to_use = existing if existing else header
            else:
                header_to_use = header

            # Write rows
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            # If creating a new file, write metadata comments first
            with open(self.csv_path, mode, newline="", encoding="utf-8") as f:
                if mode == "w":
                    try:
                        meta_lines = self._compose_metadata_lines(header_to_use)
                        for ln in meta_lines:
                            f.write(ln + "\n")
                    except Exception:
                        # Best-effort: metadata should never prevent writing metrics
                        pass

                writer = csv.DictWriter(f, fieldnames=header_to_use, extrasaction="ignore")
                # If the file is new (mode == 'w') writeheader() after metadata
                if mode == "w" or not file_exists:
                    writer.writeheader()
                for row in self._memory:
                    row_to_write = dict(row)
                    if episode is not None:
                        row_to_write["episode"] = episode
                    # Ensure all columns exist
                    for key in header_to_use:
                        if key not in row_to_write:
                            row_to_write[key] = ""
                    writer.writerow(row_to_write)

            # Clear memory after dumping
            self._memory.clear()

    def flush(self) -> None:
        """Alias to write metrics without tagging the episode."""
        self.write_episode(episode=None)

    def get_memory(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the in-memory metrics."""
        with self._lock:
            return [r.copy() for r in self._memory]

    def clear_memory(self) -> None:
        """Clear memory without writing to disk."""
        with self._lock:
            self._memory.clear()


__all__ = ["SaveMetrics"]
