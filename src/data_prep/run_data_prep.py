from __future__ import annotations

import subprocess
import sys

MODULES = [
    "src.data_prep.scan_dataset",
    "src.data_prep.extract_scene_metadata",
    "src.data_prep.validate_frames",
    "src.data_prep.link_pix4d",
    "src.data_prep.link_nadir",
    "src.data_prep.build_benchmark_subset",
    "src.data_prep.prepare_roi",
    "src.data_prep.build_master_manifest",
]


def main() -> None:
    args = sys.argv[1:]

    for module in MODULES:
        cmd = [sys.executable, "-m", module, *args]
        print("[run_data_prep]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print("[run_data_prep] Done.")


if __name__ == "__main__":
    main()

