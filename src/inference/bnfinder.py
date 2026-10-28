import subprocess
from pathlib import Path

from src.config import BNFINDER_EXEC


def run_bnfinder(
    dataset_path: Path, output_dir: Path, method: str
) -> tuple[bool, str]:
    """Executes BNFinder2 inside the isolated legacy Python 2.7 environment."""
    if not BNFINDER_EXEC.exists():
        return False, f"BNFinder executable not found at {BNFINDER_EXEC}"

    output_file = output_dir / "structure.cpd"
    cmd = [
        str(BNFINDER_EXEC),
        "-e",
        str(dataset_path),
        "-s",
        method,
        "-c",
        str(output_file),
        "-v",
    ]

    try:
        subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True, "Success"
    except subprocess.CalledProcessError as exc:
        return False, f"Error Code {exc.returncode}"