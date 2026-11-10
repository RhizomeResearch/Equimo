"""The hosted full-weight SAM 3.1 video fixture.

The fixture holds a clip's preprocessed frames and the float32 reference's decision
trace and per-frame masks, too large to keep in the repository. It is packed as a
``.tar.lz4`` beside the converted archives so that ``equimo.serialization.download``
fetches and verifies it with the digest pinned here. Packing and reading a fixture
archive only need the standard library and ``lz4``.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import lz4.frame
import numpy as np

from equimo.serialization import DEFAULT_REPOSITORY_URL, download

IDENTIFIER = "sam31_video_reference"
ARCHIVE_SHA256 = "5946d471743bac299736484adffca8f7349ca28be890dde791edbcdc9b7e016f"
_MEMBER = f"{IDENTIFIER}.npz"


def fixture_path() -> Path:
    """Return where the downloaded fixture archive is cached."""
    return Path(f"~/.cache/equimo/sam31/{IDENTIFIER}.tar.lz4").expanduser()


def download_fixture() -> Path:
    """Download the fixture archive, verifying its pinned digest."""
    return download(IDENTIFIER, DEFAULT_REPOSITORY_URL, expected_sha256=ARCHIVE_SHA256)


def load_fixture(archive: Path) -> dict[str, np.ndarray]:
    """Read the arrays out of a fixture archive."""
    with lz4.frame.open(archive, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as contents:
            for member in contents:
                if member.name != _MEMBER:
                    continue
                payload = contents.extractfile(member).read()
                with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                    return {name: arrays[name] for name in arrays.files}
    raise ValueError(f"{archive} does not contain {_MEMBER}.")


def write_fixture_archive(npz: Path, archive: Path) -> None:
    """Pack a generated fixture deterministically, so its digest is reproducible."""
    payload = npz.read_bytes()
    info = tarfile.TarInfo(name=_MEMBER)
    info.size = len(payload)
    with lz4.frame.open(
        archive, "wb", content_checksum=True, block_checksum=True
    ) as output:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as tar:
            tar.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    write_fixture_archive(Path(sys.argv[1]), Path(sys.argv[2]))
