"""Install the Piper binary and a voice.

Downloads are checked against pinned SHA-256 digests before anything is
extracted or marked executable, and archives are extracted with the member
filter that refuses paths escaping the target directory. An installer that
fetches a tarball over the network and runs chmod +x on what comes out should
not simply hope.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PIPER_VERSION = "1.2.0"
RELEASE_BASE = f"https://github.com/rhasspy/piper/releases/download/v{PIPER_VERSION}"
VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

#: Bytes read at a time when hashing, so a 60 MB model is not held in memory.
CHUNK = 1 << 20


@dataclass(frozen=True)
class Artefact:
    """One downloadable file and the digest it must have."""

    url: str
    sha256: str


@dataclass(frozen=True)
class Voice:
    """A Piper voice: the model and its configuration."""

    model: Artefact
    config: Artefact
    language: str


#: Verified against the published releases. A mismatch means the file is not
#: the one these digests were taken from, whatever the reason.
BINARIES: dict[tuple[str, str], Artefact] = {
    ("Linux", "x86_64"): Artefact(
        f"{RELEASE_BASE}/piper_amd64.tar.gz",
        "467c17935d2a22dcce9dc9e08ba07485e29be813097e7cf08c5627aa09d32e42",
    ),
    ("Linux", "aarch64"): Artefact(
        f"{RELEASE_BASE}/piper_arm64.tar.gz",
        "34b298f6b3e55b55e81f05c6157310f9ec4df3fdd3d73e4c85eb80e218c54d2c",
    ),
}

VOICES: dict[str, Voice] = {
    "en_US-lessac-medium": Voice(
        model=Artefact(
            f"{VOICE_BASE}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
        ),
        config=Artefact(
            f"{VOICE_BASE}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
        ),
        language="English",
    ),
    "fr_FR-siwis-medium": Voice(
        model=Artefact(
            f"{VOICE_BASE}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
            "641d1ab097da2b81128c076810edb052b385decc8be3381814802a64a73baf99",
        ),
        config=Artefact(
            f"{VOICE_BASE}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
            "39479916c2db192b5ac9764daddd0c744d83e023ad890c6976c0633ae4df8959",
        ),
        language="French",
    ),
}

DEFAULT_VOICE = "en_US-lessac-medium"


class VerificationError(RuntimeError):
    """A download is not the file it was supposed to be."""


def sha256_of(path: Path) -> str:
    """Digest of a file, read in pieces."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, dest: Path) -> None:
    """Download with a progress line."""
    print(f"Downloading {url}...")

    def report(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size / total_size) * 100)
            print(f"\rProgress: {percent:.1f}%", end="")

    urllib.request.urlretrieve(url, dest, report)
    print()


def fetch_verified(artefact: Artefact, dest: Path, force: bool = False) -> Path:
    """Download a file and check it, or reuse one already verified.

    A file that fails the check is deleted: leaving it on disk invites a
    later run to pick it up.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if sha256_of(dest) == artefact.sha256:
            print(f"Already present and verified: {dest.name}")
            return dest
        print(f"{dest.name} is present but does not match; downloading again")
        dest.unlink()

    download_file(artefact.url, dest)

    actual = sha256_of(dest)
    if actual != artefact.sha256:
        dest.unlink(missing_ok=True)
        raise VerificationError(
            f"{dest.name} does not match its expected digest.\n"
            f"  expected {artefact.sha256}\n"
            f"  got      {actual}\n"
            f"The download was corrupted, or the file at {artefact.url} changed."
        )

    print(f"Verified {dest.name}")
    return dest


def _member_is_safe(member: tarfile.TarInfo, target: Path) -> bool:
    """Whether a tar member stays inside the target directory."""
    if member.issym() or member.islnk():
        return False
    name = Path(member.name)
    if name.is_absolute() or ".." in name.parts:
        return False
    return (target / name).resolve().is_relative_to(target.resolve())


def extract_safely(archive: Path, target: Path) -> None:
    """Extract, refusing members that would write outside ``target``.

    tarfile's data filter does this; it landed in 3.12 and was backported, so
    the manual check covers interpreters that do not have it. Without either,
    a crafted archive can write anywhere the user can - and this one is then
    marked executable.
    """
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(target, filter="data")
            return
        except TypeError:
            pass  # older interpreter; fall through to the manual check

        members = list(tar.getmembers())
        unsafe = [m.name for m in members if not _member_is_safe(m, target)]
        if unsafe:
            raise VerificationError(
                f"{archive.name} contains entries that would write outside "
                f"{target}: {', '.join(unsafe[:3])}"
            )
        tar.extractall(target)


def install_voice(name: str, models_dir: Path) -> Path:
    """Download and verify one voice."""
    voice = VOICES[name]
    print(f"\nInstalling the {voice.language} voice {name}...")
    model = fetch_verified(voice.model, models_dir / f"{name}.onnx")
    fetch_verified(voice.config, models_dir / f"{name}.onnx.json")
    return model


def setup_piper(voice: str = DEFAULT_VOICE, piper_dir: Path | None = None) -> bool:
    """Install the binary and one voice. Returns whether it worked."""
    if voice not in VOICES:
        print(f"Unknown voice: {voice}. Known voices: {', '.join(sorted(VOICES))}")
        return False

    piper_dir = Path(piper_dir) if piper_dir else Path("piper")
    system, machine = platform.system(), platform.machine()
    print(f"Detected system: {system} {machine}")

    artefact = BINARIES.get((system, machine))
    if artefact is None:
        print(f"No published Piper build for {system} {machine}.")
        print("Install it by hand: https://github.com/rhasspy/piper")
        return False

    try:
        archive = fetch_verified(
            artefact, piper_dir / Path(artefact.url).name
        )
        print("Extracting Piper...")
        extract_safely(archive, piper_dir)

        binary = piper_dir / "piper" / "piper"
        if binary.exists():
            binary.chmod(0o755)
        else:
            print(f"Warning: no binary at {binary} after extraction")

        model = install_voice(voice, piper_dir / "models")
    except VerificationError as e:
        print(f"\nRefusing to continue: {e}")
        return False
    except OSError as e:
        print(f"\nInstallation failed: {e}")
        return False

    print("\nPiper setup complete.")
    print(f"Binary: {binary}")
    print(f"Voice : {model}")
    print(f"\nSet PIPER_MODEL={voice} in your .env")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice", default=DEFAULT_VOICE,
        help=f"which voice to install (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="show the known voices and exit",
    )
    arguments = parser.parse_args()

    if arguments.list_voices:
        for name, voice in sorted(VOICES.items()):
            print(f"  {name:24} {voice.language}")
        return 0

    print("=" * 50)
    print("Piper TTS Setup")
    print("=" * 50)

    if not setup_piper(arguments.voice):
        print("\nSetup failed")
        return 1

    print("\nNext steps:")
    print("1. Install Python dependencies: pip install -r requirements.txt")
    print("2. Set up Ollama: https://ollama.ai/download")
    print("3. Copy .env.example to .env and configure")
    print("4. Run: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
