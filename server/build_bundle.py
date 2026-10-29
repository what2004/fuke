"""Build a portable training archive with a per-file SHA256 manifest."""
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'finger-v100'
    destination = ROOT / 'dist' / f'{prefix}-{stamp}.tar.gz'
    destination.parent.mkdir(exist_ok=True)
    files = [ROOT / 'najiushi/descrption/hand.xml', ROOT / 'najiushi/descrption/view_hand.py']
    for directory in [ROOT / 'najiushi/rl', ROOT / 'server']:
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(directory)
            if any(part in ('runs', '__pycache__', '.git', 'logs', 'cache', '.matplotlib') for part in relative.parts):
                continue
            if path.is_file() and path.suffix != '.pyc' and path.name != 'installed-server.txt':
                files.append(path)
    manifest = []
    with tarfile.open(destination, 'w:gz') as archive:
        for path in sorted(set(files)):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            manifest.append(f'{hashlib.sha256(data).hexdigest()}  {relative}\n')
            archive.add(path, arcname=f'{prefix}/{relative}', recursive=False)
        data = ''.join(manifest).encode()
        info = tarfile.TarInfo(f'{prefix}/MANIFEST.sha256')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+'.sha256').write_text(f'{digest}  {destination.name}\n')
    print(f'{destination}\nFiles: {len(manifest)} | size: {destination.stat().st_size/1024**2:.2f} MiB\nSHA256: {digest}')


if __name__ == '__main__':
    main()
