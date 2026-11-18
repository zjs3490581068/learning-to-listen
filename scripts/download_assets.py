"""Download public, revision-pinned A1/A3a assets without proxies or tokens."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import load_config, add_paths_argument


SOURCES = [
    ('datasets', 'BAAI/MusicEval', '20977b88b95a944bf4d55545652958c62af0d35f', 'MusicEval',
     ['README.md', 'MusicEval-full.zip', 'data/']),
    ('models', 'OpenMuQ/MuQ-large-msd-iter', '0562a57814f6f8bbd9fdea0a25921a2fce1a841a', 'MuQ-large-msd-iter',
     ['README.md', 'config.json', 'model.safetensors']),
    ('models', 'zhudi2825/MuQ-Eval', 'fb74cb55e50e0d008b730540fc533473f88c330c', 'MuQ-Eval-A3a',
     ['README.md', 'base.yaml', 'config.yaml', 'checkpoint_info.json', 'model_state_dict.pt']),
]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_json(url):
    with OPENER.open(url, timeout=30) as response:
        return json.load(response)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    expected = item.get('lfs', {}).get('oid')
    if expected is not None:
        return digest(path) == expected
    if item.get('oid'):
        blob = path.read_bytes()
        header = f'blob {len(blob)}\0'.encode()
        return hashlib.sha1(header + blob).hexdigest() == item['oid']
    return True


def download(task):
    kind, repo, revision, directory, item = task
    name = item['path']
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if verified(target, item):
        print(f'VERIFIED {repo}/{name}', flush=True)
        return {'repo': repo, 'revision': revision, 'file': name,
                'path': f'{directory.name}/{name}', 'bytes': item['size'],
                'sha256': digest(target), 'status': 'verified'}
    if target.exists():
        raise RuntimeError(f'Existing file fails verification; left untouched: {target}')
    partial = target.with_name(target.name + '.part')
    prefix = 'datasets/' if kind == 'datasets' else ''
    for host in ['https://hf-mirror.com', 'https://huggingface.co']:
        url = f'{host}/{prefix}{repo}/resolve/{revision}/{name}?download=true'
        print(f'DOWNLOAD {repo}/{name} ({item["size"] / 1e6:.1f} MB) via {host}',
              flush=True)
        result = subprocess.run([
            'curl', '--disable', '--proxy', '', '--noproxy', '*',
            '--fail', '--location', '--silent', '--show-error',
            '--connect-timeout', '10', '--max-time', '1200',
            '--speed-time', '60', '--speed-limit', '1024',
            '--retry', '2', '--retry-delay', '3', '--continue-at', '-',
            '--output', str(partial), url,
        ], check=False)
        if result.returncode == 0 and verified(partial, item):
            partial.rename(target)
            print(f'OK {repo}/{name}', flush=True)
            return {'repo': repo, 'revision': revision, 'file': name,
                    'path': f'{directory.name}/{name}', 'bytes': item['size'],
                    'sha256': digest(target), 'source': host, 'status': 'verified'}
        print(f'DIRECT FAILED {repo}/{name}, curl={result.returncode}', flush=True)
    return {'repo': repo, 'revision': revision, 'file': name,
            'path': f'{directory.name}/{name}', 'status': 'needs-manual-upload'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--config', default='configs/A1_true_ttm.yaml')
    parser.add_argument('--assets', choices=('a1', 'personal', 'all'), default='all')
    add_paths_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths)
    for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                'all_proxy', 'ALL_PROXY'):
        os.environ.pop(key, None)
    os.environ.update(NO_PROXY='*', no_proxy='*')
    tasks = []
    for kind, repo, revision, local_name, patterns in SOURCES:
        if args.assets == 'a1' and local_name == 'MuQ-Eval-A3a':
            continue
        directory = Path(cfg.storage.data_root if kind == 'datasets' else cfg.storage.models_root) / local_name
        for host in ('https://hf-mirror.com', 'https://huggingface.co'):
            try:
                metadata = get_json(f'{host}/api/{kind}/{repo}/revision/{revision}')
                tree = get_json(f'{host}/api/{kind}/{repo}/tree/{revision}?recursive=true')
                break
            except (OSError, ValueError) as error:
                print(f'DIRECT metadata failure: {host}, {repo}: {error}', flush=True)
        else:
            raise RuntimeError(f'No direct metadata route. Manually prepare {repo}@{revision} in {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'source-metadata.json').write_text(
            json.dumps({'metadata': metadata, 'tree': tree}, indent=2) + '\n')
        for item in tree:
            if item.get('type') != 'file':
                continue
            if any(item['path'] == p or (p.endswith('/') and item['path'].startswith(p))
                   for p in patterns):
                tasks.append((kind, repo, revision, directory, item))
    print(f'{len(tasks)} files; {sum(t[4]["size"] for t in tasks) / 1e9:.2f} GB', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, tasks))
    manifest = Path(cfg.storage.models_root) / 'download-manifest.json'
    manifest.write_text(json.dumps(results, indent=2) + '\n')
    failed = [item for item in results if item['status'] != 'verified']
    print(f'Manifest: {manifest}; {len(results) - len(failed)} verified, {len(failed)} failed')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
