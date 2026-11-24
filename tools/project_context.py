"""Record timestamped staged changes; reject commits with stale project context."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HANDOFF = 'PROJECT_HANDOFF.md'
HISTORY = 'PROJECT_HISTORY.jsonl'
TAIPEI = timezone(timedelta(hours=8), 'Asia/Taipei')
STAMP = re.compile(r'^Context tracking updated: .*$', re.MULTILINE)


def git(root, *args, optional=False):
    result = subprocess.run(['git', *args], cwd=root, capture_output=True)
    if result.returncode and not optional:
        raise ValueError(result.stderr.decode(errors='replace').strip())
    return None if result.returncode else result.stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def head(root):
    value = git(root, 'rev-parse', '--verify', 'HEAD', optional=True)
    return value.decode().strip() if value else None


def staged_files(root):
    changed = git(root, 'diff', '--cached', '--name-only', '--no-renames', '-z')
    records = {}
    for path in sorted(p.decode('utf-8') for p in changed.split(b'\0') if p):
        if path in (HANDOFF, HISTORY):
            continue
        blob = git(root, 'show', ':' + path, optional=True)
        records[path] = None if blob is None else sha(blob)
    return records


def parse_history(data):
    entries = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    if not all(isinstance(entry, dict) and entry.get('schema') == 1 for entry in entries):
        raise ValueError('Unknown project history format')
    return entries


def previous_history(root):
    return git(root, 'show', 'HEAD:' + HISTORY, optional=True) or b''


def append_only(old, new):
    if not new.startswith(old) or len(new) <= len(old):
        raise ValueError('History must append a new entry without rewriting committed records')


def record(root, summary, validation, next_step, artifacts=()):
    if not all(value.strip() for value in (summary, validation, next_step)):
        raise ValueError('Summary, validation and next step must be non-empty')
    history_path = root / HISTORY
    old = history_path.read_bytes() if history_path.exists() else b''
    committed = previous_history(root)
    if not old.startswith(committed):
        raise ValueError('Working history rewrites committed entries; preserve the original history')
    entries = parse_history(old)
    if old and not old.endswith(b'\n'):
        raise ValueError('History must end in a newline before appending')
    timestamp = datetime.now(TAIPEI).isoformat(timespec='microseconds')
    if entries and datetime.fromisoformat(entries[-1]['timestamp']) >= datetime.fromisoformat(timestamp):
        raise ValueError('Clock is not later than the latest record; check the system clock')
    handoff_path = root / HANDOFF
    content = handoff_path.read_text(encoding='utf-8')
    stamp = 'Context tracking updated: ' + timestamp + ' (Asia/Taipei).'
    if STAMP.search(content):
        content = STAMP.sub(stamp, content)
    else:
        title, separator, body = content.partition('\n')
        if not separator:
            raise ValueError('Handoff requires a title and body')
        content = title + '\n\n' + stamp + '\n' + body
    # LF bytes are pinned in .gitattributes so index and working-copy hashes agree.
    handoff_bytes = content.encode('utf-8')
    entry = dict(schema=1, timestamp=timestamp, timezone='Asia/Taipei',
                 parent_commit=head(root), summary=summary.strip(),
                 validation=validation.strip(), next_step=next_step.strip(),
                 artifacts=list(artifacts), files=staged_files(root),
                 handoff_sha256=sha(handoff_bytes))
    encoded = json.dumps(entry, ensure_ascii=True, sort_keys=True).encode() + b'\n'
    handoff_path.write_bytes(handoff_bytes)
    with history_path.open('ab') as stream:
        stream.write(encoded)
    return entry


def check(root):
    staged_history = git(root, 'show', ':' + HISTORY, optional=True)
    staged_handoff = git(root, 'show', ':' + HANDOFF, optional=True)
    if staged_history is None or staged_handoff is None:
        raise ValueError('Stage both PROJECT_HANDOFF.md and PROJECT_HISTORY.jsonl')
    old = previous_history(root)
    append_only(old, staged_history)
    entries = parse_history(staged_history)
    latest = entries[-1]
    timestamp = datetime.fromisoformat(latest['timestamp'])
    if timestamp.utcoffset() != timedelta(hours=8) or latest.get('timezone') != 'Asia/Taipei':
        raise ValueError('Expected timestamp in Asia/Taipei (+08:00)')
    if latest['parent_commit'] != head(root):
        raise ValueError('Record belongs to another parent commit; record this task again')
    if latest['files'] != staged_files(root):
        raise ValueError('Staged files changed since the record; refresh context and record again')
    if latest['handoff_sha256'] != sha(staged_handoff):
        raise ValueError('Staged handoff does not match the latest record')
    for key in ('summary', 'validation', 'next_step'):
        if not isinstance(latest.get(key), str) or not latest[key].strip():
            raise ValueError('Missing context field: ' + key)
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    writer = commands.add_parser('record')
    writer.add_argument('--summary', required=True)
    writer.add_argument('--validation', required=True)
    writer.add_argument('--next-step', required=True)
    writer.add_argument('--artifact', action='append', default=[])
    commands.add_parser('check')
    args = parser.parse_args()
    try:
        root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').decode().strip())
        if args.command == 'record':
            entry = record(root, args.summary, args.validation, args.next_step, args.artifact)
            print(f'Recorded {entry["timestamp"]}: {len(entry["files"])} staged files.')
            print('Now stage PROJECT_HANDOFF.md and PROJECT_HISTORY.jsonl before committing.')
        else:
            entry = check(root)
            print('Project context is current: ' + entry['timestamp'])
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'Project context check failed: {error}\nFollow CONTEXT_TRACKING.md; do not bypass the hook.',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
