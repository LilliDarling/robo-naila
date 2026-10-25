import argparse
import asyncio
import json
from pathlib import Path
import sys
import wave

from .artifacts import summarize
from .traces import EvidenceError, TraceReader


def bounded(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f'expected {low}–{high}')
        return number
    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sequential voice benchmark; YOLO deferred to #46')
    commands = parser.add_subparsers(dest='command', required=True)
    voice = commands.add_parser('voice')
    voice.add_argument('--hub-url', required=True)
    voice.add_argument('--wav', type=Path, required=True)
    voice.add_argument('--run-dir', type=Path, required=True)
    voice.add_argument('--traces', type=Path, required=True)
    voice.add_argument('--deployment', type=Path, help='hardware/config/status provenance JSON')
    voice.add_argument('--warmup', type=bounded(0, 20), default=2)
    voice.add_argument('--iterations', type=bounded(1, 1000), default=20)
    voice.add_argument('--timeout-seconds', type=bounded(1, 600), default=120)
    offline = commands.add_parser('summarize')
    offline.add_argument('--run-dir', type=Path, required=True)
    offline.add_argument('--traces', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'voice':
            from .voice import run
            return asyncio.run(run(args))
        manifest = json.loads((args.run_dir / 'manifest.json').read_text())
        samples = [json.loads(line) for line in (args.run_dir / 'samples.jsonl').read_text().splitlines()]
        # Atomic summary publication is also the execution-finalization record;
        # recompute trace statistics, but never erase its execution error.
        try:
            finalization = json.loads((args.run_dir / 'summary.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            finalization = {}
        reader = TraceReader(args.traces)
        error = None
        try:
            snapshot = reader.read(final=True)
        except EvidenceError as exc:
            snapshot, error = reader.snapshot(), str(exc)
        result = summarize(manifest, samples, snapshot, error,
                           execution_error=finalization.get('execution_error'),
                           execution_finalized=finalization.get('execution_finalized') is True)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['complete'] else 1
    except (OSError, ValueError, ImportError, EOFError, wave.Error) as exc:
        print(f'setup error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
