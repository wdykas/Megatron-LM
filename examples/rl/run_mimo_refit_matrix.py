# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run the four-GPU MIMO refit matrix, retaining each case's log and results."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

CASES = {
    'language_shrink': [],
    'vision_shrink': ['--language-tp', '1', '--vision-tp', '2'],
    'inference_expand': ['--language-tp', '1', '--inference-tp', '2'],
    'idle_rank': ['--language-tp', '1'],
    'tied_shrink': ['--tied'],
    'tied_expand': ['--language-tp', '1', '--inference-tp', '2', '--tied'],
    'tied_to_untied': ['--tied', '--untie-target'],
    'affine_vision': [
        '--language-tp',
        '1',
        '--vision-tp',
        '2',
        '--projector',
        'affine',
        '--bias-free',
    ],
    'affine_bias': ['--projector', 'affine'],
    'gqa_expand': ['--language-tp', '1', '--inference-tp', '2', '--query-groups', '2'],
    'deep_bias_free': ['--layers', '2', '--bias-free'],
    'fp16': ['--dtype', 'fp16'],
    'gloo_expand': ['--language-tp', '1', '--inference-tp', '2', '--backend', 'gloo'],
    'wrapped_batch': ['--wrapped', '--batch-size', '2'],
    'cache_rebuild': ['--rebuild', '--steps', '3'],
    'batched_shrink': ['--execution-batch-bytes', '65536'],
    'batched_expand': [
        '--language-tp',
        '1',
        '--inference-tp',
        '2',
        '--execution-batch-bytes',
        '65536',
    ],
    'batched_idle': ['--language-tp', '1', '--execution-batch-bytes', '65536'],
}


def main():
    """Launch each case in a fresh distributed world and summarize failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke = Path(__file__).with_name('mimo_refit_smoke.py')
    results = {}
    for name in args.cases:
        output = args.output_dir / f'{name}.json'
        # A failed rerun must not leave a previous successful result behind.
        output.unlink(missing_ok=True)
        command = [
            sys.executable,
            '-m',
            'torch.distributed.run',
            '--standalone',
            '--nproc_per_node=4',
            str(smoke),
            '--output',
            str(output),
            *CASES[name],
        ]
        with (args.output_dir / f'{name}.log').open('w') as log:
            with subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            ) as process:
                try:
                    results[name] = process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    results[name] = 'timeout'
        print(f'{name}: {results[name]}', flush=True)
        (args.output_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    return int(any(code != 0 for code in results.values()))


if __name__ == '__main__':
    sys.exit(main())
