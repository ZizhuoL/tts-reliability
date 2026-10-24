#!/usr/bin/env python3
"""Render delivery figures from frozen final summary JSON; never modify frozen outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/final-evaluation-v1/analysis'))
    parser.add_argument('--output', type=Path, default=Path('results/analysis/final-delivery-plots-v2'))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing existing output: {args.output}')
    summaries = sorted(args.input.glob('*/summary.json'))
    if len(summaries) != 8:
        raise SystemExit(f'Expected eight completed analysis summaries, found {len(summaries)}')
    args.output.mkdir(parents=True)
    manifest = {'format_version': 1, 'purpose': 'Delivery-only layout correction; frozen numerical summaries and original plots remain unchanged.',
                'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
                'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
                'arguments': {k: str(v.resolve()) for k, v in vars(args).items()}, 'figures': []}
    for source in summaries:
        summary = json.loads(source.read_text())
        systems = summary['systems']; names = list(systems); groups = ['short', 'medium', 'long']
        fields = ['scored', 'unscored_as_full_deletions']
        values = {field: [[systems[name]['by_length'][group][field]['wer'] * 100
                          for group in groups] for name in names] for field in fields}
        maximum = max(v for matrix in values.values() for row in matrix for v in row if np.isfinite(v))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.subplots_adjust(left=.085, right=.975, bottom=.13, top=.72, wspace=.38)
        x = np.arange(len(groups)); width = .82 / len(names)
        for ax, field in zip(axes, fields):
            for i, name in enumerate(names):
                ax.bar(x + (i - (len(names)-1)/2)*width, values[field][i], width=width,
                       label=name, color=plt.get_cmap('tab10')(i))
            ax.set_xticks(x, groups); ax.set_ylim(0, max(1, maximum*1.13))
            ax.set_ylabel('Micro WER (%)', labelpad=12); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            ax.set_title('Scored attempts' if field == 'scored' else 'Unscored attempts imputed as full deletions', fontsize=11, pad=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.885), ncol=min(5,len(names)), frameon=False)
        fig.suptitle('ASR disagreement by input length; not a perceptual quality measure', y=.97, fontsize=14)
        destination = args.output/source.parent.name; destination.mkdir(); outputs = {}
        for ext in ['png', 'pdf']:
            target = destination/f'wer_by_length.{ext}'
            fig.canvas.draw()
            fig.savefig(target, dpi=180, bbox_inches=None)
            outputs[ext] = {'path': str(target.resolve()), 'sha256': sha(target)}
        plt.close(fig)
        manifest['figures'].append({'analysis': source.parent.name, 'source_summary': str(source.resolve()),
                                    'source_sha256': sha(source), 'systems': names, 'groups': groups,
                                    'plotted_values_percent': values, 'ylim': [0,max(1,maximum*1.13)], 'outputs': outputs})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'REPRODUCE.md').write_text('# Delivery figure reproduction\n\nFrom the project root, with an unused output path:\n\n'
        '```sh\n.venv/bin/python scripts/render_final_delivery_plots.py --input results/final-evaluation-v1/analysis --output results/analysis/final-delivery-plots-v2\n```\n\n'
        'Only layout differs from the original figures: explicit margins, shorter axis labels, panel descriptions and a separated figure legend. All bar values come from the frozen summaries. The two panels are equal here because every final attempt was scored.\n'
        'The script rejects existing output directories. `manifest.json` binds source summaries, script, versions, plotted values and both output formats. Visual review is separately recorded in `validation.json`.\n')

if __name__ == '__main__':
    main()
