#!/usr/bin/env python3
"""CPU-only resource accounting from complete runs' latest raw events.

Example: python scripts/analyze_runtime_resources.py --system A=results/starter-standard-dev
  --expected-items 90 --output results/analysis/resources
Reads only explicitly named raw.jsonl and manifests, never models or text datasets.
Existing strategy-analysis event parsing is reused; no WER rescoring is performed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from analyze_strategy_costs import read_input, number, total_known

TIME_FIELDS = ('generation_seconds', 'asr_seconds', 'attempt_seconds', 'audio_seconds')
MEMORY_FIELDS = ('process_peak_rss_bytes', 'mps_current_allocated_bytes', 'mps_driver_allocated_bytes')
CAVEATS = [
    'Latest event per raw key only. Totals are recorded item timing sums, not whole-run wall-clock or total historical retry cost; model loading and earlier failed attempts may be absent.',
    'ASR-only resume can retain an earlier generation_seconds but replace attempt_seconds. Do not add generation, ASR, and attempt times together or infer overhead by subtraction.',
    'RTF uses matched rows with known nonnegative numerator and positive recorded audio duration; unknown fields stay unknown. Audio duration is the recorded delivered waveform duration including inserted pauses, not speech-only duration; this report does not reread WAVs.',
    'process_peak_rss_bytes is the process cumulative high-water RSS, not per-item incremental memory. The maximum across latest records may span resumed processes; process identity and continuous whole-run peak are not established.',
    'MPS current/driver maxima are observed post-attempt samples, not continuous allocator or system peaks. Host-wide usage, swap, and other processes are not measured here.',
    'Audio >30 seconds flags the configured Whisper chunk window boundary, not an ASR failure or a quality judgment. Completed outputs can still terminate at a generation cap.',
]

def finite_nonnegative(value):
    return value if number(value) and value >= 0 else None


def identity(row):
    return dict(key=row['key'], id=row['id'], seed=row['seed'])


def summarize(rows):
    seconds = {f:total_known(finite_nonnegative(r.get(f)) for r in rows) for f in TIME_FIELDS}
    memory = {}
    for field in MEMORY_FIELDS:
        valid = [(r, finite_nonnegative(r.get(field))) for r in rows]
        valid = [(r,v) for r,v in valid if v is not None]
        maximum = max((v for _,v in valid), default=None)
        memory[field] = dict(max_observed_bytes=maximum, max_observed_gib=maximum/2**30 if maximum is not None else None,
            known_items=len(valid), unknown_items=len(rows)-len(valid),
            max_observation_ids=[identity(r) for r,v in valid if v==maximum])
    rtfs={}
    for field in TIME_FIELDS[:-1]:
        matched=[r for r in rows if finite_nonnegative(r.get(field)) is not None and
                 finite_nonnegative(r.get('audio_seconds')) is not None and r['audio_seconds']>0]
        denominator=sum(r['audio_seconds'] for r in matched)
        numerator=sum(r[field] for r in matched)
        rtfs[field]=dict(value=numerator/denominator if denominator else None,
            numerator_seconds=numerator if matched else None, denominator_audio_seconds=denominator if matched else None,
            matched_items=len(matched), unmatched_items=len(rows)-len(matched))
    durations=[r for r in rows if finite_nonnegative(r.get('audio_seconds')) is not None]
    longest=max((r['audio_seconds'] for r in durations), default=None)
    over=[dict(**identity(r),audio_seconds=r['audio_seconds']) for r in durations if r['audio_seconds']>30]
    stops=Counter(r.get('stop_reason','unknown') for r in rows)
    errors=[r for r in rows if r['status']=='error']
    return dict(items=len(rows), statuses=dict(Counter(r['status'] for r in rows)),seconds=seconds,
        rtf=rtfs,memory=memory,stop_reasons=dict(stops),
        cap_items=sum(stops[k] for k in ('output_limit','context_limit')),
        cap_ids=[dict(**identity(r),stop_reason=r['stop_reason']) for r in rows if r.get('stop_reason') in ('output_limit','context_limit')],
        exception_items=len(errors),exception_stages=dict(Counter(r.get('error_stage','unknown') for r in errors)),
        exception_ids=[dict(**identity(r),error_stage=r.get('error_stage','unknown')) for r in errors],
        longest_audio_seconds=longest,longest_audio_ids=[identity(r) for r in durations if r['audio_seconds']==longest],
        audio_duration_known_items=len(durations),audio_duration_unknown_items=len(rows)-len(durations),
        audio_over_30_seconds_count=len(over),audio_over_30_seconds_ids=over)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--system',action='append',required=True,metavar='NAME=RUN_DIRECTORY_OR_RAW_JSONL')
    parser.add_argument('--expected-items',type=int,required=True,help='Required latest-key count for EVERY supplied system; rejects incomplete/pending runs')
    parser.add_argument('--output',type=Path,required=True,help='New or empty directory outside every input run')
    args=parser.parse_args()
    if args.expected_items<1: parser.error('--expected-items must be positive')
    output=args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())): parser.error('--output must be empty or new')
    inputs={};folders=set()
    for spec in args.system:
        if '=' not in spec: parser.error('Use NAME=RUN_DIRECTORY_OR_RAW_JSONL')
        name,value=spec.split('=',1); supplied=Path(value).resolve()
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name in inputs: parser.error('Unique safe alphanumeric system names required')
        raw=supplied/'raw.jsonl' if supplied.is_dir() else supplied
        if raw.name!='raw.jsonl' or not raw.is_file(): parser.error(f'Missing raw.jsonl: {raw}')
        folder=raw.parent
        if folder in folders: parser.error('Duplicate input run directory')
        if output==folder or folder in output.parents or output in folder.parents: parser.error('Output must be separate from input run directories')
        if not (folder/'manifest.json').is_file(): parser.error(f'Missing run manifest: {folder}')
        folders.add(folder);inputs[name]=raw
    systems={};provenance={};items={}
    for name,raw in inputs.items():
        before_manifest=sha(raw.parent/'manifest.json')
        rows,manifest,p=read_input(name,raw)
        if p.get('ignored_torn_final_line') or len(rows)!=args.expected_items or any(r.get('status') not in ('complete','error') for r in rows):
            parser.error(f'{name}: requires exactly {args.expected_items} latest terminal items and no torn tail; observed {len(rows)}')
        if any(not r.get('key') for r in rows): parser.error(f'{name}: missing raw key')
        if before_manifest != p['manifest_sha256']: parser.error(f'{name}: manifest changed while reading')
        systems[name]=dict(status='all_expected_latest_items_terminal',expected_items=args.expected_items,**summarize(rows),
            by_seed={str(seed):summarize([r for r in rows if r['seed']==seed]) for seed in sorted({r['seed'] for r in rows})})
        provenance[name]=p
        items[name]=[dict(**identity(r),status=r['status'],stop_reason=r.get('stop_reason','unknown'),
            **{f:finite_nonnegative(r.get(f)) for f in TIME_FIELDS+MEMORY_FIELDS}) for r in rows]
    # Finish all validation before creating output. Detect source changes during analysis.
    for name,raw in inputs.items():
        if sha(raw)!=provenance[name]['raw_sha256'] or sha(raw.parent/'manifest.json')!=provenance[name]['manifest_sha256']:
            parser.error(f'{name}: input changed while auditing; retry from frozen completed data')
    output.mkdir(parents=True,exist_ok=True)
    def write(filename,obj):
        (output/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write('summary.json',dict(systems=systems,caveats=CAVEATS))
    source=Path(__file__).resolve()
    dependencies=[source,source.with_name('analyze_strategy_costs.py'),source.parents[1]/'src/tts_reliability/evaluation.py']
    write('manifest.json',dict(inputs=provenance,expected_items=args.expected_items,
        exact_command=shlex.join([sys.executable,*sys.argv]),cwd=str(Path.cwd()),
        implementation_sha256={str(p):sha(p) for p in dependencies}))
    for name,rows in items.items(): write(f'{name}.resources.json',rows)
    write('completion.json',dict(status='resource_audit_complete',systems=list(systems),
        artifacts_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps({n:dict(items=s['items'],cap_items=s['cap_items'],exception_items=s['exception_items'],longest_audio_seconds=s['longest_audio_seconds'],audio_over_30_seconds_count=s['audio_over_30_seconds_count']) for n,s in systems.items()}))

if __name__=='__main__': main()
