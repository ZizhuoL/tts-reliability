#!/usr/bin/env python3
"""Offline strategy comparisons: actual split subsets, seed effects, and costs.

Example: python scripts/analyze_strategy_costs.py --system A=results/starter-standard-dev
  --system ASplit=results/starter-split-dev --baseline A --expected-items 90
  --output results/analysis/strategy-costs

Missing/incomplete inputs are reported. Budgets remain budgets; partial segment
metadata is never called a complete waveform. No model or final text is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tts_reliability.evaluation import NORMALIZER_VERSION, normalize_text, word_errors

COUNT_FIELDS = ("errors", "reference_words", "substitutions", "deletions", "insertions")


def number(value):
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def total_known(values):
    values = list(values)
    known = [x for x in values if number(x)]
    return dict(sum_known=sum(known) if known else None, known_items=len(known),
                unknown_items=len(values)-len(known), complete_coverage=len(known)==len(values) and bool(values))


def read_input(name, supplied):
    path = supplied / "raw.jsonl" if supplied.is_dir() or supplied.suffix != ".jsonl" else supplied
    if not path.exists():
        return [], {}, dict(input=str(path.resolve()), input_status="unavailable", raw_sha256=None)
    data = path.read_bytes()
    events, ignored = {}, False
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines)-1 and not line.endswith(b"\n"):
                ignored = True
                break
            raise ValueError(f"Corrupt complete line {i+1}: {path}")
        if not isinstance(row, dict) or "id" not in row or "seed" not in row:
            raise ValueError(f"Missing id/seed: {path}:{i+1}")
        key = row.get("key", json.dumps([row["id"], row["seed"]]))
        events[key] = row
    rows = [dict(row, _result_directory=str(path.parent.resolve())) for row in events.values()]
    identities = [(str(r["id"]), r["seed"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Different keys have the same id/seed identity")
    run_ids = {r.get("run_id") for r in rows}
    if len(run_ids)>1:
        raise ValueError("Input mixes run IDs")
    manifest_path = path.parent / "manifest.json"
    analysis_path = path.parent / "analysis_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if path.name == "raw.jsonl" and manifest.get("run_id") and run_ids and run_ids != {manifest["run_id"]}:
            raise ValueError("Raw results and manifest disagree")
    elif analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        source_name = rows[0].get("system", name) if rows else name
        manifest = analysis.get("inputs", {}).get(source_name, {}).get("manifest", {})
        manifest_path = analysis_path
    return rows, manifest, dict(input=str(path.resolve()), input_status="available",
        raw_sha256=hashlib.sha256(data).hexdigest(), raw_bytes=len(data),
        ignored_torn_final_line=ignored, event_lines=len(lines), latest_items=len(rows),
        manifest=manifest, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None)


def describe(row, settings):
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing reference text: {row['id']}")
    segments = row.get("segments")
    if not isinstance(segments, list) or not all(isinstance(s, dict) for s in segments):
        segments = None
    # A saved complete waveform can survive an ASR error; it remains synthesis-complete.
    waveform = bool(row.get("wav") or row.get("wav_path")) and number(row.get("audio_seconds"))
    synthesis_complete = waveform and (row.get("status") in ("complete", "synthesized") or
        (row.get("status") == "error" and row.get("error_stage") == "asr"))
    split_enabled = row.get("split_sentences", settings.get("split_sentences"))
    n = len(segments) if segments is not None else None
    if synthesis_complete and n is not None and n>0:
        state = "actually_split" if n>1 else "not_split"
    else:
        state = "unknown"
    limits = []
    for index, segment in enumerate(segments or []):
        limits.append(dict(index=segment.get("segment_index", index), text=segment.get("text"),
            generated_tokens=segment.get("generated_tokens"), stop_reason=segment.get("stop_reason"),
            requested_output_budget=segment.get("requested_output_budget", segment.get("max_length")),
            effective_output_budget=segment.get("effective_output_budget"),
            context_output_budget=segment.get("context_output_budget"),
            lm_seconds=segment.get("elapsed_seconds"), acoustic_seconds=segment.get("acoustic_seconds")))
    def segment_sum(field):
        values=[segment.get(field) for segment in limits]
        return sum(values) if values and all(number(v) for v in values) else None
    configured_pause = settings.get("pause_seconds")
    sample_rate = row.get("sample_rate", settings.get("sample_rate"))
    sample_rate_basis = "record_or_run_config" if number(sample_rate) and sample_rate>0 else "unknown"
    if sample_rate_basis == "unknown":
        sample_rate = None
        wav_name = row.get("wav_path", row.get("wav"))
        if wav_name:
            wav_path = Path(wav_name)
            if not wav_path.is_absolute() and row.get("_result_directory"):
                wav_path = Path(row["_result_directory"])/wav_path
            if wav_path.is_absolute() and wav_path.is_file():
                try:
                    import soundfile as sf
                    sample_rate = sf.info(wav_path).samplerate
                    sample_rate_basis = "saved_wav_header"
                except (OSError, RuntimeError):
                    pass
    boundaries = n-1 if synthesis_complete and n is not None and n>0 else None
    pause = None
    pause_basis = "unknown"
    if boundaries == 0:
        pause, pause_basis = 0.0, "No segment boundary in completed waveform"
    elif boundaries is not None and split_enabled is True and number(configured_pause) and configured_pause>=0 and number(sample_rate) and sample_rate>0:
        # Saved code inserts one fixed silence block at each boundary. This is
        # a configuration-derived quantity, not an audio-measured silence metric.
        pause = boundaries*round(configured_pause*sample_rate)/sample_rate
        pause_basis = "accounted_from_completed_segments_and_config; not audio silence detection"
    if synthesis_complete and number(row.get("inserted_pause_seconds")):
        pause, pause_basis = row["inserted_pause_seconds"], "Explicitly recorded inserted_pause_seconds"
    duration = row.get("audio_seconds") if synthesis_complete else None
    generation = row.get("generation_seconds") if synthesis_complete else None
    content_duration = duration-pause if number(duration) and number(pause) and duration>pause else None
    hypothesis = row.get("asr_transcript")
    score, missing = None, None
    if row.get("status") != "complete":
        missing = f"status={row.get('status', 'unknown')}"
    elif not isinstance(hypothesis, str) or not normalize_text(hypothesis):
        missing = "empty or missing ASR transcript"
    elif not normalize_text(text):
        missing = "empty normalized reference"
    else:
        score = word_errors(text, hypothesis)
    return dict(id=str(row["id"]), seed=row["seed"], text=text,
        length_group=row.get("length_group", "unspecified"), status=row.get("status"),
        stop_reason=row.get("stop_reason", "unknown"), error_stage=row.get("error_stage"),
        metrics=score, missing_score_reason=missing, asr_transcript=hypothesis,
        synthesis_complete=synthesis_complete, split_enabled=split_enabled, split_state=state,
        observed_segment_metadata_count=n, completed_waveform_segments=n if synthesis_complete else None,
        segments=limits, actual_generated_tokens=row.get("generated_tokens"),
        completed_generated_tokens=row.get("generated_tokens") if synthesis_complete else None,
        partial_or_complete_recorded_requested_budget_sum=segment_sum("requested_output_budget"),
        completed_requested_budget_sum=segment_sum("requested_output_budget") if synthesis_complete else None,
        completed_effective_budget_sum=segment_sum("effective_output_budget") if synthesis_complete else None,
        inserted_boundaries=boundaries, configured_pause_seconds_per_boundary=configured_pause,
        sample_rate=sample_rate, sample_rate_basis=sample_rate_basis,
        inserted_pause_seconds=pause, inserted_pause_basis=pause_basis,
        audio_seconds=duration, generation_seconds=generation,
        generation_rtf=generation/duration if number(generation) and number(duration) and duration>0 else None,
        audio_seconds_excluding_configured_pauses=content_duration,
        generation_rtf_excluding_configured_pauses=generation/content_duration if number(generation) and content_duration else None,
        wav=row.get("wav_path", row.get("wav")))


def scores(rows):
    valid = [r["metrics"] for r in rows if r["metrics"] is not None]
    totals={key:sum(v[key] for v in valid) for key in COUNT_FIELDS}
    words=totals["reference_words"]
    return dict(items=len(rows), scored_items=len(valid), unscored_items=len(rows)-len(valid),
        missing_score_reasons=dict(Counter(r["missing_score_reason"] for r in rows if r["metrics"] is None)),
        **totals, wer=totals["errors"]/words if words else None,
        substitution_rate=totals["substitutions"]/words if words else None,
        deletion_rate=totals["deletions"]/words if words else None,
        insertion_rate=totals["insertions"]/words if words else None)


def costs(rows):
    fields=("completed_waveform_segments", "actual_generated_tokens", "completed_generated_tokens",
            "completed_requested_budget_sum", "completed_effective_budget_sum", "inserted_boundaries",
            "inserted_pause_seconds", "generation_seconds", "audio_seconds", "audio_seconds_excluding_configured_pauses")
    result=dict(items=len(rows), synthesis_complete_items=sum(r["synthesis_complete"] for r in rows),
        split_states=dict(Counter(r["split_state"] for r in rows)),
        stops=dict(Counter(r["stop_reason"] for r in rows)),
        recorded_segment_stops=dict(Counter(s.get("stop_reason", "unknown") for r in rows for s in r["segments"])),
        measurements={key:total_known(r[key] for r in rows) for key in fields})
    for label, denominator in (("rtf", "audio_seconds"), ("rtf_excluding_configured_pauses", "audio_seconds_excluding_configured_pauses")):
        available=[r for r in rows if number(r["generation_seconds"]) and number(r[denominator]) and r[denominator]>0]
        result[label]=dict(available_items=len(available), unknown_items=len(rows)-len(available),
            value=sum(r["generation_seconds"] for r in available)/sum(r[denominator] for r in available) if available else None)
    return result


def paired_scope(pairs):
    common=[(a,b) for a,b in pairs if a["metrics"] is not None and b["metrics"] is not None]
    left,right=scores([a for a,b in common]),scores([b for a,b in common])
    return dict(shared_attempts=len(pairs), common_scored_pairs=len(common), missing_score_pairs=len(pairs)-len(common),
        baseline_all_attempts=scores([a for a,b in pairs]), method_all_attempts=scores([b for a,b in pairs]),
        baseline_common_scores=left, method_common_scores=right,
        delta_wer=right["wer"]-left["wer"] if right["wer"] is not None and left["wer"] is not None else None,
        baseline_costs=costs([a for a,b in pairs]), method_costs=costs([b for a,b in pairs]))


def compare(baseline, method):
    left={(r["id"],r["seed"]):r for r in baseline};right={(r["id"],r["seed"]):r for r in method}
    shared=sorted(left.keys() & right.keys(), key=lambda k:(k[0], str(k[1])))
    pairs=[]
    for key in shared:
        a,b=left[key],right[key]
        if a["text"]!=b["text"] or a["length_group"]!=b["length_group"]:
            raise ValueError(f"Paired reference/group mismatch: {key}")
        pairs.append((a,b))
    def scopes(selected):
        seed_groups=defaultdict(list);length_groups=defaultdict(list)
        for a,b in selected:seed_groups[str(a["seed"])].append((a,b));length_groups[a["length_group"]].append((a,b))
        return dict(overall=paired_scope(selected), by_seed={k:paired_scope(v) for k,v in seed_groups.items()},
                    by_length={k:paired_scope(v) for k,v in length_groups.items()})
    return dict(baseline_only_items=len(left.keys()-right.keys()),method_only_items=len(right.keys()-left.keys()),
        all_shared=scopes(pairs),
        method_actually_split=scopes([(a,b) for a,b in pairs if b["split_state"]=="actually_split"]),
        method_not_split=scopes([(a,b) for a,b in pairs if b["split_state"]=="not_split"]),
        method_split_unknown=scopes([(a,b) for a,b in pairs if b["split_state"]=="unknown"]),
        interpretation="Actually-split subset requires a completed saved waveform with >1 recorded segments. Failed/partial synthesis stays unknown, never counted as unsplit. ASR failures after completed synthesis remain in subset coverage but not common-score WER.")


def analyze_system(rows, expected):
    by_seed=defaultdict(list)
    for row in rows:by_seed[str(row["seed"])].append(row)
    finished=all(r["status"] in ("complete","error") for r in rows)
    return dict(status="complete_observed_attempts" if rows and finished and (expected is None or len(rows)==expected) else "incomplete_or_unavailable",
        expected_items=expected, observed_items=len(rows), expected_count_matches=len(rows)==expected if expected is not None else None,
        scores=scores(rows),costs=costs(rows),by_seed={seed:dict(scores=scores(group),costs=costs(group)) for seed,group in by_seed.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system",action="append",required=True,metavar="NAME=RESULT_DIRECTORY_OR_JSONL")
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-items",type=int)
    args=parser.parse_args()
    systems,provenance={},{}
    for specification in args.system:
        name,value=specification.split("=",1)
        if name in systems or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            parser.error("Unique alphanumeric system names required")
        rows,manifest,provenance[name]=read_input(name,Path(value))
        systems[name]=[describe(row,manifest.get("settings",{})) for row in rows]
    if args.baseline not in systems:parser.error("--baseline must name a supplied system")
    if args.output.exists() and any(args.output.iterdir()):parser.error("--output must be empty or new")
    report=dict(normalizer=NORMALIZER_VERSION,baseline=args.baseline,
        systems={name:analyze_system(rows,args.expected_items) for name,rows in systems.items()},
        paired={name:compare(systems[args.baseline],rows) for name,rows in systems.items() if name!=args.baseline},
        caveats=["Budgets are configured limits, not measured compute or generated-token counts.",
                 "Inserted silence is derived only from completed recorded segments and a saved pause setting plus known sample rate (or an explicit measured field); it is not total acoustic silence or perceived boundary quality.",
                 "RTF uses available matched generation/audio durations; coverage is shown. Excluding configured pauses prevents added silence from artificially improving the denominator.",
                 "Missing input, missing ASR, incomplete synthesis, and unknown costs remain explicit. This observed-record analysis does not establish quality or causal superiority."])
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(inputs=provenance,normalizer=NORMALIZER_VERSION,arguments={**vars(args),"output":str(args.output.resolve())},
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exact_command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),cwd=str(Path.cwd()))
    for name,value in (("summary.json",report),("manifest.json",manifest)):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    for name,rows in systems.items():
        (args.output/f"{name}.strategy-costs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows))
    print(json.dumps({name:report["systems"][name]["status"] for name in systems}))


if __name__=="__main__":
    main()
