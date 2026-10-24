import json

import pytest

from tts_reliability.evaluation import (
    append_record, content_hash, generation_budget, grouped_summary, normalize_text,
    read_records, read_texts, split_sentences, word_errors,
)


def test_normalization_numbers_contractions_and_case():
    assert normalize_text("I've got twenty-one apples!") == normalize_text("I have got 21 apples.")
    assert normalize_text("The year is 2024.") == normalize_text("The year is twenty twenty four")


def test_alignment_preserves_error_evidence():
    score = word_errors("red green blue yellow", "red green orange yellow purple")
    assert (score["substitutions"], score["deletions"], score["insertions"]) == (1, 0, 1)
    assert score["wer"] == 0.5
    assert score["alignment"][-1]["operation"] == "insertion"
    deleted = word_errors("red green blue", "red blue")
    assert deleted["deletions"] == 1
    assert deleted["alignment"][1]["reference"] == "green"


def test_empty_reference_wer_is_undefined_not_infinite():
    assert word_errors("", "one two")["wer"] is None
    assert word_errors("red blue", "")["deletions"] == 2


def test_aggregate_is_micro_average_and_retains_failures():
    rows = [dict(status="complete", seed=1, length_group="short", stop_reason="eos",
                 metrics=word_errors("a b", "a")),
            dict(status="complete", seed=2, length_group="long", stop_reason="output_limit",
                 metrics=word_errors("a b c d", "a b c d")),
            dict(status="error", seed=2, length_group="long", stop_reason="exception")]
    summary = grouped_summary(rows)
    assert summary["overall"]["wer"] == pytest.approx(1 / 6)
    assert summary["overall"]["scored_items"] == 2
    assert summary["overall"]["exception_rate"] == pytest.approx(1 / 3)
    assert summary["overall"]["truncation_rate"] == pytest.approx(1 / 3)
    assert summary["by_length"]["short"]["wer"] == 0.5


def test_resume_events_and_torn_tail(tmp_path):
    path = tmp_path / "raw.jsonl"
    append_record(path, dict(run_id="same", key="x", status="synthesized"))
    append_record(path, dict(run_id="same", key="x", status="complete"))
    with path.open("ab") as stream:
        stream.write(b'{"run_id":')
    assert read_records(path, "same")["x"]["status"] == "complete"
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="manifest mismatch"):
        read_records(path, "different")


def test_corruption_in_middle_must_not_be_silently_discarded(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('broken\n{"run_id":"same","key":"x"}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_records(path, "same")


def test_different_sampling_configuration_changes_fingerprint():
    assert content_hash({"seed": 1, "top_k": 50}) != content_hash({"seed": 1, "top_k": 40})
    assert content_hash({"seed": 1, "top_k": 50}) == content_hash({"top_k": 50, "seed": 1})


def test_jsonl_ids_and_segmentation(tmp_path):
    path = tmp_path / "texts.jsonl"
    path.write_text(json.dumps(dict(id="a", text="Hello.")) + "\n" + json.dumps(dict(id="a", text="Again.")))
    with pytest.raises(ValueError, match="Duplicate"):
        read_texts(path)
    assert split_sentences("First sentence. Second question? Last!") == ["First sentence.", "Second question?", "Last!"]


def test_valid_terminal_json_without_newline_is_safe_to_extend(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"run_id":"same","key":"x"}')
    assert "x" in read_records(path, "same")
    append_record(path, dict(run_id="same", key="y"))
    assert set(read_records(path, "same")) == {"x", "y"}


def test_adaptive_budget_uses_train_ratio_and_tracks_context_separately():
    short = generation_budget(17, 150, 2048)
    long = generation_budget(72, 205, 2048)
    assert short["requested_output_budget"] == 500
    assert long["requested_output_budget"] == 792
    bound = generation_budget(72, 1800, 2048)
    assert bound["requested_output_budget"] == 792
    assert bound["context_output_budget"] == 249
    assert bound["effective_output_budget"] == 249
    assert generation_budget(72, 205, 2048, mode="fixed")["requested_output_budget"] == 500


def test_unicode_quotes_have_identical_contraction_normalization():
    assert normalize_text("I’ll go. There isn’t a cat. You’ll do.") == normalize_text("I'll go. There isn't a cat. You'll do.")
    assert normalize_text("The Duchess’s cook.") == normalize_text("The Duchess's cook.")


def test_sentence_split_preserves_titles_initials_and_closing_quotes():
    text = 'Mr. Holmes met Dr. J. Watson. “Is Mr. Hatherley here?” He nodded.'
    assert split_sentences(text) == ['Mr. Holmes met Dr. J. Watson.', '“Is Mr. Hatherley here?”', 'He nodded.']
    assert split_sentences('Mrs. Smith visited St. Paul. Ms. Jones stayed.') == ['Mrs. Smith visited St. Paul.', 'Ms. Jones stayed.']
    assert split_sentences('Prof. A. B. Cole agreed. Next came tea.') == ['Prof. A. B. Cole agreed.', 'Next came tea.']
