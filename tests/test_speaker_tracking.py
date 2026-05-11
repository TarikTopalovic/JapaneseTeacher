from core.speaker_tracking import SpeakerTracker


def test_pick_label_prefers_max_overlap():
    turns = [
        (0.0, 1.0, "A"),
        (1.0, 3.0, "B"),
        (3.0, 4.0, "A"),
    ]
    picked = SpeakerTracker._pick_label(1.2, 2.0, turns)
    assert picked == "B"


def test_normalize_labels_maps_stable_speaker_ids():
    labels = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    normalized = SpeakerTracker._normalize_labels(labels)
    assert normalized == {
        "SPEAKER_00": "SPEAKER_1",
        "SPEAKER_01": "SPEAKER_2",
    }
