from dataclasses import fields
import csv
import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from vlincs_gallery.weak_graph import TrackletEvidence, WeakGraphConfig, resolve_weak_graph
from vlincs_gallery.weak_labels import WeakLabelGenerationConfig, generate_weak_labels, weak_tokens_from_tracklet
from vlincs_gallery.feature_centralization import neighbor_feature_centralization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kit"))
from online import OnlineGallery  # noqa: E402


def test_tracklet_evidence_schema_has_no_reference_identity_field():
    names = {field.name for field in fields(TrackletEvidence)}

    assert "global_id" not in names
    assert "gt_id" not in names
    assert "reference_global_id" not in names
    assert not any("reference" in name for name in names)


def test_neighbor_feature_centralization_uses_mutual_neighbours_without_labels():
    feats = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    out, info = neighbor_feature_centralization(feats, k1=1, k2=1, eta=1.0)

    assert out.shape == feats.shape
    assert info.n_mutual_edges == 2
    assert np.linalg.norm(out, axis=1).round(6).tolist() == [1.0, 1.0, 1.0]
    assert float(out[0] @ out[1]) > float(feats[0] @ feats[1])


def test_weak_graph_merges_cross_camera_visual_and_language_evidence():
    records = [
        TrackletEvidence(
            "trk_a",
            np.array([1.0, 0.0, 0.0], np.float32),
            video="video_1",
            camera="MCAM04",
            start_frame=10,
            end_frame=40,
            weak_tokens={"upper_color": "black", "lower_type": "pants", "hat": "no"},
        ),
        TrackletEvidence(
            "trk_b",
            np.array([0.99, 0.03, 0.0], np.float32),
            video="video_1",
            camera="MCAM08",
            start_frame=15,
            end_frame=45,
            weak_tokens={"upper_color": "black", "lower_type": "pants", "hat": "no"},
        ),
        TrackletEvidence(
            "trk_c",
            np.array([-1.0, 0.0, 0.0], np.float32),
            video="video_1",
            camera="MCAM08",
            start_frame=80,
            end_frame=120,
            weak_tokens={"upper_color": "red", "lower_type": "shorts", "hat": "yes"},
        ),
    ]

    result = resolve_weak_graph(
        records,
        WeakGraphConfig(
            visual_top_k=2,
            edge_threshold=0.58,
            max_token_df=None,
            lp_iterations=4,
        ),
    )
    assignments = {row.tracklet_key: row for row in result.assignments}

    assert result.summary["uses_ground_truth"] is False
    assert result.summary["accepted_edges"] >= 1
    assert assignments["trk_a"].predicted_global_id == assignments["trk_b"].predicted_global_id
    assert assignments["trk_c"].predicted_global_id != assignments["trk_a"].predicted_global_id
    assert {row.decision_status for row in result.assignments} == {"forced"}
    assert assignments["trk_a"].component_size == 2


def test_same_camera_overlap_is_cannot_link_even_with_matching_weak_tokens():
    records = [
        TrackletEvidence(
            "trk_a",
            np.array([1.0, 0.0, 0.0], np.float32),
            video="video_1",
            camera="MCAM04",
            start_frame=10,
            end_frame=40,
            weak_tokens="upper_color:black|lower_type:pants|hat:no",
        ),
        TrackletEvidence(
            "trk_b",
            np.array([0.99, 0.02, 0.0], np.float32),
            video="video_1",
            camera="MCAM04",
            start_frame=25,
            end_frame=55,
            weak_tokens="upper_color:black|lower_type:pants|hat:no",
        ),
    ]

    result = resolve_weak_graph(
        records,
        WeakGraphConfig(
            visual_top_k=1,
            edge_threshold=0.50,
            max_token_df=None,
            lp_iterations=4,
        ),
    )

    assert result.summary["candidate_edges"] == 1
    assert result.summary["accepted_edges"] == 0
    assert result.summary["cannot_link_edges"] == 1
    assert result.candidate_edges[0].cannot_link is True
    assert result.assignments[0].predicted_global_id != result.assignments[1].predicted_global_id
    assert all(row.decision_status == "forced" for row in result.assignments)


class FakeCursor:
    def __init__(self):
        self.mapping_rows = [
            ("external_a", "trk_a", 1),
            ("external_b", "trk_b", 2),
        ]
        self.rows_by_role = {
            "resolve": [],
            "match": [
                (
                    1,
                    "trk_a",
                    "video_1",
                    "MCAM04",
                    np.array([1.0, 0.0, 0.0], np.float32),
                    5,
                    10,
                    40,
                    {"upper_color": "black", "lower_type": "pants", "hat": "no"},
                    0.91,
                ),
                (
                    2,
                    "trk_b",
                    "video_1",
                    "MCAM08",
                    np.array([0.99, 0.03, 0.0], np.float32),
                    4,
                    15,
                    45,
                    {"upper_color": "black", "lower_type": "pants", "hat": "no"},
                    0.88,
                ),
                (
                    3,
                    "trk_c",
                    "video_1",
                    "MCAM08",
                    np.array([-1.0, 0.0, 0.0], np.float32),
                    6,
                    80,
                    120,
                    {"upper_color": "red", "lower_type": "shorts", "hat": "yes"},
                    0.83,
                ),
            ],
        }
        self.executed = []
        self.executemany_calls = []
        self._last_rows = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "WHERE e.role = %s" in sql:
            role = params[1]
            self._last_rows = self.rows_by_role[role]
        elif "SELECT tracklet_key, entity_id, seq FROM tracklets" in sql:
            wanted = set(params[0])
            self._last_rows = [row for row in self.mapping_rows if row[0] in wanted]

    def executemany(self, sql, params):
        rows = list(params)
        self.executemany_calls.append((sql, rows))

    def fetchall(self):
        return list(self._last_rows)


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1


def test_online_gallery_resolve_weak_global_applies_forced_assignments_from_db_evidence():
    fake = FakeConnection()
    gallery = OnlineGallery.__new__(OnlineGallery)
    gallery.con = fake
    gallery._global_resolved = False

    info = gallery.resolve_weak_global(
        WeakGraphConfig(visual_top_k=2, edge_threshold=0.58, max_token_df=None, lp_iterations=4),
        embedding_role="resolve",
        fallback_to_match=True,
        weak_source="clip-vitb32",
        apply=True,
        offset=20_000_000,
    )

    assert info["embedding_role"] == "match"
    assert info["n_tracklets"] == 3
    assert info["n_weak_labels"] == 3
    assert info["decision_status"] == "forced"
    assert info["uses_ground_truth"] is False
    assert info["clusters"] == 2
    assert gallery._global_resolved is True
    assert fake.commits >= 2

    update_assignment_calls = [
        (sql, params)
        for sql, params in fake.cursor_obj.executed
        if "UPDATE assignments a SET gid = r.g" in sql and "decision_type = 'forced'" in sql
    ]
    assert update_assignment_calls, "forced assignment update was not issued"
    _sql, params = update_assignment_calls[0]
    seqs, gids, scores = params
    assert seqs == [1, 2, 3]
    assert gids[0] == gids[1]
    assert gids[2] != gids[0]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert any(
        "WHERE gid >= %s AND NOT (gid = ANY" in sql
        for sql, _params in fake.cursor_obj.executed
    ), "stale weak identities were not cleared"
    assert any(
        "a.decision_type <> 'forced'" in sql
        for sql, _params in fake.cursor_obj.executed
    ), "forced delivery outputs must not populate committed identity summaries"


def test_online_gallery_import_weak_labels_csv_joins_by_tracklet_key_and_ignores_reference_columns():
    fake = FakeConnection()
    gallery = OnlineGallery.__new__(OnlineGallery)
    gallery.con = fake

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "weak.csv"
        csv_path.write_text(
            "tracklet_key,upper_color,hat,reference_global_id,confidence\n"
            "external_a,black,no,M0012,0.9\n"
            "external_b,red,yes,M0007,0.8\n"
            "missing_key,blue,no,M0008,0.7\n",
            encoding="utf-8",
        )
        info = gallery.import_weak_labels_csv(str(csv_path), source="clip-vitb32")

    assert info["csv_rows"] == 3
    assert info["matched"] == 2
    assert info["missing"] == 1
    assert info["ignored_identity_columns"] == ["reference_global_id"]

    weak_insert_calls = [
        rows for sql, rows in fake.cursor_obj.executemany_calls
        if "weak_tracklet_labels" in sql
    ]
    assert weak_insert_calls, "weak labels were not upserted"
    rows = weak_insert_calls[0]
    assert rows[0][0] == "trk_a"
    assert rows[0][1] == 1
    assert rows[0][2] == "clip-vitb32"
    assert "reference_global_id" not in rows[0][3]
    assert "upper_color" in rows[0][3]
    assert fake.commits >= 1


def test_demo_tracklet_record_supports_stable_tracklet_key():
    demo = importlib.import_module("demo")

    legacy = ("video", "MCAM04", [1], [[0, 0, 1, 1]], np.ones(2), [0.9], 0, ["det_a"])
    current = legacy + ("video::tracklet-001",)

    assert demo._unpack_tracklet_record(legacy)[-1] is None
    assert demo._unpack_tracklet_record(current)[-1] == "video::tracklet-001"


def test_demo_ds1_loader_preserves_tracklet_key_and_resolve_alignment():
    demo = importlib.import_module("demo")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stem = "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6"
        for sub in ("tracklets", "embeddings", "resolve/embeddings"):
            (root / sub / stem).mkdir(parents=True)
        tracklet_df = pd.DataFrame(
            {
                "tracklet_key": ["trk_keep", "trk_keep"],
                "frame_idx": [12000, 12001],
                "x1": [10.0, 11.0],
                "y1": [20.0, 21.0],
                "x2": [30.0, 31.0],
                "y2": [40.0, 41.0],
                "score": [0.90, 0.91],
                "local_track_id": [7, 7],
                "coco_cls": [0, 0],
            }
        )
        (root / "tracklets" / stem / "tracklets.parquet").write_bytes(b"parquet-placeholder")
        np.savez(
            root / "embeddings" / stem / "embeddings.npz",
            vectors=np.array([[1.0, 0.0, 0.0]], np.float32),
            track_ids=np.array(["trk_keep"], dtype=object),
        )
        np.savez(
            root / "resolve/embeddings" / stem / "embeddings.npz",
            vectors=np.array([[0.0, 1.0, 0.0, 0.0]], np.float32),
            track_ids=np.array(["trk_keep"], dtype=object),
        )

        orig_input_dir = demo._ds1_input_dir
        orig_read_parquet = pd.read_parquet
        demo._ds1_input_dir = lambda sub, run, mlflow_path: (root / sub, "unit-test")
        pd.read_parquet = lambda path: tracklet_df.copy()
        try:
            tracklets, resolve = demo._ds1_from_mlflow(
                {
                    "inputs": {
                        "track": {"mlflow_run": "track"},
                        "embed": {"mlflow_run": "embed"},
                        "resolve_embed": {"mlflow_run": "resolve"},
                    },
                    "reduce": {"method": "none"},
                }
            )
        finally:
            demo._ds1_input_dir = orig_input_dir
            pd.read_parquet = orig_read_parquet

    assert len(tracklets) == 1
    assert tracklets[0][-1] == "trk_keep"
    assert tracklets[0][7] == [
        f"{stem}::MCAM04:12000:7:0",
        f"{stem}::MCAM04:12001:7:0",
    ]
    assert resolve.shape == (1, 4)
    assert np.allclose(resolve[0], np.array([0.0, 1.0, 0.0, 0.0], np.float32))


def test_generate_weak_labels_from_tracklet_boxes_without_gt_or_video():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        video_dir = root / "tracklets" / "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6"
        video_dir.mkdir(parents=True)
        pq = video_dir / "tracklets.parquet"
        pq.write_bytes(b"placeholder")
        out = root / "weak.csv"
        df = pd.DataFrame(
            {
                "video_key": ["vlincs_MS01_MC0001_MCAM04_2024-03-Tc6"] * 4,
                "tracklet_key": ["trk_a", "trk_a", "trk_b", "trk_b"],
                "frame_idx": [10, 20, 15, 30],
                "x1": [10.0, 12.0, 200.0, 210.0],
                "y1": [20.0, 22.0, 300.0, 310.0],
                "x2": [40.0, 42.0, 260.0, 270.0],
                "y2": [120.0, 122.0, 430.0, 440.0],
                "score": [0.8, 0.9, 0.4, 0.5],
                "reference_global_id": ["M0012", "M0012", "M0007", "M0007"],
            }
        )
        orig_read_parquet = pd.read_parquet
        pd.read_parquet = lambda path: df.copy()
        try:
            info = generate_weak_labels(
                root / "tracklets",
                out,
                cfg=WeakLabelGenerationConfig(include_crop_colors=False),
            )
        finally:
            pd.read_parquet = orig_read_parquet

        rows = list(csv.DictReader(out.open(encoding="utf-8")))

    assert info["rows"] == 2
    assert [row["tracklet_key"] for row in rows] == ["trk_a", "trk_b"]
    tokens = json.loads(rows[0]["weak_tokens"])
    assert "reference_global_id" not in tokens
    assert tokens["bbox_height"] in {"small", "medium", "large", "xlarge"}
    assert tokens["track_duration"] in {"short", "medium", "long", "very_long"}
    assert rows[0]["camera"] == "MCAM04"


def test_weak_tokens_from_tracklet_uses_only_boxes_frames_and_confidence():
    tokens = weak_tokens_from_tracklet(
        frames=[100, 110, 120],
        boxes=[[10, 20, 40, 120], [15, 22, 46, 124], [20, 25, 52, 128]],
        confs=[0.8, 0.7, 0.9],
    )

    assert tokens["motion_x"] == "right"
    assert tokens["det_conf"] in {"high", "very_high"}
    assert not any("global" in key or "gt" in key or "reference" in key for key in tokens)
