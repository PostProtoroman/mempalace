"""Tests for mempalace.dedup — near-duplicate drawer detection and removal."""

from unittest.mock import MagicMock, patch


from mempalace import dedup


# ── get_source_groups ─────────────────────────────────────────────────


def test_get_source_groups_aborts_on_hnsw_divergence():
    """count() on a diverged HNSW segment can hard-crash the process
    (#1222); get_source_groups must not call it when hnsw_capacity_status
    reports divergence for the caller-supplied palace_path (#92). Omitting
    palace_path (as every existing test above does) skips the check
    entirely -- this only fires when a real caller passes one."""
    col = MagicMock()
    with patch(
        "mempalace.backends.chroma.hnsw_capacity_status",
        return_value={"diverged": True, "message": "test divergence"},
    ):
        groups = dedup.get_source_groups(col, palace_path="/fake/palace")
    assert groups == {}
    col.count.assert_not_called()


def test_get_source_groups_basic():
    col = MagicMock()
    col.count.return_value = 5
    col.get.side_effect = [
        {
            "ids": ["d1", "d2", "d3", "d4", "d5"],
            "metadatas": [
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
            ],
        },
        {"ids": []},
    ]
    groups = dedup.get_source_groups(col, min_count=5)
    assert "a.txt" in groups
    assert len(groups["a.txt"]) == 5


def test_get_source_groups_below_min():
    col = MagicMock()
    col.count.return_value = 2
    col.get.side_effect = [
        {
            "ids": ["d1", "d2"],
            "metadatas": [
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
            ],
        },
        {"ids": []},
    ]
    groups = dedup.get_source_groups(col, min_count=5)
    assert len(groups) == 0


def test_get_source_groups_source_filter():
    col = MagicMock()
    col.count.return_value = 6
    col.get.side_effect = [
        {
            "ids": ["d1", "d2", "d3", "d4", "d5", "d6"],
            "metadatas": [
                {"source_file": "project_a.txt"},
                {"source_file": "project_a.txt"},
                {"source_file": "project_a.txt"},
                {"source_file": "project_a.txt"},
                {"source_file": "project_a.txt"},
                {"source_file": "other.txt"},
            ],
        },
        {"ids": []},
    ]
    groups = dedup.get_source_groups(col, min_count=5, source_pattern="project_a")
    assert "project_a.txt" in groups
    assert "other.txt" not in groups


def test_get_source_groups_wing_filter():
    col = MagicMock()
    col.count.return_value = 5
    col.get.side_effect = [
        {
            "ids": ["d1", "d2", "d3", "d4", "d5"],
            "metadatas": [
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
            ],
        },
        {"ids": []},
    ]
    dedup.get_source_groups(col, min_count=5, wing="my_wing")
    # Verify where filter was passed
    first_call = col.get.call_args_list[0]
    assert first_call.kwargs.get("where") == {"wing": "my_wing"}


def test_get_source_groups_missing_source_file():
    col = MagicMock()
    col.count.return_value = 5
    col.get.side_effect = [
        {
            "ids": ["d1", "d2", "d3", "d4", "d5"],
            "metadatas": [{}, {}, {}, {}, {}],
        },
        {"ids": []},
    ]
    groups = dedup.get_source_groups(col, min_count=5)
    assert "unknown" in groups


# ── dedup_source_group ────────────────────────────────────────────────


def test_dedup_source_group_all_unique():
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": ["long document one content here", "different document two here"],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    col.query.return_value = {
        "ids": [["d1"]],
        "distances": [[0.8]],  # far apart = unique
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert len(kept) == 2
    assert len(deleted) == 0


def test_dedup_source_group_with_near_duplicate():
    """Same length but different text, reported as near (below threshold) by
    the similarity search -- a real mining artifact, not an exact repeat."""
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": [
            "long document content that is fairly long!",
            "long document content that is fairly long?",
        ],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    col.query.return_value = {
        "ids": [["d1"]],
        "distances": [[0.05]],  # very close = duplicate
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert len(kept) == 1
    assert len(deleted) == 1


def test_dedup_source_group_exact_duplicate_content_is_protected():
    """Byte-identical content is a deliberate verbatim repeat (per
    ids.py's filed_at-based id scheme) -- the module must NOT delete it,
    even though it is reported as maximally near (distance 0)."""
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": [
            "long document content that is fairly long",
            "long document content that is fairly long",
        ],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    col.query.return_value = {
        "ids": [["d1"]],
        "distances": [[0.0]],
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert deleted == []
    assert set(kept) == {"d1", "d2"}


def test_dedup_source_group_short_docs_survive_when_unique():
    """Short-but-unique drawers are real content, not a mining artifact --
    the module no longer deletes purely for being under 20 chars."""
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": ["long enough document to keep in the palace", "tiny"],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    col.query.return_value = {
        "ids": [["d1"]],
        "distances": [[0.8]],  # far apart = unique
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert "d2" in kept
    assert deleted == []


def test_dedup_source_group_empty_doc_survives():
    """A drawer with no document text has nothing to compare -- it is kept,
    never auto-deleted."""
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": ["real document content here that is long enough", None],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert "d2" in kept
    assert deleted == []


def test_dedup_source_group_exact_repeat_survives():
    """Two drawers with byte-identical content are a deliberate verbatim
    repeat (distinguished from a mining artifact by filed_at, per
    ids.py's make_exchange_drawer_id) -- never deleted, even though the
    similarity search reports them as near (distance 0)."""
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": [
            "the exact same words said twice on two different days",
            "the exact same words said twice on two different days",
        ],
        "metadatas": [
            {"wing": "a", "filed_at": "2026-01-01T00:00:00"},
            {"wing": "a", "filed_at": "2026-06-01T00:00:00"},
        ],
    }
    col.query.return_value = {
        "ids": [["d1"]],
        "distances": [[0.0]],  # identical text -> distance 0
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert deleted == []
    assert set(kept) == {"d1", "d2"}


def test_dedup_source_group_near_dup_not_exact_still_deleted():
    """A near-duplicate that is NOT an exact text match (a real mining
    artifact -- same content, slightly different chunk boundary) is still
    eligible for deletion."""
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": [
            "long document content that is fairly long and rich!",
            "long document content that is fairly long and rich?",
        ],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    col.query.return_value = {
        "ids": [["d1"]],
        "distances": [[0.05]],
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert "d2" in deleted


def test_dedup_source_group_live_deletes():
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": ["long document content here enough!", "long document content here enough?"],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    col.query.return_value = {
        "ids": [["d1"]],
        "distances": [[0.05]],
    }
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=False)
    col.delete.assert_called_once()


def test_dedup_source_group_query_failure_keeps():
    col = MagicMock()
    col.get.return_value = {
        "ids": ["d1", "d2"],
        "documents": [
            "long document one content here enough",
            "long document two content here enough",
        ],
        "metadatas": [{"wing": "a"}, {"wing": "a"}],
    }
    col.query.side_effect = Exception("query failed")
    kept, deleted = dedup.dedup_source_group(col, ["d1", "d2"], threshold=0.15, dry_run=True)
    assert len(kept) == 2  # both kept on error


# ── show_stats ────────────────────────────────────────────────────────


def _install_mock_collection(mock_get_collection, collection):
    mock_get_collection.return_value = collection
    return collection


@patch("mempalace.dedup.get_collection")
def test_show_stats(mock_get_collection, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 5
    mock_col.get.side_effect = [
        {
            "ids": ["d1", "d2", "d3", "d4", "d5"],
            "metadatas": [
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
                {"source_file": "a.txt"},
            ],
        },
        {"ids": []},
    ]
    _install_mock_collection(mock_get_collection, mock_col)

    dedup.show_stats(palace_path=str(tmp_path))  # should not raise


# ── dedup_palace ──────────────────────────────────────────────────────


@patch("mempalace.dedup.dedup_source_group")
@patch("mempalace.dedup.get_source_groups")
@patch("mempalace.dedup.get_collection")
def test_dedup_palace_dry_run(mock_get_collection, mock_groups, mock_dedup_group, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 10
    _install_mock_collection(mock_get_collection, mock_col)

    mock_groups.return_value = {"a.txt": ["d1", "d2", "d3", "d4", "d5"]}
    mock_dedup_group.return_value = (["d1", "d2", "d3"], ["d4", "d5"])

    dedup.dedup_palace(palace_path=str(tmp_path), dry_run=True)
    mock_dedup_group.assert_called_once()


@patch("mempalace.dedup.get_source_groups")
@patch("mempalace.dedup.get_collection")
def test_dedup_palace_aborts_on_hnsw_divergence(mock_get_collection, mock_groups, tmp_path):
    """dedup_palace's OWN count() print (a few lines before it calls
    get_source_groups) is a separate call site from #92 -- count() on a
    diverged segment can hard-crash the process, so this print must never
    be reached either when hnsw_capacity_status reports divergence."""
    mock_col = MagicMock()
    mock_col.count.side_effect = AssertionError("count() must not be called when diverged")
    _install_mock_collection(mock_get_collection, mock_col)

    with patch(
        "mempalace.backends.chroma.hnsw_capacity_status",
        return_value={"diverged": True, "message": "test divergence"},
    ):
        dedup.dedup_palace(palace_path=str(tmp_path), dry_run=True)

    mock_groups.assert_not_called()


@patch("mempalace.dedup.dedup_source_group")
@patch("mempalace.dedup.get_source_groups")
@patch("mempalace.dedup.get_collection")
def test_dedup_palace_with_wing(mock_get_collection, mock_groups, mock_dedup_group, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 10
    _install_mock_collection(mock_get_collection, mock_col)

    mock_groups.return_value = {}
    dedup.dedup_palace(palace_path=str(tmp_path), wing="test_wing", dry_run=True)
    mock_groups.assert_called_once_with(
        mock_col, 5, None, wing="test_wing", palace_path=str(tmp_path)
    )


@patch("mempalace.dedup.dedup_source_group")
@patch("mempalace.dedup.get_source_groups")
@patch("mempalace.dedup.get_collection")
def test_dedup_palace_no_groups(mock_get_collection, mock_groups, mock_dedup_group, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 3
    _install_mock_collection(mock_get_collection, mock_col)

    mock_groups.return_value = {}
    dedup.dedup_palace(palace_path=str(tmp_path), dry_run=True)
    mock_dedup_group.assert_not_called()
