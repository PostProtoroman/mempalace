"""Regression tests for:

1. ``candidate_strategy="union"`` becoming the authoritative default across
   ``search_memories``, the ``mempalace_search`` MCP tool, and the CLI
   search path (``searcher.search``) — one constant
   (``searcher.DEFAULT_CANDIDATE_STRATEGY``), not three independent
   defaults that can drift.
2. A corpus where a mechanical exact-token match (a hex commit id) embeds
   far from a natural-language query: found under the new ("union")
   default, missed under explicit ``candidate_strategy="vector"``.
3. ``search_degraded`` visibility on ``mempalace_status`` (MCP) and
   ``mempalace status`` (CLI) when the HNSW segment is quarantined/
   diverged — see docs/recovery/index-metadata-recovery.md.
"""

from pathlib import Path
from unittest.mock import patch

from _chroma_palace_helper import make_minimal_chroma_sqlite

from mempalace.palace import get_collection
from mempalace.searcher import DEFAULT_CANDIDATE_STRATEGY, search, search_memories


# ---------------------------------------------------------------------------
# 1. One authoritative constant, and it is "union".
# ---------------------------------------------------------------------------


def test_default_candidate_strategy_constant_is_union():
    assert DEFAULT_CANDIDATE_STRATEGY == "union"


def test_search_memories_default_param_is_the_shared_constant():
    import inspect

    sig = inspect.signature(search_memories)
    assert sig.parameters["candidate_strategy"].default == DEFAULT_CANDIDATE_STRATEGY


def test_cli_search_default_param_is_the_shared_constant():
    import inspect

    sig = inspect.signature(search)
    assert sig.parameters["candidate_strategy"].default == DEFAULT_CANDIDATE_STRATEGY


def test_mcp_tool_search_default_param_is_the_shared_constant():
    import inspect

    from mempalace.mcp_server import tool_search

    sig = inspect.signature(tool_search)
    assert sig.parameters["candidate_strategy"].default == DEFAULT_CANDIDATE_STRATEGY


# ---------------------------------------------------------------------------
# 2. Regression corpus: exact-token (mechanical) match embeds far from a
#    natural-language query. Union (new default) must find it; explicit
#    vector must be allowed to miss it — proving the test actually bites.
# ---------------------------------------------------------------------------

_HEX_ID = "a1b2c3d4e5f60718"

_LOG_QUERY = "Why did the nightly ingest job fail last night, what was the error?"


def _seed_mechanical_corpus(palace_path):
    """A dozen ordinary prose drawers about unrelated topics, plus D_LOG: a
    raw log line (timestamp + hex commit id + stack frame) that contains
    none of the query's natural-language vocabulary but does contain the
    exact hex id an operator would grep for. Embeddings cluster the prose
    drawers together and push the log line outside the top vector
    neighbors for a hex-id query, while BM25/FTS matches it strongly once
    the id is in the query — the top n_results*4 vector-only pool (the
    default before this fix) must miss it in a large-enough distractor set.
    """
    col = get_collection(palace_path, create=True)
    prose_topics = [
        "recipe for weeknight lentil soup with cumin and lemon",
        "notes from the villa construction site walkthrough on rebar spacing",
        "packing list for a weekend trip to the mountains",
        "summary of last quarter's padel coaching schedule",
        "reminder to renew the domain registration before it lapses",
        "draft reply to a vendor about invoice payment terms",
        "thoughts on which paint color to use for the guest room",
        "grocery list for the week including coffee and olive oil",
        "outline for a blog post about local-first software design",
        "notes on the best time of year to visit Bali for diving",
        "plan for reorganizing the garage storage shelves",
        "list of birthday gift ideas for a close friend",
        # Vector-distractors: none contains the actual hex commit id, but
        # each shares surface vocabulary with the query (traceback, error,
        # KeyError, dimensionality). Under the test suite's unweighted
        # bag-of-hashed-tokens embedding, that shared vocabulary makes each
        # of these closer in cosine distance than the padded, mostly-filler
        # log line below — exactly how a real embedding model would rank a
        # narrative discussion of an error above the log line that actually
        # recorded it. BM25's IDF still favors the log line once the rare
        # hex id is in the query.
        "a common Python traceback shows KeyError when a dict access fails",
        "debugging tips: read the traceback bottom-up to find the KeyError",
        "the numpy docs explain dimensionality and reshape traceback errors",
        "onboarding guide: what to do when you hit a confusing traceback error",
        "the tensor dimensionality mismatch traceback confused the new hire",
    ]
    ids = [f"D{i}" for i in range(len(prose_topics))] + ["D_LOG"]
    # Padded with a long run of unrelated filler tokens: the test embedding
    # function is an unweighted (no-IDF) bag-of-hashed-tokens vector, so a
    # long, mostly-irrelevant document dilutes the cosine similarity
    # contributed by the few tokens it shares with the query — while BM25's
    # IDF weighting still rewards the rare exact hex-id match regardless of
    # surrounding document length. This mirrors the real-world case
    # (mechanical log/diff text embeds as noise) without depending on a real
    # semantic embedding model in the test environment.
    _FILLER = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango "
        "uniform victor whiskey xray yankee zulu widget gadget doohickey "
        "gizmo thingamajig contraption apparatus mechanism device"
    )
    documents = list(prose_topics) + [
        f"2026-08-30T02:14:07Z ERROR ingest.worker commit={_HEX_ID} "
        f"segment=vector traceback: KeyError('dimensionality') at "
        f"local_persistent_hnsw.py:412 {_FILLER} {_FILLER} {_FILLER}"
    ]
    metadatas = [
        {"wing": "notes", "room": "misc", "source_file": f"note_{i}.md"}
        for i in range(len(prose_topics))
    ] + [{"wing": "ops", "room": "ingest", "source_file": "ingest.log"}]
    col.upsert(ids=ids, documents=documents, metadatas=metadatas)


class TestMechanicalTextRecall:
    _QUERY = f"{_HEX_ID} traceback KeyError dimensionality"

    def test_found_under_new_default(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_mechanical_corpus(palace)
        result = search_memories(self._QUERY, palace, n_results=5)
        sources = [h["source_file"] for h in result["results"]]
        assert "ingest.log" in sources, (
            f"exact hex-id match must surface under the default "
            f"(union) candidate_strategy; got {sources}"
        )

    def test_found_explicitly_under_union(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_mechanical_corpus(palace)
        result = search_memories(self._QUERY, palace, n_results=5, candidate_strategy="union")
        sources = [h["source_file"] for h in result["results"]]
        assert "ingest.log" in sources

    def test_missed_under_explicit_vector(self, tmp_path):
        """Proves the test bites: without the union widening, the vector-only
        pool (top n_results*4 nearest embeddings) never includes the log
        line, because it embeds far from a hex-id query relative to the
        prose docs clustered around "ingest"."""
        palace = str(tmp_path / "palace")
        _seed_mechanical_corpus(palace)
        result = search_memories(self._QUERY, palace, n_results=1, candidate_strategy="vector")
        sources = [h["source_file"] for h in result["results"]]
        assert "ingest.log" not in sources, (
            "expected explicit vector-only mode to MISS the mechanical "
            f"exact-token match (that's the bug this default fixes); got {sources}"
        )

    def test_cli_search_path_finds_it_under_default(self, tmp_path, capsys):
        """The CLI's own (non-search_memories) query path must also honor
        the new default — this is the "CLI default" leg of the fix."""
        palace = str(tmp_path / "palace")
        _seed_mechanical_corpus(palace)
        search(self._QUERY, palace, n_results=5)
        out = capsys.readouterr().out
        assert "ingest.log" in out, (
            f"CLI search path must surface the hex-id log line under the "
            f"union default; output was:\n{out}"
        )


# ---------------------------------------------------------------------------
# 3. search_degraded visibility on mempalace_status / `mempalace status`.
# ---------------------------------------------------------------------------


class TestMcpStatusSearchDegraded:
    def test_search_degraded_true_when_hnsw_diverged(self, monkeypatch, tmp_path):
        from mempalace import mcp_server
        from mempalace.config import MempalaceConfig

        palace_path = str(tmp_path / "palace")
        Path(palace_path).mkdir(parents=True, exist_ok=True)
        make_minimal_chroma_sqlite(palace_path)
        config = MempalaceConfig(palace_path=palace_path)
        monkeypatch.setattr(mcp_server, "_config", config)
        monkeypatch.setattr(
            mcp_server,
            "hnsw_capacity_status",
            lambda *_a, **_kw: {"diverged": True, "message": "capacity mismatch"},
        )

        result = mcp_server.tool_status()

        assert result["search_degraded"] is True
        assert result["search_degraded_reason"] == "capacity mismatch"
        # Existing field kept in sync — no breaking rename.
        assert result["vector_disabled"] is True
        assert result["vector_disabled_reason"] == "capacity mismatch"

    def test_search_degraded_false_when_healthy(
        self, monkeypatch, config, palace_path, kg, seeded_collection
    ):
        from mempalace import mcp_server
        from tests.test_mcp_server import _patch_mcp_server

        _patch_mcp_server(monkeypatch, config, kg)
        result = mcp_server.tool_status()

        assert result.get("search_degraded") is False


class TestCliStatusSearchDegraded:
    def test_cli_status_warns_on_diverged_index(self, tmp_path, capsys):
        from mempalace.miner import status

        palace_path = str(tmp_path / "palace")
        Path(palace_path).mkdir(parents=True, exist_ok=True)
        make_minimal_chroma_sqlite(palace_path)

        with patch(
            "mempalace.backends.chroma.hnsw_capacity_status",
            return_value={"diverged": True, "message": "test divergence, run repair"},
        ):
            status(palace_path)

        out = capsys.readouterr().out
        assert "search_degraded" in out
        assert "test divergence, run repair" in out

    def test_cli_status_silent_when_healthy(self, tmp_path, capsys):
        from mempalace.miner import status

        palace_path = str(tmp_path / "palace")
        Path(palace_path).mkdir(parents=True, exist_ok=True)
        make_minimal_chroma_sqlite(palace_path)

        with patch(
            "mempalace.backends.chroma.hnsw_capacity_status",
            return_value={"diverged": False, "message": "ok"},
        ):
            status(palace_path)

        out = capsys.readouterr().out
        assert "search_degraded" not in out
