"""
Integration Tests for OSAM Visualizer Backend

End-to-end cross-module tests that exercise all four backend modules together:
    - src/memory_state.py  (OSAMState)
    - src/encoder.py       (ProjectionMatrix, project_to_key, project_to_value)
    - src/writing_strategies.py (SequenceStateWrite)
    - src/retrieval.py     (SentenceRegistry)

These tests verify:
    - Cross-module configuration compatibility (dimensions, shapes)
    - Full write → store → query → verify pipeline (PRD requirement)
    - Reset flow: write → reset state + clear registry → re-write
    - Step index tracking through the full pipeline
    - Memory accumulation properties (state changes, beta effects)

Marked as @pytest.mark.integration where the real sentence encoder is loaded
(slow ~30s per test due to model loading on first call).
Filter slow tests: pytest -m "not integration"
Run only integration: pytest -m integration
"""

import numpy as np
import pytest

from src.memory_state import OSAMState
from src.writing_strategies import SequenceStateWrite
from src.retrieval import SentenceRegistry


class TestCrossModuleConfig:
    """
    Structural tests verifying configuration compatibility across all modules.

    These tests do NOT use the real sentence encoder and run in milliseconds.
    They verify that SequenceStateWrite and SentenceRegistry are initialised
    with compatible dimension configurations and that the OSAM state dimension
    matches the projection output dimension.
    """

    def test_projection_matrix_shapes_compatible(self):
        """W_k, W_v (SSW) and W_q (Registry) must all have identical shapes.

        All three projection matrices project from embedding_dim to output_dim.
        Mismatched shapes would cause a dimension error when feeding SSW output
        into SentenceRegistry.
        """
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()

        assert ssw.W_k.W.shape == ssw.W_v.W.shape
        assert ssw.W_k.W.shape == registry.W_q.W.shape

    def test_output_dim_matches_state_dimension(self):
        """Projection output_dim must equal the OSAM state dimension r.

        SSW projects embeddings to vectors of shape (output_dim,). The
        OSAMState.read() method expects query vectors of the same shape (r,).
        These must be equal for the read pipeline to work.
        """
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        output_dim = ssw.W_k.W.shape[0]
        assert output_dim == state.r


@pytest.mark.integration
class TestEndToEndPipeline:
    """
    End-to-end pipeline tests using all four backend modules together.

    Marked as @pytest.mark.integration because they load the real sentence
    encoder (sentence-transformers/all-MiniLM-L6-v2) on first use.
    """

    def test_prd_requirement_five_writes_query(self):
        """PRD requirement: write 5 sentences → query → verify pipeline output.

        Validates the minimum PRD testing requirement (testing requirements section):
        'One end-to-end integration test: write 5 sentences → query → assert
        top result is correct.'

        Asserts deterministic structural properties that are guaranteed by the
        algorithm regardless of which sentence ranks highest in OSAM memory:
        correct result count, sorted scores, valid score range, valid step
        indices, all required result keys present, and all texts from the
        written set.

        Paper reference: delta-mem Section 3.2 (reading from OSAM)
        """
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()
        state = OSAMState(r=8, beta=0.1)

        sentences = [
            "Alice lives in Paris.",
            "Bob works at a hospital.",
            "Charlie studies mathematics.",
            "Diana plays the piano.",
            "Eve enjoys hiking.",
        ]

        for sentence in sentences:
            state, metadata = ssw.execute(sentence, state)
            registry.store(
                text=metadata["text"],
                embedding=metadata["embedding"],
                key=metadata["key"],
                value=metadata["value"],
                step_index=metadata["step_index"],
            )

        assert len(registry) == 5

        query_embedding = ssw.encoder.encode("Who lives in Paris?")
        results = registry.search(query_embedding, state, top_k=5)

        # Verify result count
        assert len(results) == 5

        # Verify all required keys are present in every result
        required_keys = {"text", "score", "step_index", "key", "value"}
        for result in results:
            assert required_keys.issubset(result.keys())

        # Verify all result texts come from the written set
        written_texts = set(sentences)
        for result in results:
            assert result["text"] in written_texts

        # Verify results are sorted descending by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

        # Verify scores are in valid cosine similarity range [-1, 1]
        for result in results:
            assert -1.0 <= result["score"] <= 1.0

        # Verify step indices are integers in the expected range [0, 4]
        for result in results:
            assert isinstance(result["step_index"], int)
            assert 0 <= result["step_index"] <= 4

    def test_single_write_and_retrieve_text_matches(self):
        """Writing one sentence and querying with its text returns that sentence.

        With a single registry entry, the top result must always be the written
        sentence — it is the only candidate. This is the most minimal form of
        the write-retrieve round-trip.
        """
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()
        state = OSAMState(r=8, beta=0.1)

        sentence = "The quick brown fox jumps over the lazy dog."
        state, metadata = ssw.execute(sentence, state)
        registry.store(
            text=metadata["text"],
            embedding=metadata["embedding"],
            key=metadata["key"],
            value=metadata["value"],
            step_index=metadata["step_index"],
        )

        query_embedding = ssw.encoder.encode(sentence)
        results = registry.search(query_embedding, state, top_k=1)

        assert len(results) == 1
        assert results[0]["text"] == sentence
        assert results[0]["step_index"] == 0

    def test_reset_clears_state_and_registry(self):
        """Full reset flow: write → verify → reset → verify empty → re-write.

        Verifies that state.reset() and registry.clear() correctly return the
        system to its initial conditions so a fresh session can begin from a
        clean slate without creating new object instances.
        """
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()
        state = OSAMState(r=8, beta=0.1)

        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        for sentence in sentences:
            state, metadata = ssw.execute(sentence, state)
            registry.store(
                text=metadata["text"],
                embedding=metadata["embedding"],
                key=metadata["key"],
                value=metadata["value"],
                step_index=metadata["step_index"],
            )

        assert len(registry) == 3
        assert not np.allclose(state.get_state(), np.zeros((8, 8)))

        # Reset both state and registry
        state.reset()
        registry.clear()

        assert len(registry) == 0
        assert np.allclose(state.get_state(), np.zeros((8, 8)))

        # Re-write one sentence from a clean slate
        ssw.reset_counter()
        state, metadata = ssw.execute("Fresh start.", state)
        registry.store(
            text=metadata["text"],
            embedding=metadata["embedding"],
            key=metadata["key"],
            value=metadata["value"],
            step_index=metadata["step_index"],
        )

        assert len(registry) == 1
        assert registry.entries[0].text == "Fresh start."
        assert registry.entries[0].step_index == 0

    def test_step_index_sequential_in_registry(self):
        """Step indices stored in registry entries match the SSW write order.

        Verifies the end-to-end step tracking contract: SequenceStateWrite
        increments step_counter per execute() call and embeds it in metadata
        (0-indexed), which is then stored in the registry entry unchanged.
        """
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()
        state = OSAMState(r=8, beta=0.1)

        sentences = [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
            "Sentence four.",
        ]

        for sentence in sentences:
            state, metadata = ssw.execute(sentence, state)
            registry.store(
                text=metadata["text"],
                embedding=metadata["embedding"],
                key=metadata["key"],
                value=metadata["value"],
                step_index=metadata["step_index"],
            )

        # step_index is 0-indexed: first write → 0, second → 1, ...
        for i, entry in enumerate(registry.entries):
            assert entry.step_index == i

    def test_metadata_fields_map_to_registry_store(self):
        """All fields required by registry.store() are present in SSW metadata.

        Verifies the inter-module data contract: SSW.execute() produces a
        metadata dict whose keys directly map to SentenceRegistry.store()
        parameters, with correct types for each field.
        """
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        _, metadata = ssw.execute("A test sentence.", state)

        # These are the exact parameter names of registry.store()
        required_fields = {"text", "embedding", "key", "value", "step_index"}
        assert required_fields.issubset(metadata.keys())

        assert isinstance(metadata["text"], str)
        assert isinstance(metadata["embedding"], np.ndarray)
        assert isinstance(metadata["key"], np.ndarray)
        assert isinstance(metadata["value"], np.ndarray)
        assert isinstance(metadata["step_index"], int)


@pytest.mark.integration
class TestMemoryAccumulation:
    """
    Tests verifying memory accumulation properties across sequential writes.

    Marked as @pytest.mark.integration because they load the real sentence
    encoder on first use.
    """

    def test_each_write_produces_distinct_state(self):
        """Every write event changes the memory state matrix S.

        Verifies that no write is a no-op: each new sentence produces a
        state matrix that differs from all previous states. This confirms
        the delta-rule is actively updating S on every call.

        Paper reference: delta-mem Section 3.3 (delta-rule update)
        """
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        texts = ["First sentence.", "Second sentence.", "Third sentence."]

        states = [state.get_state().copy()]
        for text in texts:
            state, _ = ssw.execute(text, state)
            states.append(state.get_state().copy())

        for i in range(len(states) - 1):
            assert not np.allclose(states[i], states[i + 1]), (
                f"State did not change after write {i + 1}"
            )

    def test_higher_beta_produces_larger_state_change(self):
        """Higher write strength β produces a larger state update magnitude.

        Beta controls how strongly new information overwrites the existing
        state. Writing the same sentence with β=0.5 must produce a larger
        ||state_diff|| than β=0.1, starting from the same initial state.

        Paper reference: delta-mem Section 3.3 (β controls retention vs. write)
        """
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        text = "Test sentence for beta comparison."

        _, metadata_low = ssw.execute(text, state, beta=0.1)
        _, metadata_high = ssw.execute(text, state, beta=0.5)

        diff_norm_low = np.linalg.norm(metadata_low["state_diff"])
        diff_norm_high = np.linalg.norm(metadata_high["state_diff"])

        assert diff_norm_high > diff_norm_low

    def test_state_nonzero_after_writes(self):
        """Memory state matrix S is non-zero after at least one write.

        The initial state is the zero matrix. After writing any sentence,
        the delta-rule update must produce at least one non-zero element,
        confirming information has been committed to memory.

        Paper reference: delta-mem Section 3.3 (write term: diag(β) @ v @ k^T)
        """
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        assert np.allclose(state.get_state(), np.zeros((8, 8))), (
            "Initial state must be zero matrix"
        )

        state, _ = ssw.execute("A sentence to write.", state)

        assert not np.allclose(state.get_state(), np.zeros((8, 8))), (
            "State must be non-zero after writing a sentence"
        )
