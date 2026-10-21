"""OKF knowledge graph built from cross-links between concepts.

Extracts the directed link graph from an OKF bundle: every markdown
link from concept A to concept B becomes a directed edge A -> B.
Provides neighbor lookups, backlink queries, and subgraph extraction
for graph-aware retrieval.

Example:
    .. code-block:: python

        from langchain_okf import OKFKnowledgeGraph

        graph = OKFKnowledgeGraph.from_bundle("./bundles/ga4/")

        graph.neighbors("tables/orders")
        # ["tables/customers", "datasets/sales"]

        graph.backlinks("tables/customers")
        # ["tables/orders"]

        graph.subgraph("tables/orders", depth=2)
        # All concept IDs reachable within 2 hops
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from langchain_okf._parser import OKFConcept, parse_bundle


class OKFKnowledgeGraph:
    """Directed graph of OKF concept relationships.

    Nodes are concept IDs (e.g. ``tables/orders``). Edges are derived
    from markdown cross-links in concept bodies.

    Args:
        concepts: List of parsed ``OKFConcept`` objects.
    """

    def __init__(self, concepts: list[OKFConcept]) -> None:
        self._concepts: dict[str, OKFConcept] = {}
        self._concept_id_set: set[str] = set()
        self._forward: dict[str, list[str]] = defaultdict(list)
        self._backward: dict[str, list[str]] = defaultdict(list)
        self._types: dict[str, list[str]] = defaultdict(list)
        self._tags: dict[str, list[str]] = defaultdict(list)

        for concept in concepts:
            self._concepts[concept.concept_id] = concept
            self._concept_id_set.add(concept.concept_id)
            self._types[concept.type].append(concept.concept_id)
            for tag in concept.tags:
                self._tags[tag].append(concept.concept_id)

        for concept in concepts:
            for target in concept.cross_links():
                if target in self._concepts:
                    self._forward[concept.concept_id].append(target)
                    self._backward[target].append(concept.concept_id)

    @classmethod
    def from_bundle(cls, bundle_path: str | Path) -> OKFKnowledgeGraph:
        """Build a knowledge graph from an OKF bundle directory.

        Args:
            bundle_path: Path to the OKF bundle root.

        Returns:
            A populated ``OKFKnowledgeGraph``.
        """
        return cls(parse_bundle(Path(bundle_path)))

    @property
    def concept_ids(self) -> list[str]:
        """All concept IDs in the graph."""
        return list(self._concepts.keys())

    @property
    def types(self) -> list[str]:
        """All distinct concept types in the graph."""
        return list(self._types.keys())

    def get_concept(self, concept_id: str) -> OKFConcept | None:
        """Look up a concept by ID."""
        return self._concepts.get(concept_id)

    def neighbors(self, concept_id: str) -> list[str]:
        """Concept IDs that this concept links to (outgoing edges)."""
        return list(self._forward.get(concept_id, []))

    def backlinks(self, concept_id: str) -> list[str]:
        """Concept IDs that link to this concept (incoming edges)."""
        return list(self._backward.get(concept_id, []))

    def concepts_by_type(self, type_name: str) -> list[str]:
        """All concept IDs matching a given type."""
        return list(self._types.get(type_name, []))

    def concepts_by_tag(self, tag: str) -> list[str]:
        """All concept IDs carrying a given tag."""
        return list(self._tags.get(tag, []))

    def subgraph(self, concept_id: str, *, depth: int = 1) -> set[str]:
        """Collect concept IDs reachable within ``depth`` hops.

        Follows both forward and backward links. The starting concept
        is always included.

        Args:
            concept_id: Starting concept ID.
            depth: Maximum number of link hops to follow.

        Returns:
            Set of reachable concept IDs including the start.
        """
        visited: set[str] = set()
        frontier = {concept_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for cid in frontier:
                if cid in visited or cid not in self._concepts:
                    continue
                visited.add(cid)
                next_frontier.update(self._forward.get(cid, []))
                next_frontier.update(self._backward.get(cid, []))
            frontier = next_frontier - visited
        visited.update(frontier & self._concept_id_set)
        return visited

    def edge_count(self) -> int:
        """Total number of directed edges in the graph."""
        return sum(len(targets) for targets in self._forward.values())

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        type_counts = {t: len(ids) for t, ids in self._types.items()}
        lines = [
            f"OKF Knowledge Graph: {len(self._concepts)} concepts, "
            f"{self.edge_count()} cross-links",
            f"Types: {type_counts}",
        ]
        return "\n".join(lines)
