"""Utilities for loading and validating the saved road network."""

import logging


LOGGER = logging.getLogger(__name__)


def normalize_edge_lengths(graph):
    """Convert GraphML edge weights to numbers and discard invalid edges.

    GraphML can contain placeholder values such as ``-`` for optional edge
    attributes. Those edges cannot be used by NetworkX's weighted path search.
    Removing them lets the remaining valid road network stay routable.
    """
    invalid_edges = []
    if graph.is_multigraph():
        edges = list(graph.edges(keys=True, data=True))
        for origin, destination, key, attributes in edges:
            try:
                attributes["length"] = float(attributes["length"])
            except (KeyError, TypeError, ValueError):
                invalid_edges.append((origin, destination, key))
    else:
        edges = list(graph.edges(data=True))
        for origin, destination, attributes in edges:
            try:
                attributes["length"] = float(attributes["length"])
            except (KeyError, TypeError, ValueError):
                invalid_edges.append((origin, destination))

    if invalid_edges:
        graph.remove_edges_from(invalid_edges)
        LOGGER.warning(
            "Ignored %s road-network edge(s) with invalid length values.",
            len(invalid_edges),
        )
    return len(invalid_edges)
