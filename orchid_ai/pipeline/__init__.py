"""Data pipelines: typed DAG execution over a run-scoped information graph.

A pipeline is declared in the ``pipelines:`` block of ``agents.yaml`` and runs
as a typed graph of jobs. This package holds the runtime pieces: the shape
compiler, the data graph, mapping, the run session and the LangGraph topology.

Importing this package is deliberately cheap: it has no eager imports, so
``import orchid_ai.pipeline`` never pulls a compiler, a graph runtime, a model
provider or a storage backend into the process. Import the submodule you need::

    from orchid_ai.pipeline.schema import compile_shapes

Submodules are added as the pipeline runtime is built; this module only
documents the package boundary.
"""

from __future__ import annotations

__all__: list[str] = []
