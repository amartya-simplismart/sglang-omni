# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics collector for the sglang-omni router."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from prometheus_client import (
    CollectorRegistry,
    GCCollector,
    PlatformCollector,
    ProcessCollector,
)
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    Metric,
)

if TYPE_CHECKING:
    from sglang_omni_router.worker import Worker


class RouterMetricsCollector:
    """Custom Prometheus collector that exposes live router worker state."""

    def __init__(self, workers: list[Worker]) -> None:
        self._workers = workers

    def collect(self) -> Iterator[Metric]:
        workers = self._workers

        pool_counts = GaugeMetricFamily(
            "sglang_omni_router_workers",
            "Number of router workers by state",
            labels=["state"],
        )
        pool_counts.add_metric(
            ["healthy"], sum(1 for w in workers if w.state == "healthy")
        )
        pool_counts.add_metric(
            ["dead"], sum(1 for w in workers if w.state == "dead")
        )
        pool_counts.add_metric(
            ["unhealthy"], sum(1 for w in workers if w.state == "unhealthy")
        )
        pool_counts.add_metric(
            ["unknown"], sum(1 for w in workers if w.state == "unknown")
        )
        pool_counts.add_metric(
            ["disabled"], sum(1 for w in workers if w.disabled)
        )
        pool_counts.add_metric(
            ["routable"], sum(1 for w in workers if w.is_routable)
        )
        yield pool_counts

        active = GaugeMetricFamily(
            "sglang_omni_router_active_requests",
            "Active in-flight requests per worker",
            labels=["worker"],
        )
        routed = CounterMetricFamily(
            "sglang_omni_router_routed_requests_total",
            "Total requests routed per worker",
            labels=["worker"],
        )
        succeeded = CounterMetricFamily(
            "sglang_omni_router_successful_requests_total",
            "Total successful requests per worker",
            labels=["worker"],
        )
        failed = CounterMetricFamily(
            "sglang_omni_router_failed_requests_total",
            "Total failed requests per worker",
            labels=["worker"],
        )

        for w in workers:
            label = [w.display_id]
            active.add_metric(label, w.active_requests)
            routed.add_metric(label, w.routed_requests)
            succeeded.add_metric(label, w.successful_requests)
            failed.add_metric(label, w.failed_requests)

        yield active
        yield routed
        yield succeeded
        yield failed


def build_router_metrics_registry(workers: list[Worker]) -> CollectorRegistry:
    registry = CollectorRegistry(auto_describe=True)
    GCCollector(registry=registry)
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    registry.register(RouterMetricsCollector(workers))
    return registry
