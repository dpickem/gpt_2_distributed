"""
This file implements a tracker for the training of a GPT-2 model.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import psutil
import time
from typing import DefaultDict, Deque, Optional, Any, Callable, Dict

import torch
import torch.distributed as dist
from torch.utils import tensorboard as tb


def _is_primary() -> bool:
    return (
        dist.get_rank() == 0 if dist.is_available() and dist.is_initialized() else True
    )


def _all_reduce_scalar(x: float | torch.Tensor) -> float:
    t = torch.as_tensor(
        x, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()

    return t.detach().item()


class ReductionStrategy(Enum):
    """Strategies for reducing metric values."""

    AVERAGE = "average"
    SUM = "sum"
    CURRENT = "current"  # Use latest value
    MAX = "max"
    MIN = "min"


@dataclass
class MetricDefinition:
    """Complete definition of a metric's behavior.

    Attributes:
        name: The name of the metric.
        frequency: The frequency at which to collect the metric (in steps).
        reduction: The strategy to use for reducing the metric values.
        tb_prefix: The prefix to use for the metric in TensorBoard.
        cli_format: The format to use for the metric in the CLI.
        processor: A function to process the metric value.
        collector: A function to collect the metric value.
        distributed: Whether to apply distributed reduction.
    """

    name: str
    frequency: int
    reduction: ReductionStrategy
    tb_prefix: str = ""
    cli_format: str = "{name}:{value:.4g}"
    processor: Optional[Callable[[Any], float]] = None
    collector: Optional[Callable[[int], Dict[str, float]]] = None
    distributed: bool = True


class MetricRegistry:
    """Global registry for metric definitions."""

    def __init__(self):
        """Initialize the MetricRegistry.

        Args:
            metrics: A dictionary of metric definitions.
            collectors: A dictionary of collector functions.
        """
        self._metrics: Dict[str, MetricDefinition] = {}
        self._collectors: Dict[str, Callable] = {}

    def register(self, definition: MetricDefinition) -> Callable:
        """Decorator to register a metric definition."""

        def decorator(func: Callable) -> Callable:
            self._metrics[definition.name] = definition
            if definition.collector:
                self._collectors[definition.name] = definition.collector
            return func

        return decorator

    def metric(
        self,
        name: str,
        frequency: int = 1,
        reduction: ReductionStrategy = ReductionStrategy.AVERAGE,
        tb_prefix: str = "",
        cli_format: str = "{name}:{value:.4g}",
        processor: Optional[Callable[[Any], float]] = None,
        collector: Optional[Callable[[int], Dict[str, float]]] = None,
        distributed: bool = True,
    ) -> Callable:
        """Decorator for registering metrics with full configuration."""
        definition = MetricDefinition(
            name=name,
            frequency=frequency,
            reduction=reduction,
            tb_prefix=tb_prefix,
            cli_format=cli_format,
            processor=processor,
            collector=collector,
            distributed=distributed,
        )
        return self.register(definition)

    def get_metric(self, name: str) -> Optional[MetricDefinition]:
        """Get metric definition by name."""
        return self._metrics.get(name)

    def get_metrics_by_frequency(self, frequency: int) -> Dict[str, MetricDefinition]:
        """Get all metrics that should be collected at this frequency."""
        return {
            name: metric
            for name, metric in self._metrics.items()
            if metric.frequency == frequency
        }

    def get_collector(self, name: str) -> Optional[Callable]:
        """Get collector function for a metric."""
        return self._collectors.get(name)


# Global metric registry
METRIC_REGISTRY = MetricRegistry()


# Metric Definitions with Decorators
@METRIC_REGISTRY.metric(
    name="loss",
    frequency=1,
    reduction=ReductionStrategy.AVERAGE,
    tb_prefix="train/",
    cli_format="loss:{value:.4g}",
    distributed=True,
)
def loss_metric():
    """Training loss - averaged across processes."""
    pass


@METRIC_REGISTRY.metric(
    name="lr",
    frequency=1,
    reduction=ReductionStrategy.CURRENT,
    tb_prefix="train/",
    cli_format="lr:{value:.2e}",
    distributed=False,  # LR is same across processes
)
def lr_metric():
    """Learning rate - current value."""
    pass


@METRIC_REGISTRY.metric(
    name="grad_norm",
    frequency=1,
    reduction=ReductionStrategy.AVERAGE,
    tb_prefix="train/",
    cli_format="grad_norm:{value:.4g}",
    distributed=True,
)
def grad_norm_metric():
    """Gradient norm - averaged across processes."""
    pass


@METRIC_REGISTRY.metric(
    name="epoch",
    frequency=1,
    reduction=ReductionStrategy.CURRENT,
    tb_prefix="train/",
    cli_format="epoch:{value:d}",
    processor=lambda x: int(x),
    distributed=False,
)
def epoch_metric():
    """Epoch number - discrete count."""
    pass


@METRIC_REGISTRY.metric(
    name="batch",
    frequency=1,
    reduction=ReductionStrategy.CURRENT,
    tb_prefix="train/",
    cli_format="batch:{value:d}",
    processor=lambda x: int(x),
    distributed=False,
)
def batch_metric():
    """Batch number - discrete count."""
    pass


def _collect_performance_metrics(step: int, tracker: StatsTracker) -> Dict[str, float]:
    """Collector for performance metrics.

    Args:
        step: The current step.
        tracker: The StatsTracker object.

    Returns:
        A dictionary of performance metrics (per-worker values).
    """
    metrics = {}
    now = time.time()

    # Tokens/sec calculation - per worker
    dt = now - tracker.last_tok_time
    if dt > 0:
        metrics["tokens_per_second"] = tracker.tok_accum / dt

    # Total tokens processed - per worker
    metrics["total_tokens"] = float(tracker.total_tokens)

    # Epoch time (if we're tracking epochs) - same across workers
    if tracker.epoch_start_time is not None:
        metrics["epoch_time"] = now - tracker.epoch_start_time

    return metrics


@METRIC_REGISTRY.metric(
    name="tokens_per_second",
    frequency=1,
    reduction=ReductionStrategy.SUM,
    tb_prefix="perf/",
    cli_format="tok/s:{value:,.0f}",
    collector=_collect_performance_metrics,
    distributed=True,
)
def tokens_per_second_metric():
    """Tokens per second - total system throughput."""
    pass


@METRIC_REGISTRY.metric(
    name="total_tokens",
    frequency=1,
    reduction=ReductionStrategy.SUM,
    tb_prefix="perf/",
    cli_format="total_tok:{value:,.0f}",
    distributed=True,
)
def total_tokens_metric():
    """Total tokens processed - total system count."""
    pass


@METRIC_REGISTRY.metric(
    name="epoch_time",
    frequency=1,
    reduction=ReductionStrategy.CURRENT,
    tb_prefix="perf/",
    cli_format="epoch_time:{value:.2f}s",
    distributed=False,
)
def epoch_time_metric():
    """Time elapsed in current epoch."""
    pass


def _collect_memory_metrics(step: int, tracker: StatsTracker) -> Dict[str, float]:
    """Collector for memory metrics.

    Args:
        step: The current step.
        tracker: The StatsTracker object.

    Returns:
        A dictionary of memory metrics (per-worker values).
    """
    metrics = {}

    if torch.cuda.is_available():
        metrics["gpu_alloc_gb"] = torch.cuda.memory_allocated() / (1024**3)
        metrics["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        metrics["gpu_max_alloc_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

        # GPU utilization percentage
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        metrics["gpu_utilization_pct"] = (metrics["gpu_alloc_gb"] / gpu_total) * 100

    metrics["cpu_mb"] = psutil.Process().memory_info().rss / 2**20
    return metrics


@METRIC_REGISTRY.metric(
    name="gpu_alloc_gb",
    frequency=20,
    reduction=ReductionStrategy.AVERAGE,
    tb_prefix="mem/",
    cli_format="GPU:{value:.2f}GB",
    collector=_collect_memory_metrics,
    distributed=True,
)
def gpu_alloc_metric():
    """GPU memory allocated - average per GPU."""
    pass


@METRIC_REGISTRY.metric(
    name="gpu_reserved_gb",
    frequency=20,
    reduction=ReductionStrategy.AVERAGE,
    tb_prefix="mem/",
    distributed=True,
)
def gpu_reserved_metric():
    """GPU memory reserved - average per GPU."""
    pass


@METRIC_REGISTRY.metric(
    name="gpu_max_alloc_gb",
    frequency=20,
    reduction=ReductionStrategy.MAX,
    tb_prefix="mem/",
    cli_format="max:{value:.2f}GB",
    distributed=True,
)
def gpu_max_alloc_metric():
    """GPU memory max allocated - peak across all GPUs."""
    pass


@METRIC_REGISTRY.metric(
    name="gpu_utilization_pct",
    frequency=20,
    reduction=ReductionStrategy.AVERAGE,
    tb_prefix="mem/",
    cli_format="util:{value:.1f}%",
    distributed=True,
)
def gpu_util_metric():
    """GPU utilization percentage - average across GPUs."""
    pass


@METRIC_REGISTRY.metric(
    name="cpu_mb",
    frequency=20,
    reduction=ReductionStrategy.SUM,  # Total CPU memory across all processes
    tb_prefix="mem/",
    cli_format="CPU:{value:.0f}MB",
    distributed=True,  # Aggregate across workers/processes
)
def cpu_metric():
    """CPU memory usage - total across all processes."""
    pass


class StatsTracker:
    """
    Metric tracking system with decorator-based registry.

    Features:
        - Decorator-based metric registration
        - Configurable collection frequencies
        - Custom reduction strategies per metric
        - Hook-based processing
        - Separation of concerns
    """

    def __init__(
        self,
        *,
        tb_dir: str | None,
        batch_size: int,
        seq_len: int,
        world_size: int | None = None,
        tb_every: int = 1,
        cli_every: int = 20,
        window: int = 50,
        flush_secs: int = 30,
    ):
        """
        Initialize the Tracker object.

        Args:
            tb_dir: The directory to write the TensorBoard logs to.
            batch_size: The batch size.
            seq_len: The sequence length.
            world_size: The number of processes in the world.
            tb_every: The frequency of TensorBoard writes.
            cli_every: The frequency of CLI writes.
            window: The window size for the running mean.
            flush_secs: The frequency of GPU/CPU memory snapshot.
        """
        self.tb_every: int = tb_every
        self.cli_every: int = cli_every
        self.window: int = window
        self.buf: DefaultDict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

        # Token accounting - per worker
        self.tokens_per_step = batch_size * seq_len  # Local tokens per step
        self.last_tok_time = time.time()
        self.tok_accum = 0
        self.total_tokens = 0  # Cumulative total tokens processed by this worker

        # Epoch time tracking
        self.epoch_start_time: Optional[float] = None
        self.current_epoch: Optional[int] = None

        # Cached metrics - computed once per collection cycle and reused
        self.cached_metrics: dict[str, float] = {}
        self.last_collection_step: dict[int, int] = {}  # frequency -> last_step

        # Writers.
        # TODO: Create a dedicated tensorboard sub-directory for each run.
        self.tb = tb.SummaryWriter(tb_dir) if (tb_dir and _is_primary()) else None
        self.last_flush = time.time()
        self.flush_secs = flush_secs

        # TODO: On how to combine train and validation loss into a single TensorBoard plot.
        #       https://stackoverflow.com/questions/37146614/
        #           tensorboard-plot-training-and-validation-losses-on-the-same-graph

    def start_epoch(self, epoch: int) -> None:
        """
        Mark the start of a new epoch for timing.

        Args:
            epoch: The epoch number.
        """
        self.epoch_start_time = time.time()
        self.current_epoch = epoch

    def _should_collect_metrics(self, step: int, frequency: int) -> bool:
        """Check if metrics should be collected at this step for given frequency."""
        return step % frequency == 0

    def _process_metric_value(self, name: str, value: Any) -> float:
        """Process a metric value using its registered processor."""
        metric_def = METRIC_REGISTRY.get_metric(name)
        if metric_def and metric_def.processor:
            return metric_def.processor(value)
        return float(value)

    def _reduce_metric_values(self, name: str, values: Deque[float]) -> float:
        """Reduce metric values using the registered reduction strategy."""
        if not values:
            return 0.0

        metric_def = METRIC_REGISTRY.get_metric(name)
        if not metric_def:
            return sum(values) / len(values)  # Default: average

        if metric_def.reduction == ReductionStrategy.CURRENT:
            return values[-1]
        elif metric_def.reduction == ReductionStrategy.AVERAGE:
            return sum(values) / len(values)
        elif metric_def.reduction == ReductionStrategy.SUM:
            return sum(values)
        elif metric_def.reduction == ReductionStrategy.MAX:
            return max(values)
        elif metric_def.reduction == ReductionStrategy.MIN:
            return min(values)
        else:
            return sum(values) / len(values)

    def _collect_metrics_by_frequency(
        self, step: int, frequency: int
    ) -> Dict[str, float]:
        """Collect all metrics that should be gathered at this frequency."""
        collected = {}

        # Find all metrics with this frequency that have collectors
        for name, metric_def in METRIC_REGISTRY._metrics.items():
            if metric_def.frequency == frequency and metric_def.collector:
                try:
                    collector_metrics = metric_def.collector(step, self)
                    # Apply distributed reduction to collector metrics if needed
                    for metric_name, value in collector_metrics.items():
                        metric_def = METRIC_REGISTRY.get_metric(metric_name)
                        if metric_def and metric_def.distributed:
                            # Apply distributed reduction
                            value = _all_reduce_scalar(value)
                        collected[metric_name] = value
                except Exception as e:
                    print(f"Warning: Failed to collect {name}: {e}")

        return collected

    def update(self, step: int, **metrics: Any) -> None:
        """
        Call **once per optimization step**.

        Example:
            tracker.update(
                step=step,
                loss=loss_val,
                lr=lr,
                grad_norm=gn,
                epoch=epoch,
                batch=batch_count
            )

        Args:
            step: The current step.
            **metrics: The metrics to track.
        """
        # Process user-provided metrics
        for name, value in metrics.items():
            metric_def = METRIC_REGISTRY.get_metric(name)
            if metric_def:
                # Process the value
                processed_value = self._process_metric_value(name, value)

                # Apply distributed reduction if needed
                if metric_def.distributed:
                    processed_value = _all_reduce_scalar(processed_value)

                # Store in buffer (only on primary for non-distributed metrics)
                if metric_def.distributed or _is_primary():
                    self.buf[name].append(processed_value)
            else:
                # Unknown metric - store as-is with default processing
                if _is_primary():
                    self.buf[name].append(float(value))

        # Update token accounting
        self.tok_accum += self.tokens_per_step
        self.total_tokens += self.tokens_per_step

        # Collect metrics based on frequencies
        unique_frequencies = set(m.frequency for m in METRIC_REGISTRY._metrics.values())
        for frequency in unique_frequencies:
            if self._should_collect_metrics(step, frequency):
                collected = self._collect_metrics_by_frequency(step, frequency)
                self.cached_metrics.update(collected)
                self.last_collection_step[frequency] = step

        # TensorBoard writes
        if step % self.tb_every == 0 and self.tb and _is_primary():
            self._write_tensorboard_metrics(step)

        # CLI writes
        if step % self.cli_every == 0 and _is_primary():
            self._write_cli_metrics(step)

        # Reset token window every cli_every steps for fresh estimate
        if step % self.cli_every == 0:
            self.tok_accum = 0
            self.last_tok_time = time.time()

    def _write_tensorboard_metrics(self, step: int) -> None:
        """Write metrics to TensorBoard using registry definitions."""
        if not self.tb:
            return

        # Write buffered metrics
        for name, values in self.buf.items():
            if values:
                metric_def = METRIC_REGISTRY.get_metric(name)
                if metric_def:
                    reduced_value = self._reduce_metric_values(name, values)
                    tb_name = (
                        f"{metric_def.tb_prefix}{name}"
                        if metric_def.tb_prefix
                        else name
                    )
                    self.tb.add_scalar(tb_name, reduced_value, step)

        # Write cached metrics (from collectors)
        for name, value in self.cached_metrics.items():
            metric_def = METRIC_REGISTRY.get_metric(name)
            if metric_def:
                tb_name = (
                    f"{metric_def.tb_prefix}{name}" if metric_def.tb_prefix else name
                )
                self.tb.add_scalar(tb_name, value, step)

        # Flush occasionally
        now = time.time()
        if now - self.last_flush >= self.flush_secs:
            self.tb.flush()
            self.last_flush = now

    def _write_cli_metrics(self, step: int) -> None:
        """Write metrics to CLI using registry definitions."""
        msg = [f"[{step:,}]"]
        mem_msg = []

        # Process buffered metrics
        for name, values in self.buf.items():
            if values:
                metric_def = METRIC_REGISTRY.get_metric(name)
                if metric_def:
                    reduced_value = self._reduce_metric_values(name, values)
                    formatted = metric_def.cli_format.format(
                        name=name, value=reduced_value
                    )

                    # Group memory metrics separately
                    if metric_def.tb_prefix == "mem/":
                        mem_msg.append(formatted)
                    else:
                        msg.append(formatted)

        # Process cached metrics (from collectors)
        for name, value in self.cached_metrics.items():
            metric_def = METRIC_REGISTRY.get_metric(name)
            if metric_def:
                formatted = metric_def.cli_format.format(name=name, value=value)

                # Group memory metrics separately
                if metric_def.tb_prefix == "mem/":
                    mem_msg.append(formatted)
                else:
                    msg.append(formatted)

        # Print training metrics and memory on separate lines
        print("  ".join(msg), flush=True)
        if mem_msg:
            print(f"    MEMORY: {' | '.join(mem_msg)}", flush=True)

    def close(self) -> None:
        """
        Close the Tracker object.
        """
        if self.tb:
            self.tb.close()
