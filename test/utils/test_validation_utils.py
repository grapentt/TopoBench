"""Tests for validation utilities."""

import pytest
import torch

from topobench.utils.validation_utils import (
    MemoryTracker,
    calculate_oom_params,
    expect_oom,
)


class TestMemoryTracker:
    """Test MemoryTracker class."""

    def test_tracker_basic(self):
        """Test basic memory tracking."""
        tracker = MemoryTracker()
        
        with tracker:
            # Allocate some memory
            x = torch.randn(1000, 1000)
            tracker.update()
        
        # Should have tracked some memory
        assert tracker.peak_ram_mb > 0
        assert tracker.start_ram_mb > 0

    def test_tracker_delta(self):
        """Test memory delta calculation."""
        tracker = MemoryTracker()
        
        with tracker:
            # Allocate memory
            x = torch.randn(1000, 1000)
            tracker.update()
        
        # Delta should be non-negative
        assert tracker.get_delta_ram() >= 0

    def test_tracker_report(self, capsys):
        """Test memory report output."""
        tracker = MemoryTracker()
        
        with tracker:
            pass
        
        tracker.report(prefix="Test ")
        captured = capsys.readouterr()
        
        assert "Test Memory Usage Report" in captured.out
        assert "RAM:" in captured.out


class TestCalculateOOMParams:
    """Test calculate_oom_params function."""

    def test_inductive_params(self):
        """Test parameter calculation for inductive."""
        params = calculate_oom_params(
            max_ram_gb=4.0,
            approach="inductive",
            structure_type="triangles"
        )
        
        # Should return required keys
        assert "num_graphs" in params
        assert "nodes_per_graph" in params
        assert "estimated_structures" in params
        assert "estimated_memory_gb" in params
        
        # Should exceed target RAM
        assert params["estimated_memory_gb"] > 4.0 * 0.7

    def test_transductive_params(self):
        """Test parameter calculation for transductive."""
        params = calculate_oom_params(
            max_ram_gb=4.0,
            approach="transductive",
            structure_type="triangles"
        )
        
        # Should return required keys
        assert "num_nodes" in params
        assert "avg_degree" in params
        assert "estimated_structures" in params
        assert "estimated_memory_gb" in params
        
        # Should exceed target RAM
        assert params["estimated_memory_gb"] > 4.0 * 0.7

    def test_4cliques_params(self):
        """Test parameter calculation for 4-cliques."""
        params = calculate_oom_params(
            max_ram_gb=4.0,
            approach="inductive",
            structure_type="4-cliques"
        )
        
        # Should have valid parameters
        assert params["num_graphs"] > 0
        assert params["estimated_memory_gb"] > 0


class TestExpectOOM:
    """Test expect_oom function."""

    def test_oom_detection_memory_error(self):
        """Test detection of MemoryError."""
        def will_oom():
            raise MemoryError("Out of memory")
        
        oom_occurred, msg = expect_oom(will_oom)
        assert oom_occurred
        assert "MemoryError" in msg

    def test_oom_detection_runtime_error(self):
        """Test detection of RuntimeError with OOM message."""
        def will_oom():
            raise RuntimeError("CUDA out of memory")
        
        oom_occurred, msg = expect_oom(will_oom)
        assert oom_occurred
        assert "RuntimeError" in msg

    def test_no_oom(self):
        """Test detection when no OOM occurs."""
        def wont_oom():
            return True
        
        oom_occurred, msg = expect_oom(wont_oom)
        assert not oom_occurred
        assert "without OOM" in msg

    def test_other_exception(self):
        """Test handling of other exceptions."""
        def other_error():
            raise ValueError("Something else")
        
        oom_occurred, msg = expect_oom(other_error)
        assert not oom_occurred
        assert "Unexpected exception" in msg
