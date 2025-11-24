"""Internal module for high-performance on-disk preprocessing components.

This module contains the internal implementation details for the superior
on-disk preprocessor architecture. These components are not part of the
public API and may change between versions.

Components
----------
storage_backend : Storage layer
    - MemoryMappedStorage: Fast mmap-based storage with compression

parallel_processor : Multi-core preprocessing
    - ParallelProcessor: Batch processing with ProcessPoolExecutor

transform_pipeline : Transform management
    - TransformPipeline: Two-tier and DAG-based transform handling

lazy_access : Smart data access layer
    - SmartLazyList: Lazy lists with caching and prefetching
    - LRUCache: In-memory cache for hot samples

utils : Shared utilities
    - Helper functions and common patterns
"""

# Storage backends will be imported as needed
# Not exposed in public API

__all__ = []  # No public exports from internal module
