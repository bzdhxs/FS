"""Feature selection algorithms with automatic plugin discovery."""

from core.registry import discover_plugins
import feature_selection as _self_pkg

# Trigger automatic discovery and registration of all algorithms
discover_plugins(_self_pkg)
