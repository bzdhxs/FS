"""Regression models with automatic plugin discovery."""

from core.registry import discover_plugins
import model as _self_pkg

# Trigger automatic discovery and registration of all models
discover_plugins(_self_pkg)
