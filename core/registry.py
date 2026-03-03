"""Plugin registration system for automatic algorithm and model discovery.

This module provides decorators and functions for registering and discovering
feature selection algorithms and regression models.
"""

import importlib
import logging
import pkgutil
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)

# Global registries
_algorithm_registry: Dict[str, Type] = {}
_model_registry: Dict[str, Type] = {}


def register_algorithm(name: str):
    """Decorator to register a feature selection algorithm.

    Args:
        name: Algorithm name (e.g., "HHO", "GA", "PCA")

    Example:
        @register_algorithm("HHO")
        class HHOSelector(BaseMealpySelector):
            ...
    """
    def decorator(cls):
        _algorithm_registry[name] = cls
        return cls
    return decorator


def register_model(name: str):
    """Decorator to register a regression model.

    Args:
        name: Model name (e.g., "PLS", "RF", "SVM")

    Example:
        @register_model("PLS")
        class PLSModel(BaseModel):
            ...
    """
    def decorator(cls):
        _model_registry[name] = cls
        return cls
    return decorator


def get_algorithm(name: str) -> Type:
    """Get a registered algorithm class by name.

    Args:
        name: Algorithm name

    Returns:
        Algorithm class

    Raises:
        KeyError: If algorithm not found
    """
    if name not in _algorithm_registry:
        available = ', '.join(_algorithm_registry.keys())
        raise KeyError(
            f"Algorithm '{name}' not found. Available algorithms: {available}"
        )
    return _algorithm_registry[name]


def get_model(name: str) -> Type:
    """Get a registered model class by name.

    Args:
        name: Model name

    Returns:
        Model class

    Raises:
        KeyError: If model not found
    """
    if name not in _model_registry:
        available = ', '.join(_model_registry.keys())
        raise KeyError(
            f"Model '{name}' not found. Available models: {available}"
        )
    return _model_registry[name]


def discover_plugins(package):
    """Automatically discover and import all modules in a package.

    This function walks through all modules in the given package and imports them,
    triggering any @register_algorithm or @register_model decorators.

    Args:
        package: Package object (e.g., import feature_selection; discover_plugins(feature_selection))
    """
    prefix = package.__name__ + "."
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        if not modname.endswith('.base'):  # Skip base modules
            try:
                importlib.import_module(modname)
            except Exception as e:
                # Log but don't fail - allows partial loading
                logger.warning(f"Failed to load {modname}: {e}")


def list_algorithms():
    """List all registered algorithms."""
    return list(_algorithm_registry.keys())


def list_models():
    """List all registered models."""
    return list(_model_registry.keys())
