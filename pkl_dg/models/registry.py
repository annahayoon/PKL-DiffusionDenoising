"""
Component Registry System for PKL Diffusion Models

This module provides a centralized registry system for all diffusion model components,
following the pattern used by HuggingFace Diffusers and Transformers.

Features:
- Automatic component discovery and registration
- Type-safe component creation with validation
- Plugin architecture for easy extension
- Configuration-based component instantiation
- Dependency injection and composition patterns
"""

from typing import Dict, Any, Type, Optional, Union, Callable, List, Tuple
from abc import ABC, abstractmethod
import inspect
import warnings
from collections import defaultdict
import importlib

# Type aliases for better code clarity
ComponentClass = Type[Any]
ComponentFactory = Callable[..., Any]
ComponentConfig = Dict[str, Any]


class ComponentRegistry:
    """Centralized registry for diffusion model components.
    
    Inspired by HuggingFace's registry pattern, this provides a clean way to
    register, discover, and instantiate components with proper validation.
    """
    
    def __init__(self, name: str):
        """Initialize registry for a specific component type.
        
        Args:
            name: Name of the component type (e.g., 'schedulers', 'samplers')
        """
        self.name = name
        self._components: Dict[str, ComponentClass] = {}
        self._factories: Dict[str, ComponentFactory] = {}
        self._configs: Dict[str, ComponentConfig] = {}
        self._aliases: Dict[str, str] = {}
        self._dependencies: Dict[str, List[str]] = defaultdict(list)
        
    def register(
        self,
        name: str,
        component_class: ComponentClass,
        factory: Optional[ComponentFactory] = None,
        config: Optional[ComponentConfig] = None,
        aliases: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        exist_ok: bool = False,
    ):
        """Register a component with the registry.
        
        Args:
            name: Unique name for the component
            component_class: The component class to register
            factory: Optional factory function for complex instantiation
            config: Default configuration for the component
            aliases: Alternative names for the component
            dependencies: List of required dependencies
            exist_ok: Whether to allow overwriting existing components
        """
        if name in self._components and not exist_ok:
            raise ValueError(f"Component '{name}' already registered in {self.name} registry")
        
        # Validate component class
        if not inspect.isclass(component_class):
            raise TypeError(f"Expected class, got {type(component_class)}")
        
        # Register main component
        self._components[name] = component_class
        self._factories[name] = factory
        self._configs[name] = config or {}
        self._dependencies[name] = dependencies or []
        
        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases and not exist_ok:
                    warnings.warn(f"Alias '{alias}' already exists, overwriting")
                self._aliases[alias] = name
        
        print(f"✅ Registered {self.name} component: {name}")
    
    def get(self, name: str) -> ComponentClass:
        """Get a registered component class.
        
        Args:
            name: Name or alias of the component
            
        Returns:
            The registered component class
        """
        # Check aliases first
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._components:
            available = list(self._components.keys()) + list(self._aliases.keys())
            raise KeyError(f"Component '{name}' not found in {self.name} registry. "
                          f"Available: {available}")
        
        return self._components[actual_name]
    
    def create(
        self,
        name: str,
        config: Optional[ComponentConfig] = None,
        **kwargs
    ) -> Any:
        """Create an instance of a registered component.
        
        Args:
            name: Name or alias of the component
            config: Configuration dict for the component
            **kwargs: Additional keyword arguments
            
        Returns:
            Instantiated component
        """
        actual_name = self._aliases.get(name, name)
        component_class = self.get(name)
        
        # Merge configurations
        final_config = {}
        final_config.update(self._configs.get(actual_name, {}))
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        # Check dependencies
        self._check_dependencies(actual_name)
        
        # Use factory if available
        factory = self._factories.get(actual_name)
        if factory:
            return factory(component_class, final_config)
        
        # Standard instantiation
        try:
            return component_class(**final_config)
        except Exception as e:
            raise RuntimeError(f"Failed to create {self.name} component '{name}': {e}")
    
    def _check_dependencies(self, name: str):
        """Check if all dependencies are satisfied."""
        dependencies = self._dependencies.get(name, [])
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                warnings.warn(f"Optional dependency '{dep}' not available for {name}")
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())
    
    def list_aliases(self) -> Dict[str, str]:
        """List all registered aliases."""
        return dict(self._aliases)
    
    def get_config(self, name: str) -> ComponentConfig:
        """Get default configuration for a component."""
        actual_name = self._aliases.get(name, name)
        return self._configs.get(actual_name, {}).copy()
    
    def update_config(self, name: str, config: ComponentConfig):
        """Update default configuration for a component."""
        actual_name = self._aliases.get(name, name)
        if actual_name not in self._components:
            raise KeyError(f"Component '{name}' not registered")
        
        self._configs[actual_name].update(config)
    
    def unregister(self, name: str):
        """Unregister a component."""
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._components:
            raise KeyError(f"Component '{name}' not registered")
        
        # Remove from all registries
        del self._components[actual_name]
        if actual_name in self._factories:
            del self._factories[actual_name]
        if actual_name in self._configs:
            del self._configs[actual_name]
        if actual_name in self._dependencies:
            del self._dependencies[actual_name]
        
        # Remove aliases
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == actual_name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
    
    def __contains__(self, name: str) -> bool:
        """Check if component is registered."""
        actual_name = self._aliases.get(name, name)
        return actual_name in self._components
    
    def __len__(self) -> int:
        """Get number of registered components."""
        return len(self._components)
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ComponentRegistry('{self.name}', {len(self._components)} components)"


# Global registries for different component types
SCHEDULER_REGISTRY = ComponentRegistry("schedulers")
SAMPLER_REGISTRY = ComponentRegistry("samplers")
LOSS_REGISTRY = ComponentRegistry("losses")
MODEL_REGISTRY = ComponentRegistry("models")
STRATEGY_REGISTRY = ComponentRegistry("strategies")


class AutoRegistry:
    """Automatic component registration using decorators.
    
    Provides decorators for easy component registration following
    the pattern used in modern ML frameworks.
    """
    
    @staticmethod
    def scheduler(
        name: str,
        aliases: Optional[List[str]] = None,
        config: Optional[ComponentConfig] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """Decorator to register a scheduler component."""
        def decorator(cls):
            SCHEDULER_REGISTRY.register(
                name=name,
                component_class=cls,
                aliases=aliases,
                config=config,
                dependencies=dependencies,
                exist_ok=True,
            )
            return cls
        return decorator
    
    @staticmethod
    def sampler(
        name: str,
        aliases: Optional[List[str]] = None,
        config: Optional[ComponentConfig] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """Decorator to register a sampler component."""
        def decorator(cls):
            SAMPLER_REGISTRY.register(
                name=name,
                component_class=cls,
                aliases=aliases,
                config=config,
                dependencies=dependencies,
                exist_ok=True,
            )
            return cls
        return decorator
    
    @staticmethod
    def loss(
        name: str,
        aliases: Optional[List[str]] = None,
        config: Optional[ComponentConfig] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """Decorator to register a loss component."""
        def decorator(cls):
            LOSS_REGISTRY.register(
                name=name,
                component_class=cls,
                aliases=aliases,
                config=config,
                dependencies=dependencies,
                exist_ok=True,
            )
            return cls
        return decorator
    
    @staticmethod
    def model(
        name: str,
        aliases: Optional[List[str]] = None,
        config: Optional[ComponentConfig] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """Decorator to register a model component."""
        def decorator(cls):
            MODEL_REGISTRY.register(
                name=name,
                component_class=cls,
                aliases=aliases,
                config=config,
                dependencies=dependencies,
                exist_ok=True,
            )
            return cls
        return decorator
    
    @staticmethod
    def strategy(
        name: str,
        aliases: Optional[List[str]] = None,
        config: Optional[ComponentConfig] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """Decorator to register a training strategy component."""
        def decorator(cls):
            STRATEGY_REGISTRY.register(
                name=name,
                component_class=cls,
                aliases=aliases,
                config=config,
                dependencies=dependencies,
                exist_ok=True,
            )
            return cls
        return decorator


def create_component(
    registry: ComponentRegistry,
    name: str,
    config: Optional[ComponentConfig] = None,
    **kwargs
) -> Any:
    """Generic component creation function.
    
    Args:
        registry: The registry to use
        name: Component name
        config: Configuration dict
        **kwargs: Additional arguments
        
    Returns:
        Created component instance
    """
    return registry.create(name, config, **kwargs)


def get_available_components() -> Dict[str, List[str]]:
    """Get all available components across all registries."""
    return {
        "schedulers": SCHEDULER_REGISTRY.list_components(),
        "samplers": SAMPLER_REGISTRY.list_components(),
        "losses": LOSS_REGISTRY.list_components(),
        "models": MODEL_REGISTRY.list_components(),
        "strategies": STRATEGY_REGISTRY.list_components(),
    }


def print_registry_status():
    """Print status of all registries."""
    print("🚀 PKL Diffusion Component Registry Status")
    print("=" * 50)
    
    registries = [
        ("Schedulers", SCHEDULER_REGISTRY),
        ("Samplers", SAMPLER_REGISTRY),
        ("Losses", LOSS_REGISTRY),
        ("Models", MODEL_REGISTRY),
        ("Strategies", STRATEGY_REGISTRY),
    ]
    
    for name, registry in registries:
        components = registry.list_components()
        aliases = registry.list_aliases()
        
        print(f"\n{name}: {len(components)} registered")
        if components:
            print(f"  Components: {', '.join(components)}")
        if aliases:
            print(f"  Aliases: {dict(list(aliases.items())[:3])}{'...' if len(aliases) > 3 else ''}")


# Convenience aliases
register_scheduler = AutoRegistry.scheduler
register_sampler = AutoRegistry.sampler
register_loss = AutoRegistry.loss
register_model = AutoRegistry.model
register_strategy = AutoRegistry.strategy


__all__ = [
    "ComponentRegistry",
    "AutoRegistry",
    "SCHEDULER_REGISTRY",
    "SAMPLER_REGISTRY", 
    "LOSS_REGISTRY",
    "MODEL_REGISTRY",
    "STRATEGY_REGISTRY",
    "create_component",
    "get_available_components",
    "print_registry_status",
    "register_scheduler",
    "register_sampler",
    "register_loss", 
    "register_model",
    "register_strategy",
]
