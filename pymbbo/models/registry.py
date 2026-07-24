import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, Type, Optional, Callable
from pymbbo.architectures.base_arch import BaseArchitecture

ARCHITECTURE_REGISTRY: Dict[str, Type[BaseArchitecture]] = {}

def register_architecture(name: Optional[str] = None):
    """
    Decorator to register a custom architecture class into PYMBBO.
    Usage:
        @register_architecture("my_custom_net")
        class MyCustomNet(BaseArchitecture):
            ...
    """
    def decorator(cls: Type[BaseArchitecture]):
        arch_name = (name or cls.__name__).lower()
        ARCHITECTURE_REGISTRY[arch_name] = cls
        return cls

    if inspect.isclass(name):
        cls = name
        arch_name = cls.__name__.lower()
        ARCHITECTURE_REGISTRY[arch_name] = cls
        return cls

    return decorator


def discover_architectures(custom_dir: Optional[str] = None) -> Dict[str, Type[BaseArchitecture]]:
    """
    Automatically scans the `pymbbo/architectures/` folder (and optional custom user directories)
    for architecture subfolders and registers any subclass of BaseArchitecture found.

    Supports both single-file architectures and multi-file packages with relative imports.
    """
    target_dirs = []
    
    # Built-in architectures directory
    base_arch_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_architectures_dir = os.path.join(base_arch_dir, "architectures")
    if os.path.exists(default_architectures_dir):
        target_dirs.append(("pymbbo.architectures", default_architectures_dir))

    if custom_dir and os.path.exists(custom_dir):
        target_dirs.append((None, custom_dir))

    for parent_module_name, search_dir in target_dirs:
        # Track directories already loaded as packages to skip in second pass
        loaded_package_dirs = set()

        # First pass: try importing subfolders as proper packages (supports relative imports)
        for entry in os.listdir(search_dir):
            entry_path = os.path.join(search_dir, entry)
            init_file = os.path.join(entry_path, "__init__.py")

            if os.path.isdir(entry_path) and os.path.isfile(init_file) and not entry.startswith("__"):
                if parent_module_name:
                    full_module_name = f"{parent_module_name}.{entry}"
                else:
                    full_module_name = f"pymbbo_dynamic_pkg_{entry}"

                try:
                    # Use importlib.import_module for proper package imports
                    if full_module_name not in sys.modules:
                        mod = importlib.import_module(full_module_name)
                    else:
                        mod = sys.modules[full_module_name]

                    # Scan for BaseArchitecture subclasses in the loaded module
                    for obj_name, obj in inspect.getmembers(mod, inspect.isclass):
                        if issubclass(obj, BaseArchitecture) and obj is not BaseArchitecture:
                            reg_name = getattr(obj, "ARCH_NAME", obj_name.lower())
                            if reg_name not in ARCHITECTURE_REGISTRY:
                                ARCHITECTURE_REGISTRY[reg_name] = obj

                    loaded_package_dirs.add(os.path.normpath(entry_path))
                except Exception:
                    pass

        # Second pass: scan individual .py files (single-file architectures and non-package folders)
        for root, dirs, files in os.walk(search_dir):
            # Skip directories already loaded as packages
            if os.path.normpath(root) in loaded_package_dirs:
                dirs.clear()  # Don't descend into subdirectories
                continue

            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    filepath = os.path.join(root, file)
                    module_name = f"pymbbo_dynamic_{os.path.basename(root)}_{file[:-3]}"
                    
                    # Skip if already loaded
                    if module_name in sys.modules:
                        continue

                    try:
                        spec = importlib.util.spec_from_file_location(module_name, filepath)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = mod
                            spec.loader.exec_module(mod)
                            
                            # Find any subclasses of BaseArchitecture
                            for obj_name, obj in inspect.getmembers(mod, inspect.isclass):
                                if issubclass(obj, BaseArchitecture) and obj is not BaseArchitecture:
                                    reg_name = getattr(obj, "ARCH_NAME", obj_name.lower())
                                    if reg_name not in ARCHITECTURE_REGISTRY:
                                        ARCHITECTURE_REGISTRY[reg_name] = obj
                    except Exception as e:
                        # Continue if file has unfulfilled optional dependencies
                        pass

    return ARCHITECTURE_REGISTRY
