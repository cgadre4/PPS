"""
Architecture rules for the particle_mesh package, checked statically (AST) so a violation fails fast:
- only particle_mesh.kernels imports the Taichi kernel module (taichi_impl);
- ti.init / ti.reset appear only in particle_mesh.runtime.bootstrap;
- physics never imports render / ui / app, and render never imports ui / app;
- package modules never import the transitional flat top-level names (config, engine, ...).
"""
import ast
import os

import pytest

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(PM_DIR, "particle_mesh")
FLAT_NAMES = {"config", "colors", "camera", "grid_ops", "potentials", "pm_solver", "nbody_solver", "engine",
              "kernels", "kernels_torch", "kernels_taichi", "mesh_renderer", "circle_renderer", "gui", "image_loader"}


def _package_modules():
    for root, _, files in os.walk(PKG_DIR):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, PM_DIR)[:-3].replace(os.sep, ".")
                yield (rel[:-len(".__init__")] if rel.endswith(".__init__") else rel), path


MODULES = sorted(_package_modules())


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
            out += [f"{node.module}.{a.name}" for a in node.names]
    return tree, out


def _layer(module):
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


@pytest.mark.parametrize("module,path", MODULES, ids=[m for m, _ in MODULES])
def test_layering(module, path):
    tree, imports = _imports(path)
    layer = _layer(module)
    for imp in imports:
        if imp.endswith("taichi_impl"):
            assert module.startswith("particle_mesh.kernels"), f"{module} imports {imp}"
        assert imp.split(".")[0] not in FLAT_NAMES, f"{module} imports flat module {imp}"
        if imp.startswith("particle_mesh."):
            target = _layer(imp)
            if layer == "physics":
                assert target not in ("render", "ui", "app"), f"{module} imports {imp}"
            if layer == "render":
                assert target not in ("ui", "app"), f"{module} imports {imp}"
    if module != "particle_mesh.runtime.bootstrap":
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("init", "reset") \
                    and isinstance(node.value, ast.Name) and node.value.id == "ti":
                pytest.fail(f"{module} calls ti.{node.attr} (only runtime.bootstrap may)")


def test_init_taichi_is_idempotent():
    import taichi as ti
    from particle_mesh.runtime import bootstrap
    assert bootstrap.is_initialized()
    arch = ti.lang.impl.current_cfg().arch
    assert bootstrap.init_taichi() == arch
    assert bootstrap.init_taichi("cpu") == arch      # already initialised: the preference is ignored


def test_runtime_device_is_shared():
    from particle_mesh.kernels import to_taichi, taichi_shares_memory
    from particle_mesh.physics import engine
    from particle_mesh.runtime import device
    assert to_taichi is device.to_taichi and taichi_shares_memory is device.taichi_shares_memory
    assert engine.DEVICE == device.get_torch_device()
