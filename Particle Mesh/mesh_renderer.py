"""Transitional alias: this module moved to particle_mesh.render.mesh_renderer."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('particle_mesh.render.mesh_renderer')
