"""Transitional alias: this module moved to particle_mesh.render.circle_renderer."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('particle_mesh.render.circle_renderer')
