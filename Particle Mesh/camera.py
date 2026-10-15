"""Transitional alias: this module moved to particle_mesh.render.camera."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('particle_mesh.render.camera')
