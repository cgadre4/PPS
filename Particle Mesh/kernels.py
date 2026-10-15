"""Transitional alias: this module moved to particle_mesh.kernels."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('particle_mesh.kernels')
