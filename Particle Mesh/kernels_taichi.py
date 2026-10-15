"""Transitional alias: this module moved to particle_mesh.kernels.taichi_impl."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('particle_mesh.kernels.taichi_impl')
