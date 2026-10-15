"""Transitional alias: this module moved to particle_mesh.kernels.torch_impl."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('particle_mesh.kernels.torch_impl')
