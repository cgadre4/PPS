"""Transitional alias: this module moved to particle_mesh.physics.engine."""
import importlib, sys
sys.modules[__name__] = importlib.import_module('particle_mesh.physics.engine')
