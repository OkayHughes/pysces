#!/usr/bin/env python3
"""Generate API documentation automatically."""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sphinx.ext.autosummary.generate import generate_autosummary_docs

# List of modules to document
modules = [
    'pysces',
    'pysces.distributed_memory',
    'pysces.mesh_generation',
    'pysces.models_3d',
    'pysces.operations_2d',
    'pysces.shallow_water_models',
]

# Output directory for generated rst files
output_dir = 'api'

for module in modules:
    generate_autosummary_docs(
        [module],
        output_dir=output_dir,
        suffix='.rst',
        base_path='.'
    )

print("API documentation generated successfully!")