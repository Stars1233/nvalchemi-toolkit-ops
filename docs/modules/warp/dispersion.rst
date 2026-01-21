:mod:`nvalchemiops.interactions.dispersion`: Dispersion Corrections
===================================================================

.. automodule:: nvalchemiops.interactions.dispersion
    :no-members:
    :no-inherited-members:

Warp-Level Interface
--------------------

DFT-D3(BJ) Dispersion Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip::
   This is the low-level Warp interface that operates on ``warp.array`` objects.
   For PyTorch tensor support, see :doc:`../torch/dispersion`.

The DFT-D3 implementation supports two neighbor representation formats:

- **Neighbor matrix** (dense): ``[num_atoms, max_neighbors]`` with padding
- **Neighbor list** (sparse COO): ``[2, num_pairs]`` without padding

Both formats produce identical results and support all features including periodic
boundary conditions, batching, and smooth cutoff functions.

.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.wp_dftd3_nm
.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.wp_dftd3_nl
