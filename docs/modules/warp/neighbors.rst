:mod:`nvalchemiops.neighbors`: Neighbor Lists
===================================================

.. automodule:: nvalchemiops.neighbors
    :no-members:
    :no-inherited-members:

Warp-Level Interface
--------------------

.. tip::
   This is the low-level Warp interface that operates on ``warp.array`` objects.
   For PyTorch tensor support, see :doc:`../torch/neighbors`.

Naive Algorithm
^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.naive.wp_naive_neighbor_matrix
.. autofunction:: nvalchemiops.neighbors.naive.wp_naive_neighbor_matrix_pbc

Cell List Algorithm
^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.cell_list.wp_build_cell_list
.. autofunction:: nvalchemiops.neighbors.cell_list.wp_query_cell_list

Batched Naive Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.batch_naive.wp_batch_naive_neighbor_matrix
.. autofunction:: nvalchemiops.neighbors.batch_naive.wp_batch_naive_neighbor_matrix_pbc

Batched Cell List Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.batch_cell_list.wp_batch_build_cell_list
.. autofunction:: nvalchemiops.neighbors.batch_cell_list.wp_batch_query_cell_list

Naive Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.naive_dual_cutoff.wp_naive_neighbor_matrix_dual_cutoff
.. autofunction:: nvalchemiops.neighbors.naive_dual_cutoff.wp_naive_neighbor_matrix_pbc_dual_cutoff

Batched Naive Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.batch_naive_dual_cutoff.wp_batch_naive_neighbor_matrix_dual_cutoff
.. autofunction:: nvalchemiops.neighbors.batch_naive_dual_cutoff.wp_batch_naive_neighbor_matrix_pbc_dual_cutoff

Rebuild Detection
^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.rebuild_detection.wp_check_cell_list_rebuild
.. autofunction:: nvalchemiops.neighbors.rebuild_detection.wp_check_neighbor_list_rebuild

Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.neighbor_utils.wp_zero_array
.. autofunction:: nvalchemiops.neighbors.neighbor_utils.estimate_max_neighbors
