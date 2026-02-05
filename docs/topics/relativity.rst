Relativity
==========

Special relativity
------------------

.. code-block:: python

   from PyPhysicist.relativity.special import time_dilation, length_contraction

   t = time_dilation(proper_time=1.0, velocity=100_000_000)
   l = length_contraction(proper_length=10.0, velocity=100_000_000)

Relativistic gravity
--------------------

.. code-block:: python

   from PyPhysicist.relativity.gravity import schwarzschild_radius

   r_s = schwarzschild_radius(mass=5.0)
