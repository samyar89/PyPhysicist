Thermodynamics
==============

Ideal gases
-----------

.. code-block:: python

   from PyPhysicist.thermodynamics.ideal_gases import ideal_gas_pressure

   p = ideal_gas_pressure(n=1.0, t=300.0, v=0.024)

Heat and entropy
----------------

.. code-block:: python

   from PyPhysicist.thermodynamics.heat import heat_capacity, entropy_change

   c = heat_capacity(heat=500.0, mass=2.0, delta_t=5.0)
   ds = entropy_change(heat=100.0, temperature=300.0)
