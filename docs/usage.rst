Usage
=====

Quick start
-----------

.. code-block:: python

   import PyPhysicist as pp

   force = pp.newton_second_law(mass=2.0, acceleration=9.81)
   voltage = pp.V(I=0.5, R=10.0)
   energy = pp.relativistic_energy(mass=1.0, c=299_792_458)

   print(force, voltage, energy)

Additional examples
-------------------

.. code-block:: python

   from PyPhysicist import momentum, ideal_gas_pressure, refractive_index

   p = momentum(mass=3.2, velocity=4.5)
   pressure = ideal_gas_pressure(n=1.0, r=8.314, t=300.0, v=0.024)
   index = refractive_index(c=299_792_458, v=200_000_000)

   print(p, pressure, index)
