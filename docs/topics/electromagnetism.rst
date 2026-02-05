Electromagnetism
================

Electrostatics
--------------

Use electrostatics helpers for field and force relationships.

.. code-block:: python

   from PyPhysicist.electromagnetism.electrostatics import coulomb_force, electric_field

   f = coulomb_force(charge1=1e-6, charge2=2e-6, distance=0.05)
   e = electric_field(force_value=0.01, charge=1e-6)

Circuits
--------

Use circuit helpers for Ohm's law and equivalent resistance calculations.

.. code-block:: python

   from PyPhysicist.electromagnetism.circuits import voltage, resistance_parallel

   v = voltage(current=0.5, resistance=10.0)
   r_eq = resistance_parallel(100.0, 220.0, 330.0)
