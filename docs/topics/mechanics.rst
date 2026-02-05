Mechanics
=========

Kinematics
----------

Use the kinematics helpers for velocity and circular motion relationships.

.. code-block:: python

   from PyPhysicist.mechanics.kinematics import velocity, centripetal_acceleration

   v = velocity(distance=120.0, time=6.0)
   a_c = centripetal_acceleration(speed=12.0, radius=4.0)

Dynamics
--------

Use the dynamics helpers for forces, momentum, and weight.

.. code-block:: python

   from PyPhysicist.mechanics.dynamics import force, momentum, weight

   f = force(mass=2.0, acceleration=9.81)
   p = momentum(mass=3.2, velocity=4.5)
   w = weight(mass=5.0, gravity=9.81)

Energy
------

Use the energy helpers for mechanical energy relationships.

.. code-block:: python

   from PyPhysicist.mechanics.energy import kinetic_energy, work

   ke = kinetic_energy(mass=4.0, velocity=3.0)
   w = work(force_value=12.0, displacement=2.0)
