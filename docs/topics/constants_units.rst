Constants & Units
=================

SI constants
------------

.. code-block:: python

   from PyPhysicist.constants import SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

   print(SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT)

Units strategy
--------------

PyPhysicist ships with a lightweight :class:`PyPhysicist.units.Quantity` helper
for tagging values with units. It is intentionally minimal; for full unit
algebra, use Pint and pass Pint quantities into the formula helpers.

.. code-block:: python

   from PyPhysicist.units import Quantity

   v = Quantity(12.0, "m/s")
   print(v)
