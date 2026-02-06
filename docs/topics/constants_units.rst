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
for tagging values with units, along with a small converter that validates
dimensions and handles common SI unit conversions. It is intentionally minimal;
for full unit algebra, use Pint and pass Pint quantities into the formula
helpers.

.. code-block:: python

   from PyPhysicist.units import Quantity

   v = Quantity(12.0, "m/s")
   print(v)

The :class:`~PyPhysicist.units.Quantity` helper also supports basic arithmetic
operators. Addition/subtraction requires compatible units, while
multiplication/division propagates unit dimensions:

.. code-block:: python

   from PyPhysicist import Quantity

   force = Quantity(575, "N")
   mass = Quantity(10011, "Mg")
   acceleration = force / mass

   print(acceleration)  # Quantity(value=..., unit='m/s^2')

You can also convert values or catch dimensional errors explicitly:

.. code-block:: python

   from PyPhysicist.units import UnitError, convert_value

   speed_mps = convert_value(90, "km/hr", "m/s")
   try:
       convert_value(1, "kg", "m")
   except UnitError:
       print("Incompatible units.")
