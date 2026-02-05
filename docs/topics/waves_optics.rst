Waves & Optics
==============

Waves
-----

.. code-block:: python

   from PyPhysicist.waves_optics.waves import frequency, wavelength

   f = frequency(wave_speed=340.0, wavelength_value=0.5)
   lam = wavelength(wave_speed=340.0, frequency_hz=680.0)

Optics
------

.. code-block:: python

   from PyPhysicist.waves_optics.optics import refractive_index

   n = refractive_index(speed_of_light=299_792_458, medium_speed=200_000_000)
