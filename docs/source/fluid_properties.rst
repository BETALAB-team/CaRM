Fluid Properties
================

The :class:`~carm.fluid.Fluid` dataclass holds the thermophysical properties
of the heat carrier fluid circulating in the borehole heat exchanger.


Using CoolProp
--------------

`CoolProp <http://www.coolprop.org>`_ can be used to compute the fluid properties
at a given temperature and pass them directly to :class:`~carm.fluid.Fluid`.

Common fluid strings for BHE applications:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - CoolProp string
     - Fluid
     - Freeze point (approx.)
   * - ``'Water'``
     - Pure water
     - 0 °C
   * - ``'INCOMP::MEG[0.20]'``
     - Monoethylene glycol 20 %
     - −9 °C
   * - ``'INCOMP::MEG[0.25]'``
     - Monoethylene glycol 25 %
     - −12 °C
   * - ``'INCOMP::MEG[0.30]'``
     - Monoethylene glycol 30 %
     - −15 °C
   * - ``'INCOMP::MEG[0.35]'``
     - Monoethylene glycol 35 %
     - −20 °C
   * - ``'INCOMP::MPG[0.20]'``
     - Monopropylene glycol 20 %
     - −7 °C
   * - ``'INCOMP::MPG[0.25]'``
     - Monopropylene glycol 25 %
     - −10 °C
   * - ``'INCOMP::MPG[0.30]'``
     - Monopropylene glycol 30 %
     - −13 °C
   * - ``'INCOMP::MPG[0.35]'``
     - Monopropylene glycol 35 %
     - −18 °C

The fraction value in brackets is the mass fraction (e.g. ``0.25`` = 25 % by mass).
The full list of available incompressible mixtures is available on the
`CoolProp documentation <http://www.coolprop.org/fluid_properties/Incompressible.html>`_.

Example
-------

.. code-block:: python

   from CoolProp.CoolProp import PropsSI
   from carm.fluid import Fluid

   fluid_str = 'INCOMP::MEG[0.25]'
   T = 278.15  # K  (5 °C, typical BHE fluid temperature)
   P = 101325  # Pa

   rho = PropsSI('D', 'T', T, 'P', P, fluid_str)

   fluid = Fluid(
       k_w   = PropsSI('L', 'T', T, 'P', P, fluid_str),
       rho_w  = rho,
       cp_w   = PropsSI('C', 'T', T, 'P', P, fluid_str),
       ni_w   = PropsSI('V', 'T', T, 'P', P, fluid_str) / rho,
   )