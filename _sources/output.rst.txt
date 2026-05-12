Output
======

Simulation results
------------------

``Simulation.run()`` returns ``T_history``, a NumPy array of shape:

.. code-block:: python

    (n_steps + 1, n_bhes, n_dof)

where:

- ``n_steps + 1`` — timesteps, index 0 is the initial condition
- ``n_bhes`` — number of boreholes
- ``n_dof`` — total degrees of freedom per borehole

The state vector along the third axis follows the domain layout described
in :doc:`discretization`:

.. code-block:: text

    [ sup | ground | borehole | inf ]

Any quantity can be extracted by computing the correct offset into this
vector. The examples in ``examples/`` cover the most common cases; users
can freely extend them to extract any quantity of interest.

Offsets
-------

.. code-block:: python

    nsup    = m_mesh_sup + 1        # start of ground domain
    nground = n_mesh * m_mesh       # start of borehole domain

Extracting results
------------------

**Outlet fluid temperature over time** for borehole ``b``:

.. code-block:: python

    Tfout = T_history[1:, b, nsup + nground + (props_b.n_equations - 1)]

**Shell temperature vertical profile** at timestep ``t``:

.. code-block:: python

    slice_shell = [
        nsup + nground + j * props_b.n_equations
        for j in range(m_mesh)
    ]
    T_shell = T_history[t, b, slice_shell]

**Fluid down/up temperature vertical profile** at timestep ``t``:

.. code-block:: python

    slice_down = [
        nsup + nground + j * props_b.n_equations + (props_b.n_equations - 2)
        for j in range(m_mesh)
    ]
    slice_up = [
        nsup + nground + j * props_b.n_equations + (props_b.n_equations - 1)
        for j in range(m_mesh)
    ]
    T_down = T_history[t, b, slice_down]
    T_up   = T_history[t, b, slice_up]

The offsets ``n_equations - 2`` and ``n_equations - 1`` always point to
fluid down and fluid up regardless of BHE type.

Depth axis
----------

To associate profile values with physical depth:

.. code-block:: python

    import numpy as np

    dz    = model.ground[0].dz
    depth = np.arange(-L_sup, -L_sup - dz * m_mesh, -dz)

``depth`` has length ``m_mesh`` and matches the profile arrays element-by-element.

.. note::
   Complete working examples are available in the ``examples/`` directory
   of the repository, covering single-borehole, parallel, and series configurations.