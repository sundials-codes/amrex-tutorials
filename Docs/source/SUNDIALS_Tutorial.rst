.. _tutorials_sundials:

=============================
SUNDIALS and Time Integrators
=============================

These example codes demonstrate how to use the AMReX TimeIntegrator class
with SUNDIALS backend for integration.

The first example code at ``amrex-tutorials/ExampleCodes/SUNDIALS/Single-Rate``
solves the heat equation:

.. math:: \frac{\partial\phi}{\partial t} = \nabla^2\phi.

The inputs file contains a template for single process time integration strategies.

The second example code at ``amrex-tutorials/ExampleCodes/SUNDIALS/Reaction-Diffusion``
solves the reaction-diffusion equation, where :math:`R` and :math:`D` are
user-supplied reaction and diffusion coefficients:

.. math:: \frac{\partial\phi}{\partial t} = D \nabla^2\phi - R \phi.

The inputs file contains a template for MRI approaches, where the diffusion process
can be treated as a "fast" partition relative to the reaction process.

The third example code at ``amrex-tutorials/ExampleCodes/SUNDIALS/AdvDiff-MLMGPrecon``
solves the advection-diffusion equation, where :math:`A` and :math:`D` are
user-supplied advection and diffusion coefficients:

.. math:: \frac{\partial\phi}{\partial t} = D \nabla^2\phi - A \nabla\phi.

The inputs file contains a template for IMEX approaches, where diffusion is
treated implicitly with a SUNDIALS preconditioner that uses AMReX MLMG. The
example can also use MLMG's HYPRE bottom solver interface when AMReX is built
with HYPRE support.

The fourth example code at
``amrex-tutorials/ExampleCodes/SUNDIALS/AdvDiff-TimeDependentDiffusion`` extends
the preconditioned advection-diffusion example to anisotropic, piecewise-constant
diffusion coefficients that change during the run.

AdvDiff-TimeDependentDiffusion
------------------------------

This example solves a two-dimensional periodic advection-diffusion problem on
the physical domain :math:`[-1,1]^2`.  The scalar solution is cell-centered and
stored in an AMReX ``MultiFab``.  Advection is treated explicitly and diffusion
is treated implicitly by an IMEX SUNDIALS method:

.. math::

   \frac{\partial \phi}{\partial t}
   =
   D_x(t) \frac{\partial^2 \phi}{\partial x^2}
   + D_y(t) \frac{\partial^2 \phi}{\partial y^2}
   - A_x \frac{\partial \phi}{\partial x}
   - A_y \frac{\partial \phi}{\partial y}.

The explicit advection term uses first-order upwinding.  The implicit diffusion
term uses the standard second-order centered stencil.  The initial condition is
a Gaussian pulse; unlike ``AdvDiff-MLMGPrecon``, this time-dependent problem does
not report an analytic exact-solution error at the end of the run.

The active diffusion coefficients are controlled by the base inputs
``diffCoeffx`` and ``diffCoeffy`` and by a schedule:

.. math::

   D_x(t) = \texttt{diffCoeffx}\,m_x^k,\qquad
   D_y(t) = \texttt{diffCoeffy}\,m_y^k,
   \qquad
   t_k \le t < t_{k+1}.

Here :math:`t_k` comes from ``diffusion.times``, and :math:`m_x^k` and
:math:`m_y^k` come from ``diffusion.x_multipliers`` and
``diffusion.y_multipliers``.  These three arrays must have the same length,
``diffusion.times`` must be strictly increasing and start at or before
:math:`t=0`, and every active coefficient must be positive.  When the evolution
reaches a schedule jump, the example reinitializes the SUNDIALS integrator and
drops the cached preconditioner so the next implicit solve uses the new
diffusion segment.

SUNDIALS preconditioner
~~~~~~~~~~~~~~~~~~~~~~~

The implicit SUNDIALS solve needs repeated approximate solutions of systems of
the form

.. math::

   \left(I - \gamma J_D(t)\right) z = r,

where :math:`\gamma` is supplied by SUNDIALS for the current implicit stage and
:math:`J_D(t)` is the Jacobian of the implicit diffusion right-hand side.  For
this linear diffusion operator,

.. math::

   J_D(t) z =
   D_x(t) \frac{\partial^2 z}{\partial x^2}
   + D_y(t) \frac{\partial^2 z}{\partial y^2}.

The example uses AMReX ``MLABecLaplacian`` and ``MLMG`` as a left
preconditioner for SUNDIALS GMRES.  ``MLABecLaplacian`` represents the canonical
AMReX cell-centered operator

.. math::

   \left(A \alpha - B \nabla \cdot \beta \nabla\right) z = r.

For this preconditioner the example sets :math:`A=1`, :math:`B=1`,
:math:`\alpha=1`, and face-centered coefficients

.. math::

   \beta_x = \gamma D_x(t),\qquad
   \beta_y = \gamma D_y(t).

Therefore the MLMG solve applies

.. math::

   P(\gamma,t) z =
   z - \gamma \left[
       D_x(t) \frac{\partial^2 z}{\partial x^2}
       + D_y(t) \frac{\partial^2 z}{\partial y^2}
   \right]
   = r,

which matches the SUNDIALS linear-system form :math:`I-\gamma J_D(t)`.  The code
reuses the ``MLABecLaplacian`` and ``MLMG`` objects while :math:`\gamma` and the
diffusion schedule segment are unchanged.  When either changes, it refreshes the
face-centered :math:`\beta` coefficients; at diffusion jumps the cached
preconditioner is rebuilt.

Useful inputs
~~~~~~~~~~~~~

The default ``Exec/inputs_sundials`` selects ``integration.type = SUNDIALS`` and
an ``IMEX-RK`` method with GMRES and left preconditioning.  The main
preconditioner controls are in the ``mlmg`` namespace:

- ``mlmg.reltol`` and ``mlmg.abstol`` set the relative and absolute tolerances
  passed to ``MLMG::solve``.
- ``mlmg.max_iter``, ``mlmg.max_fmg_iter``, ``mlmg.verbose``, and
  ``mlmg.bottom_verbose`` control the multigrid iteration and output.
- ``mlmg.max_coarsening_level`` limits geometric coarsening.  Setting it to
  ``0`` makes the configured bottom solver operate on the current level.
- ``mlmg.use_hypre``, ``mlmg.hypre_interface``, and
  ``mlmg.hypre_options_namespace`` enable and configure MLMG's HYPRE bottom
  solver interface when AMReX is built with HYPRE support.

The ``integration.sundials`` namespace controls the SUNDIALS method, linear
solver, preconditioning side, maximum linear iterations, and optional nonlinear
solver controls such as ``nlscoef``, ``max_nonlinear_iters``, ``eps_lin``,
``lsetup_frequency``, and ``jac_eval_frequency``.  The example also includes
optional ``rl_data`` and ``rl_control`` inputs for collecting per-step SUNDIALS
and MLMG telemetry or changing these SUNDIALS nonlinear controls during a run.

Please see the inputs file or
`AMReX User Guide:Time Integration`_ for more details.

.. _`AMReX User Guide:Time Integration`: https://amrex-codes.github.io/amrex/docs_html/TimeIntegration_Chapter.html#
