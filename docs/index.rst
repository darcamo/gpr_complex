Welcome to gpr_complex's documentation!
=======================================

A Gaussian Process regression library that can work with complex numbers.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Usage
=====

Create an object of the :class:`~gpr_complex.model.GPR` class using one of the
kernels available in :mod:`~gpr_complex.kernels`, such as
:class:`~gpr_complex.kernels.RBF_ComplexProper`. Then use the
:meth:`~gpr_complex.model.GPR.fit` method to train the model and the
:meth:`~gpr_complex.model.GPR.predict` method to perform predictions.

For models trained with only one feature you can visualize the model with the
:meth:`~gpr_complex.model.GPR.chart` method.


Modules in gpr_complex
======================

.. toctree::
   :maxdepth: 2

   gpr_complex.kernels
   gpr_complex.model
   gpr_complex.plot
   gpr_complex.gaussian_process


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
