miniball
========

These are python bindings to Bernd Gärtners `miniball software`__.

Setup
-----

To install *miniball* clone the repository and run the setup script:

.. code-block:: bash
   
   python setup.py install

There is also a package for Arch Linux: https://aur.archlinux.org/packages/python-miniball/

Example
-------

.. code-block:: python

   import math
   import random
   import miniball

   P = [(random.uniform(0, 100), random.uniform(0, 100)) for i in range(10000)]
   mb = miniball.Miniball(P)
   print('Center', mb.center())
   print('Radius', math.sqrt(mb.squared_radius()))

Notes
-----
This algorithm has some numerical challenges worth mentioning. The result may deviate from the optimal result by
10 times the machine epsilon and sometimes even more:

.. code-block:: python

   P = [(642123.5528970208, 5424489.146461355),
        (651592.349934072, 5424969.380667617),
        (642591.1068130962, 5425775.320365907),
        (646380.0282527813, 5418648.987550308),
        (648098.891235107, 5426586.3920675),
        (650011.5835629451, 5426132.820254512),
        (650297.6960375579, 5419125.777007122),
        (645249.2122321032, 5421055.739722816),
        (645333.9125837489, 5426228.852409409)]

   mb = miniball.Miniball(P)
   if not mb.is_valid():
       print('Possibly invalid!')
   print('Relative error', mb.relative_error())

If this is a problem for you, shifting the input towards (0,0) may help:

.. code-block:: python

   minx = min(P, key=lambda p: p[0])[0]
   miny = min(P, key=lambda p: p[1])[1]

   P = [(p[0] - minx, p[1] - miny) for p in P]

   mb = miniball.Miniball(P)
   if not mb.is_valid():
       print('Possibly invalid!')
   print('Relative error', mb.relative_error())

__ http://www.inf.ethz.ch/personal/gaertner/miniball.html
