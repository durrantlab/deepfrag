Changes
=======

1.0.3
-----

* CLI parameters `--cx`, `--cy`, `--cz`, `--rx`, `--ry`, and `--rz` can now be
  floats (not just integers). We recommend specifying the exact atomic
  coordinates of the connection and removal points.
* Fixed a bug that caused the `--full` parameter to throw an error when
  performing fragment addition (but not fragment replacement) using the CLI
  implementation.
* Minor updates to the documentation.

1.0.2
-----

* Added a CLI implementation of the program. See `README.md` for details.
* Added a version number and citation to the program output.

1.0.1
-----

* Removed open-babel dependency.
* Added option (`cpu_gridify`) to improve use on CPUs when no GPU is
  available.
* Updated `data/README.md` with new location of data files.
* Fixed a config import.

1.0
---

Original version.
