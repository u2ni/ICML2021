#include <Python.h>
#include "Miniball.hpp"

static PyObject* miniball_miniball(PyObject *self, PyObject *args)
{
  // Convert input to C++
  PyObject *list;
  if (!PyArg_ParseTuple(args, "O", &list))
    return NULL;

  PyObject *iter = PyObject_GetIter(list);
  if (!iter)
    return NULL;

  int size = PyObject_Size(list);
  PyObject *next = PyIter_Next(iter);
  int dimension = PyObject_Size(next);

  double** points = new double*[PyObject_Size(list)];

  int i = 0;
  while (next) {
    PyObject *point_iter = PyObject_GetIter(next);
    points[i] = new double[dimension];
    PyObject *x = PyIter_Next(point_iter);
    int n = 0;
    while (x) {
      double xx = PyFloat_AsDouble(x);
      points[i][n] = xx;
      Py_DECREF(x);
      x = PyIter_Next(point_iter);
      n++;
    }
    Py_DECREF(point_iter);
    Py_DECREF(next);
    next = PyIter_Next(iter);
    i++;
  }
  Py_DECREF(iter);
  // Do the math
  typedef double* const* PointIterator;
  typedef const double* CoordIterator;

  typedef Miniball::
    Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> >
    MB;

  MB mb(dimension, points, points+size);

  // Convert result to python
  PyObject *result = PyList_New(6);

  const double* center = mb.center();
  PyObject *pycenter = PyList_New(dimension);
  for (int n = 0; n < dimension; n++)
    PyList_SetItem(pycenter, n, PyFloat_FromDouble(center[n]));
  PyList_SetItem(result, 0, pycenter);

  PyList_SetItem(result, 1, PyFloat_FromDouble(mb.squared_radius()));

  double suboptimatily;
  PyList_SetItem(result, 2, PyFloat_FromDouble(mb.relative_error(suboptimatily)));
  PyList_SetItem(result, 3, PyFloat_FromDouble(suboptimatily));
  PyList_SetItem(result, 4, PyBool_FromLong(mb.is_valid()));
  PyList_SetItem(result, 5, PyFloat_FromDouble(mb.get_time()));
  // Clean up
  for (int j = 0; j < size; j++) {
    delete[] points[j];
  }
  delete points;  

  return result;
}

static PyMethodDef MiniballMethods[] = {
  {"miniball", miniball_miniball, METH_VARARGS, "Compute miniball"},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef miniballmodule = {
  PyModuleDef_HEAD_INIT,
  "bindings",
  "Python bindings to Bernd Gaertners miniball software (V3.0)",
  -1,
  MiniballMethods
};

PyMODINIT_FUNC PyInit_bindings(void)
{
  return PyModule_Create(&miniballmodule);
}
