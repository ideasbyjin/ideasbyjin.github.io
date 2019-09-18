---
layout: post
title: "Building Python modules using C++"
date: 2019-01-28
categories: code
---

Python is an amazing programming language for lots of applications, particularly for bioinformatics.
One of the potential downsides to using Python (apart from whitespace, non-static typing) is its speed.
It's certainly faster than languages such as R, but it's nowhere near the level of C/C++. In fact, many
Python modules are already written in C/C++ (such as NumPy) but it might be practical to have your own
C/C++ code to interface with your Python objects. However, getting them to work together is surprisingly
difficult. Luckily, the [Boost library](https://www.boost.org) is a great way to interface code, and can
also work with NumPy objects (e.g. NumPy arrays) to give you versatility. Below is a set of example files,
along with the compilation command.

Make sure you have BOOST built on your machine, even if it is a local installation. Here's a sample C++ file
calculating the Euclidean norm of a NumPy array, as `vectors.cpp`:

{% highlight cpp %}
#include<cmath>
#include<boost/python/module.hpp>
#include<boost/python/def.hpp>
#include<boost/python/extract.hpp>
#include<boost/python/numpy.hpp>

using namespace boost::python;
namespace np = boost::python::numpy;

/* Define a C++ function as you would */
double eucnorm(np::ndarray axis){
  const int n = axis.shape(0);   
  double norm = 0.0;             
  for(int i = 0; i < n; i++){ 
    double A = extract<double>(axis[i]);
    norm += A*A;
  }
  return sqrt(norm);
}

/* Define your module name within BOOST_PYTHON_MODULE */ 
BOOST_PYTHON_MODULE(vectors){ 
  /* Initialise numpy */
  np::initialize();
  /* Define your function, named eucnorm */
  def("eucnorm", eucnorm);
}
{% endhighlight %}

Compiling this was possibly the toughest part for me, but this is the way to do it:

{% highlight bash %}
g++ vectors.cpp -shared -fpic -I $PYTHONPATH\
-I $BOOST_ROOT -L $BOOST_LIB_PATH -lboost_numpy -lboost_python -o vectors.so
{% endhighlight %}

In Python, the module file in `vectors.so` is called by:

{% highlight python %}
# vectors.py
from vectors import *
import numpy as np
v = np.array([1,1,1])
N = eucnorm(v) # 1.7320508075688772
{% endhighlight %}

And that's a very, very simple tutorial of how to mesh C++ and Python together!