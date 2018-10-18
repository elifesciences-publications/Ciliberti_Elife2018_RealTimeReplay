
To build cython interface locally:

1. cd to compressed_decoder/gmmcompression
2. python setup.py build_ext --inplace

To import:

import onlinedecoding.gmmcompression as gmm

-------

To compile test programs:

gcc -O2 -DHAVE_INLINE simple_test_2d.c mixture_c.c component_c.c covariance_c.c -lgsl -lgslcblas -lm -o simple_test_2d
gcc -O2 -DHAVE_INLINE simple_test.c mixture_c.c component_c.c covariance_c.c -lgsl -lgslcblas -lm -o simple_test

To compile test programs against ATLAS libraries

gcc -O2 -DHAVE_INLINE simple_test_2d.c mixture_c.c component_c.c covariance_c.c -lgsl -lcblas -latlas -lm -o simple_test_2d
gcc -O2 -DHAVE_INLINE simple_test.c mixture_c.c component_c.c covariance_c.c -lgsl -lcblas -latlas -lm -o simple_test
