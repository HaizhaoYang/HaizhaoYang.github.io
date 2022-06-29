
This set of MATLAB files contain an implementation of the
block Chebyshev-Davidson with inner-outer restart algorithms
described in the paper 

@Article{bchebdav,
  author =	 {Y. Zhou},
  title =	 {A block {Chebyshev-Davidson} method with inner-outer
                  restart for large eigenvalue problems},
  journal =	 {J. Comput. Phys.},
  year =	 2010,
  volume =	 229,
  number =	 24,
  pages =	 {9188-9200},
  doi =		 {{doi:10.1016/j.jcp.2010.08.032}},
}

This paper used "opts.filter=1", the non-scaled filter.
(In March 2010, a simple scaled filter "opts.filter=2" is added,
which turned out to have slightly better performance than the
results reported in the paper.)

Both the paper and the code are available at
http://faculty.smu.edu/yzhou/code.htm

Some matrix data related to the real-space DFT calculation are available at
http://www.cise.ufl.edu/research/sparse/matrices/PARSEC/

--------------------------------------------------------------------------------

The main algorithm is implemented in  "bchdav.m".

For usage details, check the code or type "help bchdav" at the MATLAB prompt.
(The file  "sample.m"  provides two simple sample calls to "bchdav.m".)

This code is in development stage; any comments or bug reports are very welcome.

Contacts:  yzhou@smu.edu

--------------------------------------------------------------------------------
Copyright (2010):  Y. Zhou   

bchdav is free software, distributed under the terms of the GNU 
General Public License 3.0.  ( http://www.gnu.org/licenses/gpl.html )

Permission to use, copy, modify,  and distribute this software for any
purpose  without fee  is  hereby granted,  provided  that this  entire
notice is included in all copies  of any software which is or includes
a  copy or  modification of  this software  and in  all copies  of the
supporting  documentation for  such software.  This software  is being
provided  "as  is",  without  any  express  or  implied  warranty.  In
particular, the  author(s) do not make any  representation or warranty
of any  kind concerning  the merchantability of  this software  or its
fitness for any particular purpose."

--------------------------------------------------------------------------------


