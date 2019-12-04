/**
 * \file misc.cpp
 * \brief other routines needed to solve heat equation.
 *
 * \date 17-dec-2009
 */
#include <string.h> // for memset
#include <assert.h>
#include <stdlib.h> // for rand
#include <stdio.h> // for printf

#include "param.h"
#include "misc.h"

/** 
 * initialize 2D data (\f$ t=0 \f$)
 * 
 * @param data real_t array
 */
void initCondition2D (real_t *data)
{
  memset(data,0,NX*NY*sizeof(real_t));

  // simple init condition whose analytical evolution is known
  if (!PROBLEM.compare("sinus")) {
    for( unsigned int j = 0; j < NY; ++j)
      for( unsigned int i = 0; i < NX; ++i) {
	unsigned int index = j*NX+i;
	data[index] = static_cast<real_t>(sin(M_PI/(NX-1)*i)*sin(M_PI/(NY-1)*j));
      }
  }
 
  // square box of size NX/2 , NY/2
  else if ( !PROBLEM.compare("square") ) {
    for( unsigned int j = NY/4; j < 3*NY/4; ++j)
      for( unsigned int i = NX/4; i < 3*NX/4; ++i) {
	unsigned int index = j*NX+i;
	data[index] = (real_t) 1.0;
      }
  }

  // put 1's on one border
  else if (!PROBLEM.compare("border")) {
    for (unsigned int i=0; i<NX; ++i) {
      // first column (j=0)
      unsigned int index = i;
      data[index] = 1.0f;
      // second column (j=1)
      index += NX;
      data[index] = 1.0f;
    }
  }

  // random field
  else if (!PROBLEM.compare("random")) {
    srand(123);
    for( unsigned int j = 1; j < NY-1; ++j) {
      for( unsigned int i = 1; i < NX-1; ++i) {
	unsigned int index = j*NX+i;
	data[index] = 1.0*rand()/RAND_MAX; 
      }
    }
  }

} // initCondition2D

/** 
 * set border condition for 2D buffer (Dirichlet condition)
 * 
 * @param data 
 * @param borderCond
 */
void setBorderCond2D (real_t *data, real_t borderCond)
{
  for (unsigned int i=0; i<NX; ++i) {
    // first column (j=0)
    unsigned int index = i;
    data[index] = borderCond;
    
    // last column (j=NY-1)
    index = (NY-1)*NX+i;
    data[index] = borderCond;
  }

  for (unsigned int j=0; j<NY; ++j) {
    // first row (i=0)
    unsigned int index = j*NX;
    data[index] = borderCond;
    
    // last row (i=NX-1))
    index = j*NX+NX-1;
    data[index] = borderCond;
  }
} // setBorderCond2D

/** 
 * initialize 3D data (\f$ t=0 \f$)
 * 
 * @param data real_t array
 */
void initCondition3D (real_t *data)
{

  // simple init condition whose analytical evolution is known
  if ( !PROBLEM.compare("sinus") ) {
    for( unsigned int k = 0; k < NZ; ++k)
      for( unsigned int j = 0; j < NY; ++j)
	for( unsigned int i = 0; i < NX; ++i) {
	  unsigned int index = (k*NY+j)*NX+i;
	  data[index] = static_cast<real_t>(sin(M_PI/(NX-1)*i)*sin(M_PI/(NY-1)*j)*sin(M_PI/(NZ-1)*k));
	}
  }

  // square box
  else if ( !PROBLEM.compare("square") ) {
    memset(data,0,NX*NY*NZ*sizeof(real_t));
    assert(NX>20);
    assert(NY>20);
    assert(NZ>20);
    for( unsigned int k = NZ/2-10; k < NZ/2+10; ++k)
      for( unsigned int j = NY/2-10; j < NY/2+10; ++j)
	for( unsigned int i = NX/2-10; i < NX/2+10; ++i) {
	  unsigned int index = (k*NY+j)*NX+i;
	  data[index] = 1.0f;
	}
  }

  // random field
  else if (!PROBLEM.compare("random")) {
    srand(123);
    for( unsigned int k = 1; k < NZ-1; ++k) {
      for( unsigned int j = 1; j < NY-1; ++j) {
	for( unsigned int i = 1; i < NX-1; ++i) {
	  unsigned int index = (k*NY+j)*NX+i;
	  data[index] = 1.0*rand()/RAND_MAX; 
	}
      }
    }
  }
  
} // initCondition3D

/** 
 * set border condition for 3D buffer (Dirichlet condition)
 * 
 * @param data 
 * @param borderCond
 */
void setBorderCond3D (real_t *data, real_t borderCond)
{

  for (unsigned int j=0; j<NY; ++j) 
    for (unsigned int i=0; i<NX; ++i) {
    // plane k=0
      unsigned int index = j*NX+i;
      data[index] = borderCond;
    
      // plane k=NZ-1
      index = ((NZ-1)*NY+j)*NX+i;
      data[index] = borderCond;
    }

  for (unsigned int k=0; k<NZ; ++k)
    for (unsigned int j=0; j<NY; ++j) {
      
      // plane i=0
      unsigned int index = (k*NY+j)*NX;
      data[index] = borderCond;
      
      // plane i=NX-1
      index = (k*NY+j)*NX+NX-1;
      data[index] = borderCond;
    }

  for (unsigned int k=0; k<NZ; ++k)
    for (unsigned int i=0; i<NX; ++i) {
      
      // plane j=0
      unsigned int index = (k*NY)*NX+i;
      data[index] = borderCond;
      
      // plane k=NY-1
      index = (k*NY+NY-1)*NX+i;
      data[index] = borderCond;
    } 
} // setBorderCond3D
