/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Declarations of data structures for storing the input      *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>

/* Sparse vector datatype */
typedef struct sparse_t{
    float* x; /* stores the non zero values */
    int* idx; /* stores the indices of the non zero values */
    int nz;   /* number of non zero values */
}sparse_t;

typedef struct dataset_t{
    sparse_t* example; /* array of examples */
    int* target;       /* Target values */
    int nfeat;         /* number of features */
    int nex;           /* number of examples */
    float sparsity;    /* fraction of nonzero features in a typical vector */
}dataset_t;

void loadData(const char* name, dataset_t* d);
int getDimensions(FILE* fp, int* examples, int* features);
int readExample(FILE* fp, int maxline, int maxfeat, sparse_t* s, int* target);
void freeData(dataset_t* d); 
void clipvectors(int inputs, sparse_t* v, int len);

#endif
