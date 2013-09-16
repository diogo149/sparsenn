/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Declaration of neural net data type and related functions. *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#ifndef NNET_H
#define NNET_H

#include "dataset.h"

typedef struct nnet_t{
    float** W1; /* first layer weights */
    float* b1; /* first layer biases  */
    float* a1; /* inputs to activation function of the hidden units */
    float* x1; /* outputs of activation function of the hidden units */
    float* g1; /* respective derivatives  */
    float* d1; /* error in first layer */
    float* W2; /* second layer weights */
    float b2; /* second layer bias */
    float a2; /* input to activation function of the output unit */
    float x2; /* output of activation function of the output unit  */
    float g2; /* respective derivative  */
    float d2; /* error in second layer */
    float eta; /* learning rate */
    int inputs;
    int hidden;
}nnet_t;

void createnet(nnet_t* n, dataset_t* d, int hid, float rate);

void destroynet(nnet_t* n);

void savenet(const char* name, nnet_t* n);
void loadnet(const char* name, nnet_t* n);

void activation(float* p, float* f, float* g, int n);

void train(nnet_t* n, sparse_t* v, int target);

float value(nnet_t* n, sparse_t* v);

void clipvectors(int inputs, sparse_t* v, int len);

void trainnet(nnet_t* n, dataset_t* d, int *perm);

void testnet(nnet_t* n, dataset_t* d, float *p);
#endif /* NNET_H */
