/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Functions to create, train and test a neural net.          *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/
 
#include "dataset.h"
#include "nnet.h"
#include <cblas.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* generate a random value in the interval [-x,x] */  
float symrand(float x){
    return 2.0f*x*rand()/(RAND_MAX+1.0f)-x;
}

/* Create a neural network with enough inputs to handle the
 * examples in dataset d, hid hidden units and learning rate
 * equal to rate. Store the network in n 
 */ 
void createnet(nnet_t* n, dataset_t* d, int hid, float rate){
    int i;
    float q,r;

    n->inputs=d->nfeat;
    n->hidden=hid;

    /* These choices are loosely based on the 
     * efficient backprop paper by LeCun et. al. 
     */
    q=sqrtf(0.003f/(d->sparsity*n->inputs+1.0f));
    r=sqrtf(0.003f/(hid+1.0f));

    /* We store W1 in transposed form */
    n->W1 = malloc(sizeof(float*)*n->inputs);
    n->W1[0] = malloc(sizeof(float)*n->inputs*n->hidden);
    for(i=1; i<n->inputs; i++)
        n->W1[i]=n->W1[0]+i*n->hidden;

    n->b1 = malloc(sizeof(float)*n->hidden);
    n->a1 = malloc(sizeof(float)*n->hidden);
    n->x1 = malloc(sizeof(float)*n->hidden);
    n->g1 = malloc(sizeof(float)*n->hidden);
    n->d1 = malloc(sizeof(float)*n->hidden);
    n->W2 = malloc(sizeof(float)*n->hidden);
    n->eta = rate;
    for(i=0; i<n->inputs*n->hidden; i++){
        n->W1[0][i] = symrand(q);
    }
    for(i=0; i<n->hidden; i++){
        n->b1[i] = symrand(q); 
        n->W2[i] = symrand(r); 
    }
    n->b2 = symrand(r);
}

/* Saves the network n to a file */
void savenet(const char* name, nnet_t* n){
    FILE *fp;
    fp=fopen(name,"w");
    if(fp==NULL){
        fprintf(stderr,"Could not write to file %s\n",name);
        return;
    }
    fprintf(fp,"inputs %d\n",n->inputs);
    fprintf(fp,"hidden %d\n",n->hidden);
    fprintf(fp,"rate %g\n",n->eta);
    fwrite(n->W1[0],sizeof(float),n->inputs*n->hidden,fp);
    fwrite(n->b1,sizeof(float),n->hidden,fp);
    fwrite(n->W2,sizeof(float),n->hidden,fp);
    fwrite(&n->b2,sizeof(float),1,fp);
    fclose(fp);
}

/* Loads a network from a file to memory */
void loadnet(const char* name, nnet_t* n){
    FILE *fp;
    int i,c;
    fp=fopen(name,"r");
    if(fp==NULL){
        fprintf(stderr,"Could not load file %s\n",name);
        return;
    }

    fscanf(fp,"%*s%d",&n->inputs);
    fscanf(fp,"%*s%d",&n->hidden);
    fscanf(fp,"%*s%f",&n->eta);
    do
        c=fgetc(fp);
    while(c!='\n');

    n->W1 = malloc(sizeof(float*)*n->inputs);
    n->W1[0] = malloc(sizeof(float)*n->inputs*n->hidden);
    for(i=1; i<n->inputs; i++)
        n->W1[i]=n->W1[0]+i*n->hidden;

    n->b1 = malloc(sizeof(float)*n->hidden);
    n->a1 = malloc(sizeof(float)*n->hidden);
    n->x1 = malloc(sizeof(float)*n->hidden);
    n->g1 = malloc(sizeof(float)*n->hidden);
    n->d1 = malloc(sizeof(float)*n->hidden);
    n->W2 = malloc(sizeof(float)*n->hidden);

    fread(n->W1[0],sizeof(float),n->inputs*n->hidden,fp);
    fread(n->b1,sizeof(float),n->hidden,fp);
    fread(n->W2,sizeof(float),n->hidden,fp);
    fread(&n->b2,sizeof(float),1,fp);
    fclose(fp);
}

/* Releases the memory held by a network */
void destroynet(nnet_t* n){
    free(n->W1[0]);
    free(n->W1);
    free(n->b1);
    free(n->a1);
    free(n->x1);
    free(n->g1);
    free(n->d1);
    free(n->W2);
}

/* Activation function and derivative(s).
 * x is an array of n values whose activation and derivate(s) will be computed 
 * f will store the activation and g will store the first derivative.
 * The commented out parts involving h are computing the second derivative
 * of the activation function. This is not needed by backpropagation.
 * The activation function used is the following:
 * f(x)=sqrt(3)*tanh(atanh(1/sqrt(3))*x)
 * This function has the following properties
 * (a) f(1) = 1, f(-1) = -1
 * (b) the second derivative of f takes its extreme values at 1 and -1
 * See the Efficient Backprop paper by LeCun et. al. on why this is desirable
 * tanh is an expensive function so instead we use an approximation based on
 * an eight degree Chebychev polynomial and a constant function (when the 
 * input is larger than 10. Finally the value of the derivative is biased
 * by 0.01 to avoid flat spots.
 */
void activation(float* x, float *f, float * g, /* float * h,*/ int n){
    float d,dd,y,y2,z,t,u,s;
    int i;
    for(i=0; i<n; i++){
        if(x[i]<0){
            z=-x[i];
            s=-1.0f;
        }
        else{
            z=x[i];
            s=1.0f;
        }
        if(z>10){
            f[i] = 1.732050807568877f*s;
            g[i] = 0.01f;
            /* h[i] = 0.0f; */
            continue;
        }
        y  = 0.2f*z-1.0f;
        y2 = 2.0f*y;
        d  = 0.0f;
        dd = 0.0f;
        /* chebyshev 8 */
        dd = y2*d - dd + 0.00366079966971855f;
        d  = y2*dd - d - 0.00609954274163606f;
        dd = y2*d - dd + 0.00292879407703366f;
        d  = y2*dd - d + 0.0154392072571603f;
        dd = y2*d - dd - 0.0620003898943956f;
        d  = y2*dd - d + 0.144460219650769f;
        dd = y2*d - dd - 0.251652398279877f;
        d  = y2*dd - d + 0.347342011478419f;

        t = s*(y*d - dd + 0.806857360076642);
        u = 1-t*t;
        f[i] = 1.732050807568877f*t;
        g[i] = 1.14051899445142f*u+0.01f;
        /* h[i] = -1.502015496335549f*u*t; */
    }
}

/* Trains a network by presenting an example and 
 * adjusts the weights by stochastic gradient 
 * descent to reduce a squared hinge loss
 */
void train(nnet_t* n, sparse_t* v, int target){
    int i;
    /* Forward pass */
    cblas_scopy(n->hidden,n->b1,1,n->a1,1);
    for(i=0; i<v->nz; i++){
        cblas_saxpy(n->hidden, v->x[i], n->W1[v->idx[i]], 1, n->a1, 1);
    }
    activation(n->a1,n->x1,n->g1,n->hidden);
    n->a2 = n->b2 + cblas_sdot(n->hidden, n->W2, 1, n->x1, 1);
    activation(&n->a2,&n->x2,&n->g2,1);
    if(target*n->x2 > 1)
        /* Hinge loss, no error -> no need to backpropagate */
        return;
    /* Backward pass */
    n->d2 = (target-n->x2)*n->g2;
    cblas_scopy(n->hidden,n->W2,1,n->d1,1);
    for(i=0; i<n->hidden; i++)
        n->d1[i] *= n->d2*n->g1[i];
    n->b2 += n->eta*n->d2;
    cblas_saxpy(n->hidden, n->eta*n->d2, n->x1, 1, n->W2, 1);
    cblas_saxpy(n->hidden, n->eta, n->d1, 1, n->b1, 1);
    /* Sparse inputs imply sparse gradients.
     * This update saves a lot of computation
     * compared to general purpose neural net
     * implementations.
     */
    for(i=0; i<v->nz; i++){
        cblas_saxpy(n->hidden, n->eta*v->x[i], n->d1, 1, n->W1[v->idx[i]], 1);
    }
}

/* Given an input vector v, compute the output of the network. */
float value(nnet_t* n, sparse_t* v){
    int i;
    cblas_scopy(n->hidden,n->b1,1,n->a1,1);
    for(i=0; i<v->nz; i++){
        cblas_saxpy(n->hidden, v->x[i], n->W1[v->idx[i]], 1, n->a1, 1);
    }
    activation(n->a1,n->x1,n->g1,n->hidden);
    n->a2 = n->b2;
    n->a2 += cblas_sdot(n->hidden, n->W2, 1, n->x1, 1);
    activation(&n->a2,&n->x2,&n->g2,1);
    return n->x2;
}

/* Run one epoch of training with a given dataset.
 * Perm stores a permutation of the examples. 
 * Shuffling the examples before each epoch helps 
 * the convergence to stochastic gradient descent.
 */
void trainnet(nnet_t* n, dataset_t* d, int* perm){
    int i;
    for(i=0; i<d->nex; i++)
        train(n, &(d->example[perm[i]]), d->target[perm[i]]);
}

/* Get the predictions of the net for the examples
 * in dataset d and store them in p.
 */
void testnet(nnet_t* n, dataset_t* d, float *p){
    int i;
    for(i=0; i<d->nex; i++)
        p[i]=value(n, &(d->example[i]));
}
