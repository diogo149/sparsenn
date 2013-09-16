/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Learning module                                            *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

/* A Debugging routine that raises a signal when
 * things like NaNs start appearing. This doesn't
 * happen with backpropagation (unless the input 
 * is in a really bad scale) but with more 
 * sophisticated algorithms which may blow up when
 * their assumptions are violated. Raising a signal
 * allows for easy tracing of the problem when run 
 * inside gdb.
 */ 
#ifndef NDEBUG
#define _GNU_SOURCE
#include <fenv.h>
int catchfpe(){ 
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    return 1;
}
#endif

#include "dataset.h"
#include "metrics.h"
#include "nnet.h"
#include <getopt.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Generate and store a permutation of 1,2,...,n in a */
void shuffle(int* a, int n){
    int r,i;
    int t;
    for(i=n-1; i>0; i--){
        r=rand()%(i+1);
        t=a[r]; a[r]=a[i]; a[i]=t;
    }
}

int main(int argc, char* argv[]){
    nnet_t n;
    dataset_t train,stop;
    float *pt,*ps;
    float at,as,et,es,rt,rs;
    float maxacc=0;
    float maxauc=0;
    float minrms=2;
    float rate=0.05;
    int *perm;
    int epochs=1000;
    int hidden=16;
    int period=10;
    int option;
    int i;
    char* prefix;
    char modelacc[1024];
    char modelrms[1024];
    char modelauc[1024];

    const char* help="Usage: %s [options] trainingset validationset model\nAvailable options:\n\
            -e <int>  : number of epochs (default: 1000)\n\
            -h <int>  : number of hidden units (default: 16)\n\
            -p <int>  : print performance every so many epochs: (default: 10)\n\
            -r <float>: learning rate (default: 0.05)\n";

    assert(catchfpe());

    while((option=getopt(argc,argv,"e:h:p:r:"))!=EOF){
        switch(option){
            case 'e': epochs=atoi(optarg); break;
            case 'h': hidden=atoi(optarg); break;
            case 'p': period=atoi(optarg); break;
            case 'r': rate=atof(optarg); break;
            case '?': fprintf(stderr,help,argv[0]); exit(1); break;
        }
    }

    if(argv[optind]==0 || argv[optind+1]==0 || argv[optind+2]==0){
        fprintf(stderr,help,argv[0]);
        exit(1);
    }

    loadData(argv[optind], &train);
    loadData(argv[optind+1], &stop);
    pt=malloc(sizeof(float)*train.nex);
    ps=malloc(sizeof(float)*stop.nex);
    perm=malloc(sizeof(int)*train.nex);
    for(i=0; i<train.nex; i++){
        perm[i]=i;
    }

    prefix = argv[optind+2];
    sprintf(modelacc,"%s.acc",prefix);
    sprintf(modelrms,"%s.rms",prefix);
    sprintf(modelauc,"%s.auc",prefix);

    srand(time(0));
    clipvectors(train.nfeat, stop.example, stop.nex);
    rate/=train.nex;

    maxacc=0;
    maxauc=0;
    minrms=2;

    createnet(&n, &train, hidden, rate);
    for(i=0; i<epochs; i++){
        shuffle(perm,train.nex);
        trainnet(&n, &train, perm);
        if(i % period == 0){
            testnet(&n, &train, pt);
            testnet(&n, &stop, ps);
            at=acc(pt, train.target, train.nex);
            et=rms(pt, train.target, train.nex);
            rt=auc(pt, train.target, train.nex);
            as=acc(ps, stop.target, stop.nex);
            es=rms(ps, stop.target, stop.nex);
            rs=auc(ps, stop.target, stop.nex);
            printf("pass %d tacc %.5f sacc %.5f trms %.5f srms %.5f tauc %.5f sauc %.5f ",i,at,as,et,es,rt,rs);
            if(as>maxacc){
                printf("( ");
                maxacc=as;
                savenet(modelacc,&n);
            }
            else
                printf(") ");
            if(es<minrms){
                printf("[ ");
                minrms=es;
                savenet(modelrms,&n);
            }
            else
                printf("] ");
            if(rs>maxauc){
                printf("{ ");
                maxauc=rs;
                savenet(modelauc,&n);
            }
            else
                printf("} ");
            printf("\n");
        }
    }
    free(ps);
    free(pt);
    freeData(&train);
    freeData(&stop);
    destroynet(&n);
    return 0;
}
