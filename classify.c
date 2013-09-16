/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Classification module.                                     *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#include "dataset.h"
#include "metrics.h"
#include "nnet.h"
#include <getopt.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){
    nnet_t n;
    dataset_t test;
    float *pt;
    int option;
    int i;
    FILE* fp;

    const char* help="Usage: %s testset model predictions\n";

    while((option=getopt(argc,argv,""))!=EOF){
        switch(option){
            case '?': fprintf(stderr,help,argv[0]); exit(1); break;
        }
    }

    if(argv[optind]==0 || argv[optind+1]==0 || argv[optind+2]==0){
        fprintf(stderr,help,argv[0]);
        exit(1);
    }

    loadData(argv[optind], &test);
    loadnet(argv[optind+1], &n);
    fp=fopen(argv[optind+2],"w");
    if(fp==NULL){
        fprintf(stderr,"Could not open output file: %s\n",argv[optind+2]);
        exit(1);
    }
    clipvectors(n.inputs, test.example, test.nex);
    pt=malloc(sizeof(float)*test.nex);
    testnet(&n, &test, pt);

    for(i=0; i<test.nex; i++){
        fprintf(fp,"%f\n",pt[i]);
    }
    fclose(fp);
    free(pt);
    freeData(&test);
    destroynet(&n);
    return 0;
}
