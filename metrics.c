/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Functions computing evaluation metrics                     *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/
 
#include "metrics.h"
#include <math.h>
#include <stdlib.h>

static int partition(int p, int r, float *pred, int *target)
{
    int i, j;
    float x, tempf;
    int tempi;

    x = pred[p];
    i = p - 1;
    j = r + 1;
    while (1) {
        do
            j--;
        while (!(pred[j] <= x));
        do
            i++;
        while (!(pred[i] >= x));
        if (i < j) {
            tempf = pred[i];
            pred[i] = pred[j];
            pred[j] = tempf;
            tempi = target[i];
            target[i] = target[j];
            target[j] = tempi;
        } else
            return (j);
    }
}


static void quicksort(int p, int r, float *pred, int *target)
{
    int q;
    if (p < r) {
        q = partition(p, r, pred, target);
        quicksort(p, q, pred, target);
        quicksort(q + 1, r, pred, target);
    }
}

float auc(float *predictions, int *targets, int n)
{
    int i;
    int item = 0;
    int no_item = n;
    int total_true_1 = 0;
    int total_true_0 = 0;
    float *pred = malloc(no_item*sizeof(float));
    float *fraction = malloc(no_item*sizeof(float));
    int *target = malloc(no_item*sizeof(int));
    double tt, tf, ft, ff;
    double sens, spec, tpf, fpf, tpf_prev, fpf_prev;
    double roc_area;

    float threshold = 0;
    for (i = 0; i < no_item; i++) {
        threshold += targets[i];
    }
    threshold = threshold / no_item;

    for (i = 0; i < no_item; i++) {
        if (targets[i] > threshold) {
            target[i] = 1;
            total_true_1++;
        } else {
            target[i] = 0;
            total_true_0++;
        }
        pred[i] = predictions[i];
    }

    quicksort(0, (no_item - 1), pred, target);

    while (item < no_item) {
        int begin = item;
        int count = 1;
        int no_poz;
        if (target[item] == 1)
            no_poz = 1;
        else
            no_poz = 0;
        while ((item < no_item - 1) && (fabs(pred[item] - pred[item + 1]) < 1e-15)) {
            item++;
            count++;
            if (target[item] == 1)
                no_poz++;
        }
        for (i = begin; i <= item; ++i)
            fraction[i] = no_poz * 1.0 / count;
        item++;
    }


    tt = 0;
    tf = total_true_1;
    ft = 0;
    ff = total_true_0;

    sens = ((double) tt) / ((double) (tt + tf));
    spec = ((double) ff) / ((double) (ft + ff));
    tpf = sens;
    fpf = 1.0 - spec;
    roc_area = 0.0;
    tpf_prev = tpf;
    fpf_prev = fpf;

    for (item = no_item - 1; item > -1; item--) {
        tt += fraction[item];
        tf -= fraction[item];
        ft += 1 - fraction[item];
        ff -= 1 - fraction[item];
        sens = ((double) tt) / ((double) (tt + tf));
        spec = ((double) ff) / ((double) (ft + ff));
        tpf = sens;
        fpf = 1.0 - spec;
        roc_area += 0.5 * (tpf + tpf_prev) * (fpf - fpf_prev);
        tpf_prev = tpf;
        fpf_prev = fpf;
    }
    return roc_area;
}

float rms(float *predictions, int *targets, int n)
{

    int i;
    int no_item = n;
    float rms,diff;

    rms=0.0;

    for (i = 0; i < no_item; i++) {
        diff=(predictions[i]-targets[i]);
        rms+=diff*diff;
    }
    rms=sqrtf(rms/no_item);

    return rms;
}

float acc(float *predictions, int *targets, int n)
{
    int i;
    int no_item = n;
    float threshold = 0.5;
    float acc;
    for (i = 0; i < no_item; i++) {
        if(targets[i]<0){
            threshold=0.0;
            break;
        }
    }

    acc=0.0;
    for(i = 0; i < no_item; i++){
        if((predictions[i]-threshold)*(targets[i]-threshold) > 0)
            acc+=1.0;
    }
    acc=acc/no_item;
    return acc;
}
