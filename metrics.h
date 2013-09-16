/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Declarations of functions computing evaluation metrics     *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/
#ifndef METRICS_H
#define METRICS_H


float acc(float *predictions, int *targets, int n);
float rms(float *predictions, int *targets, int n);
float auc(float *predictions, int *targets, int n);

#endif /* METRICS_H */
