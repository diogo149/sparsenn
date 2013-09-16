/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Functions to read a dataset into memory.                   *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataset.h"

int getDimensions(FILE* fp, int* examples, int* features){
    char buf[4096];
    int i,buflen,previous,total,max,target,len,lastfeature;
    char* line;
    char *comment,*colon,*space,*tab;

    previous=-1;
    total=0;
    max=0;
    /* find maximum line length */
    rewind(fp);
    while((buflen=fread(buf,sizeof(char),4096,fp))!=0){
        for(i=0; i<buflen; i++,total++){
            if(buf[i]=='\n'){
                len=total-previous;
                previous=total;
                if(max<len) max=len;
            }
        }
    }

    max+=4; /* Just in case I was sloppy */
    line=malloc(max*sizeof(char));

    rewind(fp);
    *examples = 0;
    *features = 0;
    while(fgets(line,max,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d",&target)==EOF)
            /* The line was a comment */
            continue;
        *examples += 1;
        colon=strrchr(line,':');
        if(colon==NULL) /* This can happen when the zero vector is in the data */
            lastfeature=1;
        else{
            *colon = '\0';
            space = strrchr(line,' ');
            space = space == NULL ? line : space;
            tab = strrchr(space,'\t');
            tab = tab == NULL ? space : tab;
            sscanf(tab,"%d",&lastfeature);
            if(*features<lastfeature)
                *features=lastfeature;
        }
    }
    rewind(fp);
    /* This is because the array of features is starting from 0 */
    *features += 1;
    free(line);
    return max;	
}

int getSizes(FILE* fp, int maxline, dataset_t* d){
    int target,offset,len,ex,nz,total;
    char* line;
    char* comment;

    total=0;
    line=malloc(maxline*sizeof(char));
    rewind(fp);

    ex=0;
    while(fgets(line,maxline,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d%n",&target,&len)==EOF)
            /* The line was a comment */
            continue;
        nz=0;
        for(offset=len; line[offset]!='\0'; offset++){
            if(line[offset]==':')
                nz+=1;
        }
        d->example[ex].nz=nz;
        total+=nz;
        ex+=1;
    }
    d->sparsity=total/(float)(d->nfeat*d->nex);

    free(line);
    return total;
}

void readExamples(FILE* fp, int maxline, dataset_t* d){
    int i,ex,target,offset,feat,len;
    float val;
    char* line;
    char* comment;

    line=malloc(maxline*sizeof(char));
    rewind(fp);
    ex=0;
    while(fgets(line,maxline,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d%n",&target,&len)==EOF)
            /* The line was a comment */
            continue;
        d->target[ex] = target <= 0 ? -1 : 1;
        for(i=0,offset=len; sscanf(line+offset,"%d:%f%n",&feat,&val,&len)>=2; i+=1,offset+=len){
            d->example[ex].idx[i]=feat;
            d->example[ex].x[i]=val;
        }
        ex+=1;
    }
    free(line);
}

int readExample(FILE* fp, int maxline, int maxfeat, sparse_t* s, int* target){
    int i,nz,offset,feat,len;
    float val;
    char* line;
    char* comment;

    line=malloc(maxline*sizeof(char));

    while(fgets(line,maxline,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d%n",target,&len)==EOF)
            /* The line was a comment */
            continue;
        *target = *target <=0 ? -1 : 1;
        nz=0;
        for(offset=len; line[offset]!='\0'; offset++){
            if(line[offset]==':')
                nz+=1;
        }
        s->nz=nz;
        for(i=0,offset=len; sscanf(line+offset,"%d:%f%n",&feat,&val,&len)>=2; i+=1,offset+=len){
            /* Throw away features that do not exist in the network */
            if (feat < maxfeat){
                s->idx[i]=feat;
                s->x[i]=val;
            }
        }
        free(line);
        return 1;
    }
    free(line);
    return 0;
}

void loadData(const char* name, dataset_t* d){
    FILE* fp;
    int total,i,maxline;

    fp=fopen(name,"r");
    if(fp==NULL){
        printf("Could not open file %s\n",name);
        exit(1);
    }
    maxline=getDimensions(fp, &d->nex, &d->nfeat);
    d->example=malloc(d->nex*sizeof(sparse_t));
    d->target=malloc(d->nex*sizeof(int));
    total=getSizes(fp, maxline, d);
    d->example[0].x=malloc(total*sizeof(float));
    d->example[0].idx=malloc(total*sizeof(int));

    for(i=1; i<d->nex; i++){
        d->example[i].x=d->example[i-1].x+d->example[i-1].nz;
        d->example[i].idx=d->example[i-1].idx+d->example[i-1].nz;
    }
    /* Finally memory has been set up and we can read the data */
    readExamples(fp, maxline, d);
    fclose(fp);
}

void freeData(dataset_t* d){  
    free(d->target);
    free(d->example[0].x);
    free(d->example[0].idx);
    free(d->example);
}

void clipvectors(int inputs, sparse_t* v, int len){
    int i,j;
    for(i=0; i<len; i++){
        for(j=v[i].nz-1; j>=0 && v[i].idx[j] >= inputs; j--)
            ;
        v[i].nz=j+1;
    }
}

