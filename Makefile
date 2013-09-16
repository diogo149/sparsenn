CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lm -lblas

ifndef build
	build=release
endif

ifeq ($(build),release)
	CFLAGS += -DNDEBUG -O3 -fomit-frame-pointer -ffast-math
endif

ifeq ($(build),profile)
	CFLAGS += -g3 -pg -fprofile-arcs -ftest-coverage
endif

ifeq ($(build),debug)
	CFLAGS += -g3
endif

%.o: %.c
	$(CC) $(CFLAGS) -c $<

all: nnlearn nnclassify

debug: 
	make build=debug

profile:
	make build=profile

nnlearn: learn.o dataset.o metrics.o nnet.o
	$(CC) $(CFLAGS) -o nnlearn learn.o dataset.o metrics.o nnet.o $(LDFLAGS) 

nnclassify: classify.o dataset.o metrics.o nnet.o
	$(CC) $(CFLAGS) -o nnclassify classify.o dataset.o metrics.o nnet.o $(LDFLAGS) 

learn.o: learn.c dataset.h metrics.h nnet.h
classify.o: classify.c dataset.h metrics.h nnet.h
dataset.o: dataset.c dataset.h
metrics.o: metrics.c metrics.h
nnet.o: nnet.c dataset.h nnet.h

clean:
	/bin/rm -f svn-commit* *.o *.gcov *.gcda *.gcno gmon.out nnlearn nnclassify

