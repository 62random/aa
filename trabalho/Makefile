CC1=g++
CC2=nvcc
SOURCE_DIR=src
BIN_DIR=bin
LIB_DIR=lib

PAPI_LIB=-L/share/apps/papi/5.5.0/lib
PAPI_INCLUDE=-I/share/apps/papi/5.5.0/include

OUTPUT1=$(BIN_DIR)/singleThread
OUTPUT2=$(BIN_DIR)/singleBlocking
OUTPUT3=$(BIN_DIR)/cuda

SOURCE1= $(SOURCE_DIR)/singleThread.cpp
SOURCE2= $(SOURCE_DIR)/singleBlocking.cpp
SOURCE3= $(SOURCE_DIR)/cuda.cu

CFLAGS= -O3 -fopt-info-optimized -fopenmp -Wall -Wextra -lm -g -lpapi -flto-report $(PAPI_LIB) $(PAPI_INCLUDE) -lpthread
CFLAGS2= -g -G -DD_GPU I/share/apps/papi/5.5.0/include -L/share/apps/papi/5.5.0/lib  -lpapi

MKDIR= mkdir -p

all: $(OUTPUT3) #$(OUTPUT2) #$(OUTPUT1)

$(OUTPUT1):  $(SOURCE1)
	    $(MKDIR) $(BIN_DIR)
	    $(CC1) $(SOURCE1) $(CFLAGS) -o $(OUTPUT1)

$(OUTPUT2):  $(SOURCE2)
	    $(MKDIR) $(BIN_DIR)
	    $(CC1) $(SOURCE2) $(CFLAGS) -o $(OUTPUT2)

$(OUTPUT3):  $(SOURCE3)
	    $(MKDIR) $(BIN_DIR)
	    $(CC2) $(SOURCE3) $(CFLAGS2) -o $(OUTPUT3)



.PHONY: clean
clean:
	#rm -f $(OUTPUT1)
	#rm -f $(OUTPUT2)
	rm -f $(OUTPUT3)
