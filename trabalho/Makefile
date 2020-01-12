CC1=g++
SOURCE_DIR=src
BIN_DIR=bin
LIB_DIR=lib

PAPI_LIB=-L/share/apps/papi/5.5.0/lib
PAPI_INCLUDE=-I/share/apps/papi/5.5.0/include

OUTPUT1=$(BIN_DIR)/singleThread

SOURCE1= $(SOURCE_DIR)/singleThread.cpp

CFLAGS= -O3 -Wall -Wextra -std=c++11 -lm -g -lpapi -flto-report $(PAPI_LIB) $(PAPI_INCLUDE) -lpthread

MKDIR= mkdir -p

all: $(OUTPUT1)

$(OUTPUT1):  $(SOURCE1)
	    $(MKDIR) $(BIN_DIR)
	    $(CC1) $(SOURCE1) $(CFLAGS) -o $(OUTPUT1)

.PHONY: clean
clean:
	rm -f $(OUTPUT1)
