CC1=g++
SOURCE_DIR=src
BIN_DIR=bin
LIB_DIR=lib

OUTPUT1=$(BIN_DIR)/singleThread

SOURCE1= $(SOURCE_DIR)/singleThread.cpp

CFLAGS= -O3 -lm -g -flto-report

MKDIR= mkdir -p

all: $(OUTPUT1)

$(OUTPUT1):  $(SOURCE1)
	    $(MKDIR) $(BIN_DIR)
	    $(CC1) $(SOURCE1) $(CFLAGS) -o $(OUTPUT1)

.PHONY: clean
clean:
	rm -f $(OUTPUT1)
