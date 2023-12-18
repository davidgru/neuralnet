CC := gcc


# Select backend files based on selected backend
# Supported values: naive, onednn
BACKEND ?= onednn

# Set to 1 to use accelerated matrix products when using naive backend
USE_AVX ?= 0

# The root directory of the oneDNN library, only needed when using
# onednn backend
ONEDNN_ROOT_DIR ?= lib/onednn
ONEDNN_INCLUDE_DIR := $(ONEDNN_ROOT_DIR)/include
ONEDNN_SHARED_DIR := $(ONEDNN_ROOT_DIR)/lib/

# Select log level
# Supported values: 1(error), 2(warn), 3(info), 4(trace)
LOG_LEVEL ?= 3

# Set to 1 to enable gdb support
DEBUG ?= 0


ifeq ($(DEBUG),1)
ifeq ($(USE_AVX),1)
$(error Can not have DEBUG=1 and USE_AVX=1 at the same time)
endif
endif


CFLAGS :=
ifdef LOG_LEVEL
CFLAGS += -DLOG_LEVEL=$(LOG_LEVEL)
endif
ifeq ($(USE_AVX),1)
CFLAGS += -march=haswell -DUSE_AVX
endif
ifeq ($(DEBUG),1)
CFLAGS += -g -DDEBUG
else
CFLAGS += -O3 -Ofast
endif


# math library
LDFLAGS := -lm


SOURCEDIR := src

# INCLUDE and SOURCE file located in the src directory
INCLUDE := -I$(SOURCEDIR)/lib -I$(SOURCEDIR)/common
SRC := $(shell find $(SOURCEDIR)/common -name '*.c')
SRC += $(SOURCEDIR)/lib/log.c $(SOURCEDIR)/lib/config_info.c $(SOURCEDIR)/lib/random.c
# Also add the target source file
SRC += $(TARGET).c


# Select backend files based on selected backend
ifeq ($(BACKEND),naive)
INCLUDE += -I$(SOURCEDIR)/naive -I$(SOURCEDIR)/include
SRC += $(shell find $(SOURCEDIR)/naive -name '*.c')
CFLAGS += -DBACKEND_NAIVE
else ifeq ($(BACKEND),onednn)
INCLUDE += -I$(SOURCEDIR)/onednn -I$(ONEDNN_INCLUDE_DIR)
SRC += $(shell find $(SOURCEDIR)/onednn -name '*.c')
LDFLAGS += -L$(ONEDNN_SHARED_DIR) -ldnnl 
CFLAGS += -DBACKEND_ONEDNN
else
$(error Only naive and onednn implementation available.)
endif


# Object files are placed in same directory as src files, just with different file extension
OBJ := $(SRC:.c=.o)
