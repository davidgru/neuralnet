
NAIVE := used


SOURCEDIR := src

INCLUDE := -I$(SOURCEDIR)/lib
SRC := $(SOURCEDIR)/dataset/ai_mnist.c $(SOURCEDIR)/main.c $(SOURCEDIR)/lib/log.c

ifdef NAIVE
INCLUDE += -I$(SOURCEDIR)/naive
SRC += $(shell find $(SOURCEDIR)/naive -name '*.c')
else
$(error Only NAIVE implementation available.)
endif

OBJ := $(SRC:.c=.o)


TARGET := main

CC := gcc

CFLAGS := -march=haswell -DAI_LOG_LEVEL=2 -DAI_USE_AVX -O3 -Ofast

LDFLAGS := -lm 			# math library

$(TARGET): $(OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)

$(SOURCEDIR)/%.o: $(SOURCEDIR)/%.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

all: $(TARGET)

clean:
	@$(RM) -rv $(TARGET) $(OBJ)

rebuild:
	make clean && make

.PHONY: all clean rebuild
