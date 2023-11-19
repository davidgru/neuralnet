
NAIVE := used


SOURCEDIR := src

INCLUDE := -I$(SOURCEDIR)/lib -I$(SOURCEDIR)/dataset
SRC := $(SOURCEDIR)/dataset/ai_mnist.c $(SOURCEDIR)/lib/log.c

ifdef NAIVE
INCLUDE += -I$(SOURCEDIR)/naive
SRC += $(shell find $(SOURCEDIR)/naive -name '*.c')
else
$(error Only NAIVE implementation available.)
endif

OBJ := $(SRC:.c=.o)

CC := gcc

CFLAGS := -DAI_LOG_LEVEL=3 -march=haswell
CFLAGS += -O3 -Ofast -DAI_USE_AVX
# CFLAGS += -g

LDFLAGS := -lm	# math library

$(TARGET): $(OBJ)
	$(CC) $(INCLUDE) $(CFLAGS) -c $@.c -o $(TARGET).o
	$(CC) $^ $(TARGET).o -o $@ $(LDFLAGS)

$(SOURCEDIR)/%.o: $(SOURCEDIR)/%.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@


all: $(TARGET)

clean:
	@$(RM) -rv $(TARGET) $(TARGET).o $(OBJ)

rebuild:
	make clean && make

.PHONY: all clean rebuild
