include config/defines.mk


# Link all object files into a source file
$(TARGET): $(OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)


# Rule to compile a single translation unit
%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@


clean:
	@$(RM) -rv $(TARGET) $(OBJ)


rebuild:
	make clean && make


run: $(TARGET)
# since oneDNN is built as a shared library, need to add its location
# to LD_LIBRARY_PATH so that the target executable can find it
ifeq ($(BACKEND),onednn)
run: export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(ONEDNN_SHARED_DIR)
endif
run:
	$(TARGET)


.PHONY: clean rebuild run
