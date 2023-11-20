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


.PHONY: clean rebuild
