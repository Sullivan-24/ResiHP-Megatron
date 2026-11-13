CC ?= cc
CFLAGS ?= -O2 -fPIC -Wall -Wextra -std=c11 -D_GNU_SOURCE
LIB := libresihp_nccl_profiler.so

$(LIB): resihp_nccl_profiler.c nccl_profiler_v5.h
	$(CC) $(CFLAGS) -shared -o $@ resihp_nccl_profiler.c -lpthread

clean:
	rm -f $(LIB)

.PHONY: clean
