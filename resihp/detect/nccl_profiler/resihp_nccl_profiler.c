/*
 * ResiHP NCCL profiler plugin: records host-side start/stop of every collective and
 * point-to-point operation into a per-process shared-memory ring that the ResiHP
 * detector reads (resihp/detect/nccl_profile_reader.py). Uses NCCL's official
 * profiler plugin interface instead of LD_PRELOAD interposition.
 *
 *   NCCL_PROFILER_PLUGIN=/path/libresihp_nccl_profiler.so
 *   RESIHP_NCCL_PROF_DIR=/dev/shm   (ring file: resihp_nccl_<pid>.ring)
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "nccl_profiler_v5.h"

#define RESIHP_RING_MAGIC 0x52455348u /* "RESH" */
#define RESIHP_RING_VERSION 1u
#define RESIHP_RING_CAPACITY 65536u
#define RESIHP_FUNC_LEN 24

typedef struct {
  uint64_t start_ns;
  uint64_t end_ns;
  uint64_t comm_id;
  uint64_t count;
  int32_t rank;
  int32_t nranks;
  int32_t type; /* ncclProfileColl or ncclProfileP2p */
  int32_t peer; /* -1 for collectives */
  char func[RESIHP_FUNC_LEN];
  _Atomic uint64_t seq; /* index + 1 once the record is complete (seqlock for the reader) */
} resihp_record_t;

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t capacity;
  uint32_t record_size;
  _Atomic uint64_t head;
} resihp_ring_header_t;

typedef struct {
  uint64_t comm_id;
  int rank;
  int nranks;
} resihp_ctx_t;

typedef struct {
  resihp_ctx_t* ctx;
  uint64_t start_ns;
  uint64_t count;
  int type;
  int peer;
  char func[RESIHP_FUNC_LEN];
} resihp_event_t;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static resihp_ring_header_t* g_ring = NULL;
static resihp_record_t* g_records = NULL;
static size_t g_bytes = 0;
static int g_refs = 0;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int open_ring(void) {
  const char* dir = getenv("RESIHP_NCCL_PROF_DIR");
  char path[1024];
  if (dir == NULL) dir = "/dev/shm";
  snprintf(path, sizeof(path), "%s/resihp_nccl_%d.ring", dir, (int)getpid());
  g_bytes = sizeof(resihp_ring_header_t) + (size_t)RESIHP_RING_CAPACITY * sizeof(resihp_record_t);
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return -1;
  if (ftruncate(fd, (off_t)g_bytes) != 0) {
    close(fd);
    return -1;
  }
  void* p = mmap(NULL, g_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return -1;
  memset(p, 0, g_bytes);
  g_ring = (resihp_ring_header_t*)p;
  g_records = (resihp_record_t*)((char*)p + sizeof(resihp_ring_header_t));
  g_ring->capacity = RESIHP_RING_CAPACITY;
  g_ring->record_size = (uint32_t)sizeof(resihp_record_t);
  g_ring->version = RESIHP_RING_VERSION;
  atomic_store(&g_ring->head, 0);
  atomic_thread_fence(memory_order_release);
  g_ring->magic = RESIHP_RING_MAGIC;
  return 0;
}

static ncclResult_t resihp_init(void** context, uint64_t commId, int* eActivationMask, const char* commName,
                                int nNodes, int nranks, int rank, ncclDebugLogger_t logfn) {
  (void)commName;
  (void)nNodes;
  (void)logfn;
  pthread_mutex_lock(&g_lock);
  if (g_ring == NULL && open_ring() != 0) {
    pthread_mutex_unlock(&g_lock);
    *eActivationMask = 0;
    *context = NULL;
    return ncclSuccess; /* never break the job because profiling is unavailable */
  }
  g_refs++;
  pthread_mutex_unlock(&g_lock);
  resihp_ctx_t* ctx = (resihp_ctx_t*)calloc(1, sizeof(resihp_ctx_t));
  if (ctx == NULL) return ncclSystemError;
  ctx->comm_id = commId;
  ctx->rank = rank;
  ctx->nranks = nranks;
  *context = ctx;
  *eActivationMask = ncclProfileColl | ncclProfileP2p;
  return ncclSuccess;
}

static ncclResult_t resihp_start(void* context, void** eHandle, ncclProfilerEventDescr_v5_t* d) {
  *eHandle = NULL;
  if (context == NULL || d == NULL) return ncclSuccess;
  if (d->type != ncclProfileColl && d->type != ncclProfileP2p) return ncclSuccess;
  resihp_event_t* ev = (resihp_event_t*)malloc(sizeof(resihp_event_t));
  if (ev == NULL) return ncclSuccess;
  ev->ctx = (resihp_ctx_t*)context;
  ev->type = (int)d->type;
  if (d->type == ncclProfileColl) {
    strncpy(ev->func, d->coll.func ? d->coll.func : "?", RESIHP_FUNC_LEN - 1);
    ev->count = d->coll.count;
    ev->peer = -1;
  } else {
    strncpy(ev->func, d->p2p.func ? d->p2p.func : "?", RESIHP_FUNC_LEN - 1);
    ev->count = d->p2p.count;
    ev->peer = d->p2p.peer;
  }
  ev->func[RESIHP_FUNC_LEN - 1] = '\0';
  ev->start_ns = now_ns();
  *eHandle = ev;
  return ncclSuccess;
}

static ncclResult_t resihp_stop(void* eHandle) {
  resihp_event_t* ev = (resihp_event_t*)eHandle;
  if (ev == NULL) return ncclSuccess;
  if (g_ring != NULL) {
    uint64_t idx = atomic_fetch_add(&g_ring->head, 1);
    resihp_record_t* r = &g_records[idx % RESIHP_RING_CAPACITY];
    atomic_store_explicit(&r->seq, 0, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    r->start_ns = ev->start_ns;
    r->end_ns = now_ns();
    r->comm_id = ev->ctx->comm_id;
    r->count = ev->count;
    r->rank = ev->ctx->rank;
    r->nranks = ev->ctx->nranks;
    r->type = ev->type;
    r->peer = ev->peer;
    memcpy(r->func, ev->func, RESIHP_FUNC_LEN);
    atomic_store_explicit(&r->seq, idx + 1, memory_order_release);
  }
  free(ev);
  return ncclSuccess;
}

static ncclResult_t resihp_record_state(void* eHandle, ncclProfilerEventState_v5_t eState,
                                        ncclProfilerEventStateArgs_v5_t* eStateArgs) {
  (void)eHandle;
  (void)eState;
  (void)eStateArgs;
  return ncclSuccess;
}

static ncclResult_t resihp_finalize(void* context) {
  free(context);
  pthread_mutex_lock(&g_lock);
  if (context != NULL && --g_refs == 0 && g_ring != NULL) {
    munmap(g_ring, g_bytes);
    g_ring = NULL;
    g_records = NULL;
  }
  pthread_mutex_unlock(&g_lock);
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclProfiler_v5_t ncclProfiler_v5 = {
    "resihp", resihp_init, resihp_start, resihp_stop, resihp_record_state, resihp_finalize,
};
