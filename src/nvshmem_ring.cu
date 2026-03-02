#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>

#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(e));                                        \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__global__ void putmem_kernel(const int *src, int *dst, int n) {
    int me   = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (me + 1) % npes;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_putmem(dst, src, (size_t)n * sizeof(int), peer);
        nvshmem_quiet();
    }
}

__global__ void atomic_kernel(long *counter_on_pe0, long *ticket_out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        long t = nvshmem_long_atomic_fetch_add(counter_on_pe0, 1LL, /*pe=*/0);
        ticket_out[0] = t;
        nvshmem_quiet();
    }
}

// Launch 2 PEs via fork(), distribute UID via pipe, then call nvshmemx_init_attr with UNIQUEID.
// This matches NVSHMEM's intended "launcher-agnostic" UID bootstrap flow. :contentReference[oaicite:1]{index=1}
static int uid_init_2pes(int &rank, int &nranks) {
    nranks = 2;

    int pfd[2];
    if (pipe(pfd) != 0) {
        perror("pipe");
        return 1;
    }

    // Parent will be rank 0, child will be rank 1.
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;

    if (pid == 0) {
        // ---- child ----
        rank = 1;
        close(pfd[1]); // close write end

        ssize_t need = (ssize_t)sizeof(uid);
        char *buf = reinterpret_cast<char*>(&uid);
        ssize_t got = 0;
        while (got < need) {
            ssize_t r = read(pfd[0], buf + got, need - got);
            if (r <= 0) { perror("read uid"); std::exit(1); }
            got += r;
        }
        close(pfd[0]);
    } else {
        // ---- parent ----
        rank = 0;
        close(pfd[0]); // close read end

        // Rank0 obtains a Unique ID token from NVSHMEM runtime. :contentReference[oaicite:2]{index=2}
        int rc = nvshmemx_get_uniqueid(&uid);
        if (rc != 0) {
            fprintf(stderr, "nvshmemx_get_uniqueid failed rc=%d\n", rc);
            std::exit(1);
        }

        ssize_t need = (ssize_t)sizeof(uid);
        const char *buf = reinterpret_cast<const char*>(&uid);
        ssize_t sent = 0;
        while (sent < need) {
            ssize_t w = write(pfd[1], buf + sent, need - sent);
            if (w <= 0) { perror("write uid"); std::exit(1); }
            sent += w;
        }
        close(pfd[1]);
    }

    // Bind GPU by rank (single node, 2 GPUs: rank0->GPU0, rank1->GPU1)
    CHECK_CUDA(cudaSetDevice(rank));

    // Build init attributes for UNIQUEID bootstrap and initialize NVSHMEM. :contentReference[oaicite:3]{index=3}
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(rank, nranks, &uid, &attr); // helper API :contentReference[oaicite:4]{index=4}

    int rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr); // non-zero flags :contentReference[oaicite:5]{index=5}
    if (rc != 0) {
        fprintf(stderr, "nvshmemx_init_attr(UNIQUEID) failed rc=%d\n", rc);
        std::exit(1);
    }

    return 0;
}

int main() {
    int rank = -1, nranks = -1;
    if (uid_init_2pes(rank, nranks) != 0) return 1;

    int me   = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    printf("PID %d: rank=%d  PE=%d/%d  GPU=%d\n", (int)getpid(), rank, me, npes, rank);

    // ---------------- Demo1: bulk putmem ring ----------------
    const int N = 8;
    int *src = (int*)nvshmem_malloc(N * sizeof(int));
    int *dst = (int*)nvshmem_malloc(N * sizeof(int));

    int hsrc[N];
    for (int i = 0; i < N; ++i) hsrc[i] = me * 100 + i;

    CHECK_CUDA(cudaMemcpyAsync(src, hsrc, N * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(dst, 0xFF, N * sizeof(int), stream)); // -1
    putmem_kernel<<<1,1,0,stream>>>(src, dst, N);
    CHECK_CUDA(cudaGetLastError());

    nvshmemx_barrier_all_on_stream(stream);

    int hdst[N];
    CHECK_CUDA(cudaMemcpyAsync(hdst, dst, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    int expected_writer = (me + npes - 1) % npes;
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (hdst[i] != expected_writer * 100 + i) { ok = false; break; }
    }
    printf("[Demo1 putmem] PE %d received: %d %d %d %d %d %d %d %d  (%s)\n",
           me, hdst[0], hdst[1], hdst[2], hdst[3], hdst[4], hdst[5], hdst[6], hdst[7],
           ok ? "OK" : "MISMATCH");

    // ---------------- Demo2: atomic ticket counter on PE0 ----------------
    long *counter = (long*)nvshmem_malloc(sizeof(long));
    long *ticket  = (long*)nvshmem_malloc(sizeof(long));

    if (me == 0) {
        long zero = 0;
        CHECK_CUDA(cudaMemcpyAsync(counter, &zero, sizeof(long), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    nvshmemx_barrier_all_on_stream(stream);

    atomic_kernel<<<1,1,0,stream>>>(counter, ticket);
    CHECK_CUDA(cudaGetLastError());

    nvshmemx_barrier_all_on_stream(stream);

    long ht = -1;
    CHECK_CUDA(cudaMemcpyAsync(&ht, ticket, sizeof(long), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("[Demo2 atomic] PE %d ticket=%ld\n", me, (long)ht);

    if (me == 0) {
        long finalv = -1;
        CHECK_CUDA(cudaMemcpyAsync(&finalv, counter, sizeof(long), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("[Demo2 atomic] PE0 final counter=%ld (expect %d)\n", (long)finalv, npes);
    }

    nvshmem_free(ticket);
    nvshmem_free(counter);
    nvshmem_free(dst);
    nvshmem_free(src);

    CHECK_CUDA(cudaStreamDestroy(stream));
    nvshmem_finalize();

    // Let parent wait child to avoid zombies
    if (rank == 0) {
        int status = 0;
        wait(&status);
    }

    return 0;
}