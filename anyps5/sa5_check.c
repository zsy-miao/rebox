/* sa5_check.exe: compiles compute shaders through rebox_anyps5.dll.
 *   sa5_check                                  built-in s_endpgm shader
 *   sa5_check CODE.bin SKIP RSRC1 RSRC2 X,Y,Z  a dumped shader (e.g. a PS09
 *                                              entry, SKIP = 36), compile only,
 *                                              with no guest memory
 *   sa5_check vs PLAN [FIRST [COUNT]]          the vertex shaders in PLAN (format: vertex_sweep);
 *                                              SA5_USER, SA5_MEMORY: a draw's user SGPRs and guest
 *                                              memory (see vertex_sweep)
 *   sa5_check ps PLAN [FIRST [COUNT]]          the pixel shaders in PLAN (format: pixel_sweep) with
 *                                              their PSO register state, compile only, no guest
 *                                              memory unless a line gives a draw's (u=, m=:
 *                                              pixel_parse); one "ps NAME ok|FAIL ..." line each
 *   sa5_check cs PLAN [FIRST [COUNT]]          the compute shaders in PLAN (format: compute_sweep),
 *                                              compile only, validated, loops counted; no guest
 *                                              memory unless SA5_USER, SA5_MEMORY give some (as for vs)
 *   sa5_check mesh                             a built-in console mesh program (one triangle),
 *                                              compiled and finalized as d3d12_xs does
 *   sa5_check ms PLAN [FIRST [COUNT]]          the mesh shaders in PLAN (format: mesh_sweep), the
 *                                              same way; SA5_USER, SA5_MEMORY as for vs
 *   sa5_check bench THREADS PLAN [FIRST [COUNT]]  compile throughput: the pixel shaders in PLAN (format:
 *                                              pixel_sweep), compiled and validated on THREADS
 *                                              threads at once
 *   sa5_check memo PLAN NAME [ITERATIONS]      the memoized path as at every dispatch: compute shader
 *                                              NAME of PLAN (format: NAME CODE SKIP RSRC1 RSRC2 X,Y,Z
 *                                              [WAVE]) with new descriptor tables and buffers at each
 *                                              call, timed by phase (d3d12_xs's)
 * Compute shaders (built-in or dumped), ps and bench compile with no cache unless SA5_USE_CACHE is set:
 * then through AnyPS5's caches, its disk cache included (ANYPS5_SHADER_CACHE_DIR), as d3d12_xs
 * compiles. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "rebox_anyps5.h"

static int compile(const uint32_t *code, uint32_t dwords, uint32_t rsrc1, uint32_t rsrc2, const uint32_t threads[3])
{
    struct sa5_compute_request req = { 0 };
    struct sa5_result *result = NULL;
    uint32_t user_data[16] = { 0 }, words = 0, count, wave;
    char error[512];

    req.code = code;
    req.code_dwords = dwords;
    req.code_address = 0x10000000;
    req.user_data = user_data;
    req.user_data_count = (rsrc2 >> 1) & 0x1f;
    memcpy(req.threads, threads, sizeof(req.threads));
    req.rsrc1 = rsrc1;
    req.rsrc2 = rsrc2;
    req.use_cache = getenv("SA5_USE_CACHE") != NULL;
    wave = 32;
    req.wave = wave;
    if (sa5_compile_compute(&req, &result, error, sizeof(error)))
    {
        printf("FAIL compile: %s\n", error);
        return 1;
    }
    sa5_result_spirv(result, &words);
    count = sa5_result_binding_count(result);
    printf("ok   compiled: %u SPIR-V words, %u bindings, subgroup %u\n", words, count,
           sa5_result_subgroup_size(result));
    for (uint32_t i = 0; i < count; i++)
    {
        struct sa5_binding b;
        sa5_result_binding(result, i, &b);
        printf("     binding kind %u set %u binding %u count %u words %u\n", b.kind, b.set, b.binding, b.count,
               b.word_count);
    }
    sa5_result_free(result);
    return 0;
}

static uint8_t *read_all(const char *path, long *size)
{
    FILE *f = fopen(path, "rb");
    uint8_t *data;

    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = malloc(*size);
    if (data && fread(data, 1, *size, f) != (size_t)*size)
    {
        free(data);
        data = NULL;
    }
    fclose(f);
    return data;
}

/* A pixel plan line: NAME CODE SKIP RSRC1 RSRC2 ENA NINTERP COL DEPTH CNTL, then optionally
 * u=V0,V1,... (hex user SGPRs from s0) and m=ADDRESS@PATH (a file's bytes as guest memory at that
 * address): a draw's user SGPRs and tables (tools/ps-plan.py --draw). Returns 1 with the request
 * filled in (the code in *data, for the caller to free; the memory in req->regions[0].data, too), 0
 * for a line that isn't one, -1 if the code or memory can't be read. */
static int pixel_parse(const char *line, char name[64], struct sa5_pixel_request *req, uint32_t *user_data,
                       uint8_t **data)
{
    char path[1024], winpath[1100], cntl[1024], ignored[64];
    unsigned skip, rsrc1, rsrc2, ena, ninterp, col, depth;
    static struct sa5_slot slots[1];
    struct sa5_layout layout;
    long size;

    *data = NULL;
    if (line[0] == '#' || sscanf(line, "%63s %1023s %u %x %x %x %u %x %u %1023s", name, path, &skip, &rsrc1, &rsrc2,
                                 &ena, &ninterp, &col, &depth, cntl) != 10)
        return 0;
    snprintf(winpath, sizeof(winpath), "%s%s", path[0] == '/' ? "Z:" : "", path);
    for (char *p = winpath; *p; p++)
        if (*p == '/') *p = '\\';
    if (!(*data = read_all(winpath, &size)) || size <= (long)skip)
    {
        free(*data);
        *data = NULL;
        return -1;
    }
    memset(req, 0, sizeof(*req));
    req->code = (const uint32_t *)(*data + skip);
    req->code_dwords = (uint32_t)((size - skip) / 4);
    req->code_address = 0x10000000;
    req->user_data = user_data;
    req->user_data_count = (rsrc2 >> 1) & 0x1f;
    req->rsrc1 = rsrc1;
    req->rsrc2 = rsrc2;
    req->input_ena = ena;
    req->interpolator_count = ninterp;
    /* SPI_PS_INPUT_CNTL_n from the PSO, comma-separated hex ("-" for none). */
    {
        char *p = cntl;
        for (unsigned i = 0; i < 32 && *p && *p != '-'; i++)
        {
            req->interpolator_settings[i] = (uint32_t)strtoul(p, &p, 16);
            if (*p == ',')
                p++;
        }
    }
    req->col_format = col;
    req->depth_export = depth;
    {
        const char *u = strstr(line, " u="), *m = strstr(line, " m=");
        if (u)
        {
            const char *p = u + 3;
            for (unsigned i = 0; i < 32; i++)
            {
                char *end;
                user_data[i] = (uint32_t)strtoul(p, &end, 16);
                if (end == p || *end != ',')
                    break;
                p = end + 1;
            }
        }
        if (m)
        {
            static __thread struct sa5_region region;
            char file[1024], winfile[1100];
            unsigned long long address;
            long bytes;
            if (sscanf(m + 3, "%llx@%1023s", &address, file) != 2)
                return -1;
            snprintf(winfile, sizeof(winfile), "%s%s", file[0] == '/' ? "Z:" : "", file);
            for (char *p = winfile; *p; p++)
                if (*p == '/') *p = '\\';
            region.address = address;
            if (!(region.data = read_all(winfile, &bytes)))
                return -1;
            region.size = (size_t)bytes;
            req->regions = &region;
            req->region_count = 1;
        }
    }
    req->use_cache = getenv("SA5_USE_CACHE") != NULL;
    req->wave = !sa5_analyse(req->code, req->code_dwords, req->user_data_count, &layout, slots, 0, ignored,
                             sizeof(ignored)) ? layout.wave : 64;
    return 1;
}

static int pixel_sweep(const char *plan, int first, int count)
{
    FILE *f = fopen(plan, "r");
    char line[4096];
    int index = 0, ok = 0, failed = 0, unreadable = 0, valid_count = 0;

    if (!f)
    {
        printf("FAIL open %s\n", plan);
        return 1;
    }
    while (fgets(line, sizeof(line), f) && (count <= 0 || ok + failed + unreadable < count))
    {
        char name[64];
        struct sa5_pixel_request req;
        struct sa5_result *result = NULL;
        uint32_t user_data[32] = { 0 }, words = 0;
        char error[600];
        uint8_t *data;
        int parsed;

        if (!(parsed = pixel_parse(line, name, &req, user_data, &data)))
            continue;
        if (index++ < first)
        {
            free(data);
            continue;
        }
        if (parsed < 0)
        {
            printf("ps %s FAIL unreadable\n", name);
            unreadable++;
            continue;
        }
        if (sa5_compile_pixel(&req, &result, error, sizeof(error)))
        {
            for (char *p = error; *p; p++)
                if (*p == '\n') *p = ' ';
            printf("ps %s FAIL %.400s\n", name, error);
            failed++;
        }
        else
        {
            const uint32_t *spirv = sa5_result_spirv(result, &words);
            int valid = !sa5_validate(spirv, words, error, sizeof(error));
            for (char *p = error; *p; p++)
                if (*p == '\n') *p = ' ';
            printf("ps %s ok %u words, wave%u, %u bindings, %s%.300s\n", name, words, req.wave,
                   sa5_result_binding_count(result), valid ? "valid" : "INVALID: ", valid ? "" : error);
            valid_count += valid;
            sa5_result_free(result);
            ok++;
        }
        fflush(stdout);
        free(data);
        if (req.region_count)
            free((void *)req.regions[0].data);
    }
    fclose(f);
    printf("%s ps-check: %d of %d pixel shaders compile with their PSO register state, %d of those pass spirv-val "
           "(%d unreadable)\n", failed || unreadable ? "info" : "ok  ", ok, ok + failed + unreadable, valid_count,
           unreadable);
    return 0;
}

/* Compile throughput: the pixel shaders of a plan, read first, then compiled and validated by
 * THREADS threads at once (each takes the next shader), as d3d12_xs's compile workers do. */
static struct bench_shader
{
    struct sa5_pixel_request req;
    uint32_t user_data[32];
    uint8_t *data;
} *bench_shaders;
static LONG bench_count, bench_next, bench_ok;

static DWORD WINAPI bench_thread(void *unused)
{
    LONG i;

    (void)unused;
    while ((i = InterlockedIncrement(&bench_next) - 1) < bench_count)
    {
        struct sa5_result *result = NULL;
        const uint32_t *spirv;
        uint32_t words = 0;
        char error[600];

        if (sa5_compile_pixel(&bench_shaders[i].req, &result, error, sizeof(error)))
            continue;
        spirv = sa5_result_spirv(result, &words);
        if (!sa5_validate(spirv, words, error, sizeof(error)))
            InterlockedIncrement(&bench_ok);
        sa5_result_free(result);
    }
    return 0;
}

static int bench(int threads, const char *plan, int first, int count)
{
    FILE *f = fopen(plan, "r");
    char line[4096], name[64];
    HANDLE handles[64];
    LARGE_INTEGER frequency, start, end;
    int index = 0, limit = count > 0 ? count : 65536;

    if (!f || threads < 1 || threads > 64)
    {
        printf("FAIL bench: open %s, 1-64 threads\n", plan);
        return 1;
    }
    bench_shaders = calloc(limit, sizeof(*bench_shaders));
    while (fgets(line, sizeof(line), f) && bench_count < limit)
    {
        struct bench_shader *s = &bench_shaders[bench_count];
        int parsed = pixel_parse(line, name, &s->req, s->user_data, &s->data);

        if (parsed && index++ >= first && parsed > 0)
            bench_count++;
        else
            free(s->data);
    }
    fclose(f);
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    for (int t = 0; t < threads; t++)
        handles[t] = CreateThread(NULL, 0, bench_thread, NULL, 0, NULL);
    WaitForMultipleObjects(threads, handles, TRUE, INFINITE);
    QueryPerformanceCounter(&end);
    printf("bench: %ld pixel shaders on %d threads in %.0f ms (%ld compile and validate)\n", bench_count, threads,
           (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart, bench_ok);
    return 0;
}

/* The memoized path at every dispatch, as d3d12_xs drives it (console_compile.c's compile_now): one
 * compute shader of a plan (format as cs plans: NAME CODE SKIP RSRC1 RSRC2 X,Y,Z [WAVE]) compiled
 * ITERATIONS times through AnyPS5's caches, its descriptor tables (laid out from sa5_analyse) at a
 * new address each time and its buffers' V#s naming new memory, as a title's per-draw data does;
 * then its identity, its bindings and, once per module key, sa5_finalize. Prints the first call's
 * time and the average of the rest, phase by phase. SA5_BENCH_SLOTS=1 gives every dispatch a new
 * module key (new heap slots, as when a title copies its descriptors around); SA5_BENCH_IMAGES=N
 * cycles the textures through N sets; SA5_BENCH_FIXED_TABLES=1 keeps the tables in one place (only
 * the buffers they name change), and SA5_BENCH_COPIES=N moves them through N places (64), after
 * which calls repeat earlier ones. */
static uint64_t bench_hash(uint64_t h, const void *data, size_t size)
{
    for (size_t i = 0; i < size; i++)
        h = (h ^ ((const uint8_t *)data)[i]) * 0x100000001b3ull;
    return h;
}

static int memo_bench(const char *plan, const char *wanted, int iterations)
{
    enum { TABLE_BASE = 0x20000000, BUFFER_BASE = 0x30000000, IMAGE_BASE = 0x40000000 };
    FILE *f = fopen(plan, "r");
    char line[4096], name[64] = "", path[1024], winpath[1100], error[1024];
    unsigned skip, rsrc1, rsrc2, x, y, z, wave = 0;
    long size = 0;
    uint8_t *data = NULL, *tables, *buffers, *images;
    struct sa5_layout layout;
    static struct sa5_slot slots[256];
    struct sa5_region regions[40];
    uint32_t user[32] = { 0 }, table_bytes = 0, region_count = 0;
    struct sa5_compute_request req = { 0 };
    LARGE_INTEGER frequency, t0, t1, t2, t3, t4, t4b, t5;
    uint64_t digest = 0xcbf29ce484222325ull;
    double first[5] = { 0 }, rest[5] = { 0 };
    uint64_t keys[64];
    int key_count = 0, finalized = 0;
    int new_slots = getenv("SA5_BENCH_SLOTS") && *getenv("SA5_BENCH_SLOTS") == '1';
    int image_sets = getenv("SA5_BENCH_IMAGES") ? atoi(getenv("SA5_BENCH_IMAGES")) : 0;
    int fixed_tables = getenv("SA5_BENCH_FIXED_TABLES") && *getenv("SA5_BENCH_FIXED_TABLES") == '1';
    uint32_t table_copies = getenv("SA5_BENCH_COPIES") ? (uint32_t)atoi(getenv("SA5_BENCH_COPIES")) : 64;

    while (f && fgets(line, sizeof(line), f))
    {
        wave = 0;
        if (line[0] != '#' && sscanf(line, "%63s %1023s %u %x %x %u,%u,%u %u", name, path, &skip, &rsrc1, &rsrc2, &x,
                                     &y, &z, &wave) >= 8 && !strcmp(name, wanted))
            break;
        name[0] = 0;
    }
    if (f)
        fclose(f);
    if (!name[0])
    {
        printf("FAIL memo: %s not in %s\n", wanted, plan);
        return 1;
    }
    snprintf(winpath, sizeof(winpath), "%s%s", path[0] == '/' ? "Z:" : "", path);
    for (char *p = winpath; *p; p++)
        if (*p == '/') *p = '\\';
    if (!(data = read_all(winpath, &size)) || size <= (long)skip)
    {
        printf("FAIL memo: can't read %s\n", path);
        return 1;
    }
    req.code = (const uint32_t *)(data + skip);
    req.code_dwords = (uint32_t)((size - skip) / 4);
    req.code_address = 0x10000000;
    req.user_data = user;
    req.user_data_count = (rsrc2 >> 1) & 0x1f;
    req.threads[0] = x;
    req.threads[1] = y;
    req.threads[2] = z;
    req.rsrc1 = rsrc1;
    req.rsrc2 = rsrc2;
    req.wave = wave ? wave : 32;
    req.use_cache = 1;
    memset(&layout, 0, sizeof(layout));
    if (sa5_analyse(req.code, req.code_dwords, req.user_data_count, &layout, slots, 256, error, sizeof(error)))
    {
        printf("FAIL memo: analyse: %s\n", error);
        return 1;
    }
    for (uint32_t i = 0; i < layout.slot_count && i < 256; i++)
        if ((slots[i].dword + slots[i].size) * 4 > table_bytes)
            table_bytes = (slots[i].dword + slots[i].size) * 4;
    table_bytes = (table_bytes + 255) & ~255u;
    if (!table_bytes)
        table_bytes = 256;
    printf("memo %s: %u code dwords, wave%u, %u tables, %u slots, %u-byte tables\n", name, req.code_dwords, req.wave,
           layout.table_count, layout.slot_count, table_bytes);

    /* Guest memory: copies of every table (16 tables per copy) to move them through, what they
     * point at, and more regions as a title has (other heaps and GPU memory chunks). */
    if (table_copies < 1 || (uint64_t)table_copies * 16 * table_bytes > BUFFER_BASE - TABLE_BASE)
        table_copies = (BUFFER_BASE - TABLE_BASE) / 16 / table_bytes;
    tables = calloc(table_copies * 16, table_bytes);
    buffers = calloc(1, 1 << 20);
    images = calloc(1, 1 << 20);
    regions[region_count++] = (struct sa5_region){ TABLE_BASE, tables, (size_t)table_copies * 16 * table_bytes };
    regions[region_count++] = (struct sa5_region){ BUFFER_BASE, buffers, 1 << 20 };
    regions[region_count++] = (struct sa5_region){ IMAGE_BASE, images, 1 << 20 };
    while (region_count < 40)
    {
        regions[region_count] = (struct sa5_region){ 0x100000000ull + region_count * 0x1000000ull, buffers, 1 << 20 };
        region_count++;
    }
    req.regions = regions;
    req.region_count = region_count;

    QueryPerformanceFrequency(&frequency);
    for (int it = 0; it < iterations; it++)
    {
        struct sa5_result *result = NULL;
        uint64_t identity[2], key;
        uint32_t copy = fixed_tables ? 0 : it % table_copies, count;
        double phase[5];

        /* This dispatch's tables: the same textures and samplers, buffers somewhere new. */
        for (uint32_t t = 0; t < layout.table_count && t < 16; t++)
        {
            uint8_t *table = tables + (size_t)(copy * 16 + t) * table_bytes;
            if (layout.table_sgprs[t] < 32)
                user[layout.table_sgprs[t]] = TABLE_BASE + (copy * 16 + t) * table_bytes;
            for (uint32_t i = 0; i < layout.slot_count && i < 256; i++)
            {
                const struct sa5_slot *s = &slots[i];
                uint32_t *w = (uint32_t *)(table + s->dword * 4);
                if (s->table != t)
                    continue;
                if (s->kind == SA5_SLOT_TEXTURE || s->kind == SA5_SLOT_STORAGE)
                {
                    static const uint32_t types[] = { 9, 8, 9, 10, 11, 12, 13, 14, 15 };
                    uint64_t address = IMAGE_BASE + (((image_sets ? it % image_sets : 0) * 16 + i) % 4096) * 0x100;
                    memset(w, 0, s->size * 4);
                    w[0] = (uint32_t)(address >> 8);
                    w[1] = (56u << 20) | (3u << 30);   /* 8_8_8_8_UNORM, width 64 */
                    w[2] = 15u | (63u << 14);           /* width 64, height 64 */
                    w[3] = 0xfacu | (types[s->dim < 9 ? s->dim : 0] << 28);
                }
                else if (s->kind == SA5_SLOT_SAMPLER)
                {
                    w[0] = 0;
                    w[1] = 0xfff000u;
                    w[2] = 0x5000000u;
                    w[3] = 0;
                }
                else
                {
                    uint64_t address = BUFFER_BASE + ((it * 7 + i) % 4096) * 0x100;
                    w[0] = (uint32_t)address;
                    w[1] = (uint32_t)(address >> 32) & 0xffff;
                    w[2] = 0x1000;
                    w[3] = 0xfacu | (77u << 12);
                }
            }
        }

        QueryPerformanceCounter(&t0);
        if (sa5_compile_compute(&req, &result, error, sizeof(error)))
        {
            for (char *p = error; *p; p++)
                if (*p == '\n') *p = ' ';
            printf("FAIL memo: compile %d: %.600s\n", it, error);
            return 1;
        }
        QueryPerformanceCounter(&t1);
        sa5_result_identity(result, identity);
        QueryPerformanceCounter(&t2);
        /* d3d12_xs's binding pass: every binding's words, and a module key from their shape. */
        key = identity[0] ^ identity[1];
        count = sa5_result_binding_count(result);
        for (uint32_t i = 0; i < count; i++)
        {
            struct sa5_binding b;
            sa5_result_binding(result, i, &b);
            key = bench_hash(key, &b.binding, sizeof(b.binding));
            key = bench_hash(key, &b.count, sizeof(b.count));
            if (b.role != 0 || b.kind != 1)
                key = bench_hash(key, &b.role, sizeof(b.role));
        }
        if (new_slots)
            key = bench_hash(key, &it, sizeof(it));
        QueryPerformanceCounter(&t3);
        {
            int known = 0;
            for (int k = 0; k < key_count && !known; k++)
                known = keys[k] == key;
            if (!known)
            {
                /* A new module: guest buffers and host data become pushed pointers, every other
                 * binding moves to a set of its own (d3d12_xs's bindless sets, without a heap). */
                struct sa5_host_buffer hosts[8];
                struct sa5_remap remaps[64];
                static uint32_t indices[4096];
                uint32_t host_count = 0, remap_count = 0, index_count = 0, *words = NULL, word_count = 0;
                for (uint32_t i = 0; i < count; i++)
                {
                    struct sa5_binding b;
                    sa5_result_binding(result, i, &b);
                    if (b.role > 2 || (b.role == 0 && b.kind == 1))
                    {
                        if (host_count < 8)
                        {
                            hosts[host_count] = (struct sa5_host_buffer){ b.set, b.binding, host_count,
                                    b.word_count ? b.word_count : 4, b.role == 0, 0, 0 };
                            host_count++;
                        }
                    }
                    else if (remap_count < 64 && index_count + b.count <= 4096)
                    {
                        for (uint32_t e = 0; e < b.count; e++)
                            indices[index_count + e] = e + (new_slots ? it : 0);
                        remaps[remap_count++] = (struct sa5_remap){ b.set, b.binding, b.kind, 1 + b.kind, 0, b.count,
                                &indices[index_count], 0 };
                        index_count += b.count;
                    }
                }
                if (sa5_finalize(result, remaps, remap_count, hosts, host_count, &words, &word_count, error,
                                 sizeof(error)))
                {
                    for (char *p = error; *p; p++)
                        if (*p == '\n') *p = ' ';
                    printf("     finalize failed: %.300s\n", error);
                }
                else
                    sa5_free_words(words);
                if (key_count < 64)
                    keys[key_count++] = key;
                finalized++;
            }
        }
        QueryPerformanceCounter(&t4);
        /* What the call gave (untimed): its module, bindings and push data, for comparing builds. */
        {
            uint32_t words = 0, push_size = 0;
            const uint32_t *spirv = sa5_result_spirv(result, &words);
            const void *push = sa5_result_push_constants(result, &push_size);
            digest = bench_hash(digest, spirv, words * 4);
            digest = bench_hash(digest, push, push_size);
            for (uint32_t i = 0; i < count; i++)
            {
                struct sa5_binding b;
                sa5_result_binding(result, i, &b);
                digest = bench_hash(digest, &b, offsetof(struct sa5_binding, word_count) + sizeof(b.word_count));
                digest = bench_hash(digest, b.words, b.word_count * 4);
                digest = bench_hash(digest, &b.image_shape, sizeof(b.image_shape) + sizeof(b.role));
            }
        }
        QueryPerformanceCounter(&t4b);
        sa5_result_free(result);
        QueryPerformanceCounter(&t5);
        phase[0] = (t1.QuadPart - t0.QuadPart) * 1000.0 / frequency.QuadPart;
        phase[1] = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
        phase[2] = (t3.QuadPart - t2.QuadPart) * 1000.0 / frequency.QuadPart;
        phase[3] = (t4.QuadPart - t3.QuadPart) * 1000.0 / frequency.QuadPart;
        phase[4] = (t5.QuadPart - t4b.QuadPart) * 1000.0 / frequency.QuadPart;
        for (int p = 0; p < 5; p++)
        {
            if (!it)
                first[p] = phase[p];
            else
                rest[p] += phase[p] / (iterations > 1 ? iterations - 1 : 1);
        }
    }
    printf("memo %s: first call %.3f ms (compile %.3f, identity %.3f, bindings %.3f, module %.3f, free %.3f)\n", name,
           first[0] + first[1] + first[2] + first[3] + first[4], first[0], first[1], first[2], first[3], first[4]);
    printf("memo %s: %d more calls, %.3f ms each (compile %.3f, identity %.3f, bindings %.3f, module %.3f, free %.3f); "
           "%d modules\n", name, iterations - 1, rest[0] + rest[1] + rest[2] + rest[3] + rest[4], rest[0], rest[1],
           rest[2], rest[3], rest[4], finalized);
    printf("memo %s: results digest %016llx\n", name, (unsigned long long)digest);
    free(data);
    free(tables);
    free(buffers);
    free(images);
    return 0;
}

/* SA5_USER=v0,v1,... (hex, from s0): user SGPRs, all but SKIP and SKIP + 1 (a fetch address the
 * caller keeps; -1 for none). */
static void env_user_data(uint32_t *user, unsigned count, int skip)
{
    const char *p = getenv("SA5_USER");

    for (unsigned i = 0; p && i < count && *p; i++)
    {
        char *end;
        uint32_t v = (uint32_t)strtoul(p, &end, 16);
        if ((int)i != skip && (int)i != skip + 1)
            user[i] = v;
        p = *end == ',' ? end + 1 : end;
        if (end == p && *end != ',')
            break;
    }
}

/* SA5_MEMORY=ADDRESS:w0,w1,... (hex): guest memory, up to max words, as one region; or
 * SA5_MEMORY=ADDRESS+BYTES (hex): that many bytes of zeros there; or SA5_MEMORY=ADDRESS@PATH: a
 * file's bytes there (tools/ps-plan.py --draw writes such tables). Returns the region count (0 or 1). */
static uint32_t env_memory(struct sa5_region *region, uint32_t *memory, uint32_t max)
{
    const char *p = getenv("SA5_MEMORY");
    static void *zeros;
    uint64_t address;
    uint32_t words = 0;
    char *end;

    if (!p)
        return 0;
    address = strtoull(p, &end, 16);
    if (*end == '@')
    {
        char winfile[1100];
        static long bytes;
        snprintf(winfile, sizeof(winfile), "%s%s", end[1] == '/' ? "Z:" : "", end + 1);
        for (char *q = winfile; *q; q++)
            if (*q == '/') *q = '\\';
        if (!zeros)
            zeros = read_all(winfile, &bytes);
        *region = (struct sa5_region){ address, zeros, zeros ? (size_t)bytes : 0 };
        return 1;
    }
    if (*end == '+')
    {
        size_t bytes = (size_t)strtoull(end + 1, NULL, 16);
        if (!zeros)
            zeros = calloc(1, bytes);
        *region = (struct sa5_region){ address, zeros, zeros ? bytes : 0 };
        return 1;
    }
    for (p = *end == ':' ? end + 1 : end; *p && words < max; p = *end == ',' ? end + 1 : end)
    {
        memory[words++] = (uint32_t)strtoul(p, &end, 16);
        if (end == p)
            break;
    }
    *region = (struct sa5_region){ address, memory, words * 4u };
    return 1;
}

/* Vertex shaders: plan lines NAME CODE SKIP RSRC1 USER_SGPRS INPUTS, one shader per line. Each
 * input r gets a synthetic RGBA32F vertex buffer (stride 16) and v[4+4r..7+4r]. */
static uint32_t memory[64];
static struct sa5_region region;

static int vertex_sweep(const char *plan, int first, int count)
{
    FILE *f = fopen(plan, "r");
    char line[4096];
    int index = 0, ok = 0, failed = 0, valid_count = 0;

    if (!f)
    {
        printf("FAIL open %s\n", plan);
        return 1;
    }
    while (fgets(line, sizeof(line), f) && (count <= 0 || ok + failed < count))
    {
        char name[64], path[1024], winpath[1100], error[600];
        unsigned skip, rsrc1, user, inputs;
        struct sa5_vertex_request req = { 0 };
        struct sa5_result *result = NULL;
        uint32_t user_data[32] = { 0 }, words = 0;
        uint8_t *data;
        long size;

        if (line[0] == '#' || sscanf(line, "%63s %1023s %u %x %u %u", name, path, &skip, &rsrc1, &user, &inputs) != 6)
            continue;
        if (index++ < first)
            continue;
        snprintf(winpath, sizeof(winpath), "%s%s", path[0] == '/' ? "Z:" : "", path);
        for (char *p = winpath; *p; p++)
            if (*p == '/') *p = '\\';
        if (!(data = read_all(winpath, &size)) || size <= (long)skip)
        {
            printf("vs %s FAIL unreadable\n", name);
            failed++;
            free(data);
            continue;
        }
        req.code = (const uint32_t *)(data + skip);
        req.code_dwords = (uint32_t)((size - skip) / 4);
        req.code_address = 0x10000000;
        req.wave = 64;
        req.user_data = user_data;
        req.user_data_count = user > 32 ? 32 : user;
        /* s[4:5]: the fetch shader's address, outside the program (AnyPS5 then takes the call for
         * the fetch it stands in for). */
        user_data[4] = 0x30000000u;
        req.rsrc1 = rsrc1;
        req.input_count = inputs > 32 ? 32 : inputs;
        for (unsigned r = 0; r < req.input_count; r++)
        {
            req.inputs[r].vsharp[0] = 0x20000000u + r * 0x100000u;
            req.inputs[r].vsharp[1] = 16u << 16;
            req.inputs[r].vsharp[2] = 1024;
            req.inputs[r].vsharp[3] = (77u << 12) | 0xfacu;
            req.inputs[r].register_start = 4 + 4 * r;
            req.inputs[r].register_count = 4;
        }
        req.push_size = getenv("SA5_PUSH_SIZE") ? (uint32_t)atoi(getenv("SA5_PUSH_SIZE")) : 0;
        /* SA5_USER=v0,v1,... (hex, from s0; s[4:5] stays the fetch address) and
         * SA5_MEMORY=ADDRESS:w0,w1,... (hex): a draw's user SGPRs and guest memory, as d3d12_xs logs
         * them, to follow the descriptors a shader resolves. */
        env_user_data(user_data, 32, 4);
        if (env_memory(&region, memory, 64))
        {
            req.regions = &region;
            req.region_count = 1;
        }
        if (sa5_compile_vertex(&req, &result, error, sizeof(error)))
        {
            for (char *p = error; *p; p++)
                if (*p == '\n') *p = ' ';
            printf("vs %s FAIL %.400s\n", name, error);
            failed++;
        }
        else
        {
            const uint32_t *spirv = sa5_result_spirv(result, &words);
            int valid = !sa5_validate(spirv, words, error, sizeof(error));
            for (char *p = error; *p; p++)
                if (*p == '\n') *p = ' ';
            printf("vs %s ok %u words, %u vertex inputs, %u bindings, %s%.300s\n", name, words,
                   sa5_result_vertex_input_count(result), sa5_result_binding_count(result),
                   valid ? "valid" : "INVALID: ", valid ? "" : error);
            for (uint32_t i = 0; i < sa5_result_binding_count(result) && getenv("SA5_MEMORY"); i++)
            {
                struct sa5_binding b;
                sa5_result_binding(result, i, &b);
                printf("     binding role %u kind %u count %u words", b.role, b.kind, b.count);
                for (uint32_t w = 0; w < b.word_count && w < 8; w++)
                    printf(" %08x", b.words[w]);
                printf("\n");
            }
            /* SA5_PUSH_SIZE set: finalize as a graphics stage does, host buffers' addresses pushed
             * after the stage's push data (the guest buffer table) or baked (the rest), and with
             * SA5_REMAP=1 each guest descriptor array (images, samplers, image and sampler tables)
             * moved onto a bindless array whose heap indices come from a draw block, as d3d12_xs
             * does (sa5_finalize_draw). */
            if (getenv("SA5_PUSH_SIZE"))
            {
                struct sa5_host_buffer hosts[8];
                struct sa5_remap remaps[32];
                uint32_t host_count = 0, remap_count = 0, index_base = 0, *out = NULL, out_count = 0, pushed = 0;
                int remap = getenv("SA5_REMAP") != NULL;
                for (uint32_t i = 0; i < sa5_result_binding_count(result) && host_count < 8; i++)
                {
                    struct sa5_binding b;
                    sa5_result_binding(result, i, &b);
                    if (b.role == 0 && b.kind == 1)
                        hosts[host_count++] = (struct sa5_host_buffer){ b.set, b.binding, pushed++, 0, 1,
                                req.push_size, 0 };
                    else if (remap && (b.role <= 2 || b.role == 8 || b.role == 9 || b.role == 11) && remap_count < 32)
                    {
                        remaps[remap_count++] = (struct sa5_remap){ b.set, b.binding, b.kind, 0, 100 + b.kind,
                                b.count ? b.count : 1, NULL, index_base };
                        index_base += b.count ? b.count : 1;
                    }
                    else if (b.role > 2)
                        hosts[host_count++] = (struct sa5_host_buffer){ b.set, b.binding, host_count, b.word_count, 0,
                                0, 0x123450000ull };
                }
                if (remap ? sa5_finalize_draw(result, remaps, remap_count, hosts, host_count, req.push_size,
                                              &out, &out_count, error, sizeof(error))
                          : sa5_finalize(result, NULL, 0, hosts, host_count, &out, &out_count, error, sizeof(error)))
                {
                    for (char *p = error; *p; p++)
                        if (*p == '\n') *p = ' ';
                    printf("vs %s finalize FAIL (%u host buffers, %u pushed): %.300s\n", name, host_count, pushed, error);
                }
                else
                {
                    printf("vs %s finalize ok (%u host buffers, %u pushed)\n", name, host_count, pushed);
                    sa5_free_words(out);
                }
            }
            valid_count += valid;
            sa5_result_free(result);
            ok++;
        }
        fflush(stdout);
        free(data);
    }
    fclose(f);
    printf("%s vs-check: %d of %d vertex shaders compile, %d of those pass spirv-val\n",
           failed ? "info" : "ok  ", ok, ok + failed, valid_count);
    return 0;
}

/* Mesh shaders: compiled with d3d12_xs's settings (wave and limits from the FSSM, 32 user SGPRs,
 * push data 0-95), then finalized with a draw block at byte 96: the guest buffer table pushed, the
 * other host buffers baked. Returns 0 and a "ok ..." detail, or -1 and the error. */
static int mesh_compile(const struct sa5_mesh_request *req, const char *name, char *detail, size_t size)
{
    struct sa5_result *result = NULL;
    struct sa5_host_buffer hosts[8];
    uint32_t host_count = 0, pushed = 0, words = 0, *out = NULL, out_count = 0;
    char error[600];
    const uint32_t *spirv;

    if (sa5_compile_mesh(req, &result, error, sizeof(error)))
    {
        snprintf(detail, size, "FAIL %.400s", error);
        return -1;
    }
    spirv = sa5_result_spirv(result, &words);
    if (sa5_validate(spirv, words, error, sizeof(error)))
    {
        snprintf(detail, size, "INVALID %u words: %.300s", words, error);
        sa5_result_free(result);
        return -1;
    }
    for (uint32_t i = 0; i < sa5_result_binding_count(result) && host_count < 8; i++)
    {
        struct sa5_binding b;
        sa5_result_binding(result, i, &b);
        if (b.role == 0 && b.kind == 1)
            hosts[host_count++] = (struct sa5_host_buffer){ b.set, b.binding, pushed++, 0, 1, 0, 0 };
        else if (b.role > 2)
            hosts[host_count++] = (struct sa5_host_buffer){ b.set, b.binding, 0, b.word_count, 0, 0, 0x123450000ull };
    }
    if (sa5_finalize_draw(result, NULL, 0, hosts, host_count, 96, &out, &out_count, error, sizeof(error)))
    {
        snprintf(detail, size, "finalize FAIL (%u bindings): %.300s", sa5_result_binding_count(result), error);
        sa5_result_free(result);
        return -1;
    }
    snprintf(detail, size, "ok %u words, wave%u, %u bindings, subgroup %u; finalized %u words", words, req->wave,
             sa5_result_binding_count(result), sa5_result_subgroup_size(result), out_count);
    if (name && getenv("SA5_DUMP_SPIRV"))
    {
        char path[1200];
        FILE *dump;
        snprintf(path, sizeof(path), "%s\\%s.spv", getenv("SA5_DUMP_SPIRV"), name);
        if ((dump = fopen(path, "wb")))
        {
            fwrite(out, 4, out_count, dump);
            fclose(dump);
        }
    }
    sa5_free_words(out);
    sa5_result_free(result);
    return 0;
}

static int mesh_builtin(void)
{
    /* Wave64: three vertices and one triangle (m0 = vertices | primitives << 12), lane n at
     * (n, 0, 0, 1), the triangle from lane 0. */
    static const uint32_t code[] = {
        0xd7650000, 0x000100c1,             /* v_mbcnt_lo_u32_b32 v0, -1, 0 */
        0xd7660000, 0x000200c1,             /* v_mbcnt_hi_u32_b32 v0, -1, v0 */
        0xb07c1003,                         /* s_movk_i32 m0, 0x1003 */
        0xbf900009,                         /* s_sendmsg GS_ALLOC_REQ */
        0x7d880083,                         /* v_cmp_gt_u32 vcc, 3, v0 */
        0xbefe046a,                         /* s_mov_b64 exec, vcc */
        0x7e020d00,                         /* v_cvt_f32_u32 v1, v0 */
        0x7e040280,                         /* v_mov_b32 v2, 0 */
        0x7e0602f2,                         /* v_mov_b32 v3, 1.0 */
        0xf80008cf, 0x03020201,             /* exp pos0 v1, v2, v2, v3 done */
        0xf800020f, 0x03030303,             /* exp param0 v3, v3, v3, v3 */
        0x7d840080,                         /* v_cmp_eq_u32 vcc, 0, v0 */
        0xbefe046a,                         /* s_mov_b64 exec, vcc */
        0x7e0802ff, 0x00200400,             /* v_mov_b32 v4, 0 | 1 << 10 | 2 << 20 */
        0xf8000941, 0x00000004,             /* exp prim v4 done */
        0xbf810000,                         /* s_endpgm */
    };
    uint32_t user[32] = { 0 };
    struct sa5_mesh_request req = { 0 };
    char detail[700];
    int failed;

    req.code = code;
    req.code_dwords = sizeof(code) / 4;
    req.code_address = 0x10000000;
    req.wave = 64;
    req.user_data = user;
    req.user_data_count = 32;
    req.rsrc1 = 0x40000000u | 1u << 21 | 0xc0u << 12;
    req.threads = 128;
    req.max_vertices = 128;
    req.max_primitives = 128;
    req.primitive_vertices = 3;
    req.push_size = 96;
    req.use_cache = 1;
    failed = mesh_compile(&req, "mesh", detail, sizeof(detail));
    printf("%s mesh: %s\n", failed ? "FAIL" : "ok  ", detail);
    return failed;
}

/* Plan lines: NAME CODE SKIP DWORDS WAVE THREADS MAX_VERTICES MAX_PRIMITIVES PRIMITIVE_VERTICES LDS
 * CLIP_CNTL RSRC1 (hex for the last two), the code at byte SKIP of the file CODE. */
static int mesh_sweep(const char *plan, int first, int count)
{
    FILE *f = fopen(plan, "r");
    char line[4096];
    int index = 0, ok = 0, failed = 0;

    if (!f)
    {
        printf("FAIL open %s\n", plan);
        return 1;
    }
    while (fgets(line, sizeof(line), f) && (count <= 0 || ok + failed < count))
    {
        char name[64], path[1024], winpath[1100], detail[700];
        unsigned skip, dwords, wave, threads, maxv, maxp, verts, lds, clip, rsrc1;
        struct sa5_mesh_request req = { 0 };
        uint32_t user[32] = { 0 }, memory[64];
        struct sa5_region region;
        uint8_t *data;
        long size;

        if (line[0] == '#' || sscanf(line, "%63s %1023s %u %u %u %u %u %u %u %u %x %x", name, path, &skip, &dwords, &wave,
                                     &threads, &maxv, &maxp, &verts, &lds, &clip, &rsrc1) != 12)
            continue;
        if (index++ < first)
            continue;
        env_user_data(user, 32, -1);
        snprintf(winpath, sizeof(winpath), "%s%s", path[0] == '/' ? "Z:" : "", path);
        for (char *p = winpath; *p; p++)
            if (*p == '/') *p = '\\';
        if (!(data = read_all(winpath, &size)) || size < (long)(skip + 4ull * dwords))
        {
            printf("ms %s FAIL unreadable\n", name);
            failed++;
            free(data);
            continue;
        }
        req.code = (const uint32_t *)(data + skip);
        req.code_dwords = dwords;
        req.code_address = 0x10000000;
        req.wave = wave;
        req.user_data = user;
        req.user_data_count = 32;
        req.rsrc1 = rsrc1;
        req.threads = threads;
        req.max_vertices = maxv;
        req.max_primitives = maxp;
        req.primitive_vertices = verts;
        req.lds_bytes = lds;
        req.clip_cntl = clip;
        req.push_size = 96;
        req.use_cache = 1;
        req.region_count = env_memory(&region, memory, 64);
        req.regions = req.region_count ? &region : NULL;
        if (mesh_compile(&req, name, detail, sizeof(detail)))
            failed++;
        else
            ok++;
        for (char *p = detail; *p; p++)
            if (*p == '\n') *p = ' ';
        printf("ms %s %s\n", name, detail);
        fflush(stdout);
        free(data);
    }
    fclose(f);
    printf("%s ms-check: %d of %d mesh shaders compile, validate and finalize\n", failed ? "info" : "ok  ", ok,
           ok + failed);
    return 0;
}

/* Loops in a module's entry point (the guest program; AnyPS5's helper functions have bounded loops
 * of their own): each OpLoopMerge, and whether some OpBranchConditional can leave it (to its merge
 * block; AnyPS5's loop guard, APS5_LOOP_GUARD, forces those exits). */
static void count_loops(const uint32_t *spirv, uint32_t words, int *loops, int *no_exit)
{
    enum { MAX_LOOPS = 512 };
    static uint32_t merge[MAX_LOOPS], conditional_targets[1 << 17];
    uint32_t n = 0, targets = 0, entry = 0, function = 0;

    *loops = *no_exit = 0;
    for (uint32_t i = 5; i < words;)
    {
        uint32_t op = spirv[i] & 0xffff, len = spirv[i] >> 16;
        if (!len || i + len > words)
            break;
        if (op == 15 /* OpEntryPoint */ && !entry)
            entry = spirv[i + 2];
        else if (op == 54 /* OpFunction */)
            function = spirv[i + 2];
        else if (op == 246 /* OpLoopMerge */ && n < MAX_LOOPS && function == entry)
            merge[n++] = spirv[i + 1];
        else if (op == 250 /* OpBranchConditional */ && targets + 2 <= ARRAYSIZE(conditional_targets))
        {
            conditional_targets[targets++] = spirv[i + 2];
            conditional_targets[targets++] = spirv[i + 3];
        }
        i += len;
    }
    for (uint32_t l = 0; l < n; l++)
    {
        int exits = 0;
        for (uint32_t t = 0; t < targets && !exits; t++)
            exits = conditional_targets[t] == merge[l];
        *no_exit += !exits;
    }
    *loops = (int)n;
}

/* Compute shaders: plan lines NAME CODE SKIP RSRC1 RSRC2 X,Y,Z [WAVE] (as for memo), compiled with
 * no guest memory and validated; with their loops counted (count_loops) and a hash of the SPIR-V.
 * SA5_DUMP_SPIRV=DIR writes each module to DIR\NAME.spv. */
static int compute_sweep(const char *plan, int first, int count)
{
    FILE *f = fopen(plan, "r");
    char line[4096];
    int index = 0, ok = 0, failed = 0, valid_count = 0, all_loops = 0, all_no_exit = 0;

    if (!f)
    {
        printf("FAIL open %s\n", plan);
        return 1;
    }
    while (fgets(line, sizeof(line), f) && (count <= 0 || ok + failed < count))
    {
        char name[64], path[1024], winpath[1100], error[600], ignored[64];
        unsigned skip, rsrc1, rsrc2, wave = 0;
        uint32_t user[32] = { 0 }, words = 0;
        struct sa5_compute_request req = { 0 };
        struct sa5_result *result = NULL;
        struct sa5_layout layout;
        static struct sa5_slot slots[256];
        uint8_t *data;
        long size;

        if (line[0] == '#' || sscanf(line, "%63s %1023s %u %x %x %u,%u,%u %u", name, path, &skip, &rsrc1, &rsrc2,
                                     &req.threads[0], &req.threads[1], &req.threads[2], &wave) < 8)
            continue;
        if (index++ < first)
            continue;
        snprintf(winpath, sizeof(winpath), "%s%s", path[0] == '/' ? "Z:" : "", path);
        for (char *p = winpath; *p; p++)
            if (*p == '/') *p = '\\';
        if (!(data = read_all(winpath, &size)) || size <= (long)skip)
        {
            printf("cs %s FAIL unreadable\n", name);
            free(data);
            failed++;
            continue;
        }
        req.code = (const uint32_t *)(data + skip);
        req.code_dwords = (uint32_t)((size - skip) / 4);
        req.code_address = 0x10000000;
        env_user_data(user, 32, -1);
        req.user_data = user;
        req.user_data_count = (rsrc2 >> 1) & 0x1f;
        req.rsrc1 = rsrc1;
        req.rsrc2 = rsrc2;
        req.region_count = env_memory(&region, memory, 64);
        req.regions = req.region_count ? &region : NULL;
        req.use_cache = getenv("SA5_USE_CACHE") != NULL;
        req.wave = wave ? wave
                        : !sa5_analyse(req.code, req.code_dwords, req.user_data_count, &layout, slots, 0, ignored,
                                       sizeof(ignored)) ? layout.wave : 32;
        if (sa5_compile_compute(&req, &result, error, sizeof(error)))
        {
            for (char *p = error; *p; p++)
                if (*p == '\n') *p = ' ';
            printf("cs %s FAIL %.300s\n", name, error);
            failed++;
        }
        else
        {
            const uint32_t *spirv = sa5_result_spirv(result, &words);
            int valid = !sa5_validate(spirv, words, error, sizeof(error)), loops, no_exit;
            uint64_t hash = 0xcbf29ce484222325ull;
            for (uint32_t i = 0; i < words; i++)
                hash = (hash ^ spirv[i]) * 0x100000001b3ull;
            for (char *p = error; *p; p++)
                if (*p == '\n') *p = ' ';
            count_loops(spirv, words, &loops, &no_exit);
            if (getenv("SA5_DUMP_SPIRV"))
            {
                char out[1200];
                FILE *dump;
                snprintf(out, sizeof(out), "%s\\%s.spv", getenv("SA5_DUMP_SPIRV"), name);
                if ((dump = fopen(out, "wb")))
                {
                    fwrite(spirv, 4, words, dump);
                    fclose(dump);
                }
            }
            printf("cs %s ok %u words, hash %016llx, %d loops (%d without a conditional branch to their merge), %u bindings, %s%.300s\n",
                   name, words, (unsigned long long)hash, loops, no_exit, sa5_result_binding_count(result),
                   valid ? "valid" : "INVALID: ", valid ? "" : error);
            for (uint32_t i = 0; i < sa5_result_binding_count(result) && getenv("SA5_MEMORY"); i++)
            {
                struct sa5_binding b;
                sa5_result_binding(result, i, &b);
                printf("     binding role %u kind %u count %u words", b.role, b.kind, b.count);
                for (uint32_t w = 0; w < b.word_count && w < 8; w++)
                    printf(" %08x", b.words[w]);
                printf("\n");
            }
            valid_count += valid;
            all_loops += loops;
            all_no_exit += no_exit;
            sa5_result_free(result);
            ok++;
        }
        fflush(stdout);
        free(data);
    }
    fclose(f);
    printf("%s cs-check: %d of %d compute shaders compile, %d of those pass spirv-val; %d loops, %d without a "
           "conditional branch to their merge\n", failed ? "info" : "ok  ", ok, ok + failed, valid_count, all_loops, all_no_exit);
    return 0;
}

int main(int argc, char **argv)
{
    /* Mesh modules, and vertex modules without SA5_REMAP=1, are finalized with only their host
     * buffers bound: their guest descriptors stay where AnyPS5 put them, which sa5_finalize
     * otherwise refuses (vkd3d-proton's layout has none of them). */
    if (argc >= 2 && (!strcmp(argv[1], "mesh") || !strcmp(argv[1], "ms") || (!strcmp(argv[1], "vs") && !getenv("SA5_REMAP")))
            && !getenv("SA5_FINALIZE_PARTIAL"))
        _putenv("SA5_FINALIZE_PARTIAL=1");
    if (argc == 2 && !strcmp(argv[1], "mesh"))
        return mesh_builtin();
    if (argc > 2 && !strcmp(argv[1], "ms"))
        return mesh_sweep(argv[2], argc > 3 ? atoi(argv[3]) : 0, argc > 4 ? atoi(argv[4]) : 0);
    if (argc > 2 && !strcmp(argv[1], "ps"))
        return pixel_sweep(argv[2], argc > 3 ? atoi(argv[3]) : 0, argc > 4 ? atoi(argv[4]) : 0);
    if (argc > 3 && !strcmp(argv[1], "bench"))
        return bench(atoi(argv[2]), argv[3], argc > 4 ? atoi(argv[4]) : 0, argc > 5 ? atoi(argv[5]) : 0);
    if (argc > 3 && !strcmp(argv[1], "memo"))
        return memo_bench(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 200);
    if (argc > 2 && !strcmp(argv[1], "vs"))
        return vertex_sweep(argv[2], argc > 3 ? atoi(argv[3]) : 0, argc > 4 ? atoi(argv[4]) : 0);
    if (argc > 2 && !strcmp(argv[1], "cs"))
        return compute_sweep(argv[2], argc > 3 ? atoi(argv[3]) : 0, argc > 4 ? atoi(argv[4]) : 0);

    static const uint32_t endpgm[] = { 0xbf810000u };
    uint32_t threads[3] = { 1, 1, 1 };

    if (argc < 6)
        return compile(endpgm, 1, 0, 0, threads);
    {
        FILE *f = fopen(argv[1], "rb");
        long size, skip = strtol(argv[2], NULL, 0);
        uint8_t *data;

        if (!f)
        {
            printf("FAIL open %s\n", argv[1]);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        size = ftell(f);
        fseek(f, 0, SEEK_SET);
        data = malloc(size);
        fread(data, 1, size, f);
        fclose(f);
        sscanf(argv[5], "%u,%u,%u", &threads[0], &threads[1], &threads[2]);
        return compile((const uint32_t *)(data + skip), (uint32_t)((size - skip) / 4),
                       (uint32_t)strtoul(argv[3], NULL, 0), (uint32_t)strtoul(argv[4], NULL, 0), threads);
    }
}
