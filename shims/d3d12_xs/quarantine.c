/* Quarantine of console programs across runs.
 *
 * REBOX_QUARANTINE=<file> (a Unix path, or a Windows one): when the device is lost, a few of the
 * console programs that ran in the seconds before are appended to the file, and every later run
 * skips those programs, as REBOX_SKIP_PROGRAMS leaves programs out by name. Each run that loses the
 * device puts a few more suspects aside, so an unattended loop of runs gets further each time.
 *   REBOX_QUARANTINE_NEW=<n>       programs added per loss (default 4, at most 16)
 *   REBOX_QUARANTINE_WINDOW=<s>    how recently before the loss a program must have run (default
 *                                  20 s: a hang is noticed only at the kernel's timeout)
 *   REBOX_QUARANTINE_GRAPHICS=0    compute programs only (by default graphics stages too)
 *
 * Which ones: of the programs that ran in the window, those whose current spell of use began
 * latest, a spell being uses with no gap over 2 s between them. A program the title just started
 * using (new, or back after a pause: an event, a new area) comes before one it has used every frame
 * for minutes, which a fault minutes in is less likely to be. "Ran" means vkd3d-proton asked for its
 * module for a dispatch or draw and got one (the command processor runs ahead of the CPU's
 * recording only a little). The loss is seen by a thread polling the device every 200 ms, or by
 * proxy.c's GPU memory waits, whichever is first.
 *
 * The file holds one program per line, by the id "console shader <id>: compiled a variant" logs
 * (the code's hash), in hex; '#' starts a comment. Delete a line to let a program run again.
 *
 * A skipped dispatch writes nothing: whatever reads its output (indirect arguments included, which
 * vkd3d-proton clamps, patch 0017) reads what was there before. A skipped vertex or mesh stage's
 * draws get the PSO's fallback pipeline; a skipped pixel stage's can get the stand-in pixel stage
 * (REBOX_STANDIN_PIXEL), else the fallback. */

#define COBJMACROS
#include <windows.h>
#include <d3d12.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "console_compile.h"
#include "gpu_memory.h"
#include "quarantine.h"

#define MAX_PROGRAMS 16384 /* a power of two */
#define MAX_BLOBS 16384
#define MAX_NEW 16
#define SPELL_GAP_MS 2000

struct program
{
    UINT64 hash;                /* 0: an empty slot */
    UINT32 stage;               /* the deferred header's: 0x20 compute, 1 vertex, 0x10 pixel, 0x80 mesh; 0 unseen */
    BOOL quarantined;
    volatile LONG skip_logged;
    /* GetTickCount64, 0 never: its first use, the start of its current spell, its last use. */
    volatile LONG64 first_ran, spell, last_ran;
};

/* A deferred shader blob vkd3d-proton keeps with its PSO, and its program: hashing the code at
 * every dispatch would cost more than the dispatch. A freed blob's address can come back for
 * another shader, so the header is compared too. */
struct blob
{
    const void *shader;
    UINT32 size, code_dwords;
    UINT64 code_address;
    struct program *program;
};

static INIT_ONCE init_once = INIT_ONCE_STATIC_INIT;
static BOOL enabled, graphics_too = TRUE;
static char path[MAX_PATH + 8];
static UINT new_count = 4;
static ULONGLONG window_ms = 20000;
static SRWLOCK lock = SRWLOCK_INIT;
static struct program programs[MAX_PROGRAMS];
static UINT program_count;
static struct blob blobs[MAX_BLOBS];
static volatile LONG lost;

static const char *stage_name(UINT32 stage)
{
    switch (stage)
    {
    case 0x20: return "compute";
    case 0x1: return "vertex";
    case 0x10: return "pixel";
    case 0x80: return "mesh";
    default: return "program";
    }
}

/* Under lock, exclusive. NULL when full. */
static struct program *program_add(UINT64 hash)
{
    for (UINT i = 0, slot = (UINT)(hash ^ hash >> 32) & (MAX_PROGRAMS - 1); i < MAX_PROGRAMS;
         i++, slot = (slot + 1) & (MAX_PROGRAMS - 1))
    {
        if (programs[slot].hash == hash)
            return &programs[slot];
        if (!programs[slot].hash)
        {
            if (program_count >= MAX_PROGRAMS / 2)
                return NULL;
            program_count++;
            programs[slot].hash = hash;
            return &programs[slot];
        }
    }
    return NULL;
}

static UINT blob_slot(const void *shader)
{
    UINT64 p = (UINT64)(ULONG_PTR)shader;
    return (UINT)((p >> 4) ^ (p >> 18)) & (MAX_BLOBS - 1);
}

static BOOL blob_matches(const struct blob *b, const struct rebox_compile_in *in)
{
    const struct rebox_deferred_shader *h = in->shader;
    return b->shader == in->shader && b->size == in->shader_size && b->code_dwords == h->code_dwords
            && b->code_address == h->code_address;
}

static DWORD WINAPI watch_device(void *arg)
{
    while (!lost)
    {
        ID3D12Device *device;
        Sleep(200);
        if ((device = rebox_shared_device()) && FAILED(ID3D12Device_GetDeviceRemovedReason(device)))
        {
            quarantine_device_lost();
            break;
        }
    }
    return 0;
}

static BOOL CALLBACK init(INIT_ONCE *once, void *param, void **context)
{
    char *value = rebox_getenv("REBOX_QUARANTINE"), line[256], list[1024] = "";
    UINT count = 0;
    size_t used = 0;
    HANDLE thread;
    FILE *f;

    if (!value || !*value)
    {
        free(value);
        return TRUE;
    }
    /* A Unix path is on Wine's Z: drive. */
    snprintf(path, sizeof(path), "%s%s", value[0] == '/' ? "Z:" : "", value);
    free(value);
    if ((value = rebox_getenv("REBOX_QUARANTINE_NEW")))
    {
        new_count = min(max(atoi(value), 1), MAX_NEW);
        free(value);
    }
    if ((value = rebox_getenv("REBOX_QUARANTINE_WINDOW")))
    {
        if (atof(value) > 0)
            window_ms = (ULONGLONG)(atof(value) * 1000.0);
        free(value);
    }
    if ((value = rebox_getenv("REBOX_QUARANTINE_GRAPHICS")))
    {
        graphics_too = value[0] != '0';
        free(value);
    }
    if ((f = fopen(path, "r")))
    {
        while (fgets(line, sizeof(line), f))
        {
            char *p = line, *end;
            UINT64 hash;
            struct program *program;
            while (*p == ' ' || *p == '\t')
                p++;
            if (*p == '#' || !(hash = strtoull(p, &end, 16)) || end == p || !(program = program_add(hash))
                    || program->quarantined)
                continue;
            program->quarantined = TRUE;
            count++;
            if (used + 20 < sizeof(list))
                used += snprintf(list + used, sizeof(list) - used, " %016llx", (unsigned long long)hash);
            else if (used + 5 < sizeof(list))
                used += snprintf(list + used, sizeof(list) - used, " ...");
        }
        fclose(f);
    }
    rebox_log("d3d12_xs: quarantine: %u console program(s) quarantined by %s, skipped:%s; a device loss adds %u of "
              "the %s programs that ran in its last %.0f s, those whose spell of use began latest", count, path,
              count ? list : " none", new_count, graphics_too ? "compute and graphics" : "compute", window_ms / 1000.0);
    enabled = TRUE;
    if ((thread = CreateThread(NULL, 0, watch_device, NULL, 0, NULL)))
        CloseHandle(thread);
    return TRUE;
}

BOOL quarantine_skips(const struct rebox_compile_in *in, void **out)
{
    const struct rebox_deferred_shader *h = in ? in->shader : NULL;
    struct program *program = NULL;
    struct blob *b;
    UINT64 hash;

    *out = NULL;
    if (!h || in->shader_size < sizeof(*h) || (h->version != 1 && h->version != 2))
        return FALSE;
    InitOnceExecuteOnce(&init_once, init, NULL, NULL);
    if (!enabled || (h->version == 2 && !graphics_too))
        return FALSE;

    b = &blobs[blob_slot(h)];
    AcquireSRWLockShared(&lock);
    if (blob_matches(b, in))
        program = b->program;
    ReleaseSRWLockShared(&lock);
    if (!program)
    {
        if (!(hash = console_program_hash(in->shader, in->shader_size)))
            return FALSE;
        AcquireSRWLockExclusive(&lock);
        /* One blob per slot: a collision replaces the older blob, which is hashed again if it comes back. */
        if ((program = program_add(hash)))
        {
            *b = (struct blob){ in->shader, in->shader_size, h->code_dwords, h->code_address, program };
            program->stage = h->stage;
        }
        ReleaseSRWLockExclusive(&lock);
        if (!program)
            return FALSE;
    }
    *out = program;
    if (!program->quarantined)
        return FALSE;
    if (!InterlockedExchange(&program->skip_logged, 1))
        rebox_log("d3d12_xs: quarantine: console %s shader %016llx is quarantined (REBOX_QUARANTINE): skipped",
                  stage_name(program->stage), (unsigned long long)program->hash);
    return TRUE;
}

void quarantine_ran(void *p)
{
    struct program *program = p;
    LONG64 now, last;
    if (!program)
        return;
    now = (LONG64)GetTickCount64();
    InterlockedCompareExchange64(&program->first_ran, now, 0);
    last = InterlockedExchange64(&program->last_ran, now);
    if (!last || now - last > SPELL_GAP_MS)
        InterlockedExchange64(&program->spell, now);
}

void quarantine_device_lost(void)
{
    struct program *picked[MAX_NEW];
    UINT picked_count = 0, recent = 0, quarantined = 0;
    char old_list[1024] = "", new_list[2048] = "", when[64];
    size_t old_used = 0, new_used = 0;
    ULONGLONG now = GetTickCount64();
    SYSTEMTIME st;
    FILE *f;

    if (!enabled || InterlockedExchange(&lost, 1))
        return;

    AcquireSRWLockExclusive(&lock);
    for (UINT i = 0; i < MAX_PROGRAMS; i++)
    {
        struct program *p = &programs[i];
        UINT j;
        if (!p->hash)
            continue;
        if (p->quarantined)
        {
            quarantined++;
            if (old_used + 20 < sizeof(old_list))
                old_used += snprintf(old_list + old_used, sizeof(old_list) - old_used, " %016llx",
                                     (unsigned long long)p->hash);
            continue;
        }
        if (!p->last_ran || now - (ULONGLONG)p->last_ran > window_ms)
            continue;
        recent++;
        /* Keep the new_count whose spells began latest, latest first: insertion into a short list. */
        for (j = picked_count; j > 0 && picked[j - 1]->spell < p->spell; j--)
            if (j < new_count)
                picked[j] = picked[j - 1];
        if (j < new_count)
        {
            picked[j] = p;
            picked_count = min(picked_count + 1, new_count);
        }
    }
    for (UINT i = 0; i < picked_count; i++)
        picked[i]->quarantined = TRUE;
    ReleaseSRWLockExclusive(&lock);

    GetLocalTime(&st);
    snprintf(when, sizeof(when), "%04u-%02u-%02u %02u:%02u:%02u", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute,
             st.wSecond);
    if ((f = fopen(path, "a")))
    {
        fprintf(f, "# %s: device lost; %u of the %u programs that ran in its last %.0f s\n", when, picked_count,
                recent, window_ms / 1000.0);
        for (UINT i = 0; i < picked_count; i++)
            fprintf(f, "%016llx  # %s; before the loss: first ran %.1f s, this spell from %.1f s, last %.1f s\n",
                    (unsigned long long)picked[i]->hash, stage_name(picked[i]->stage),
                    (now - (ULONGLONG)picked[i]->first_ran) / 1000.0, (now - (ULONGLONG)picked[i]->spell) / 1000.0,
                    (now - (ULONGLONG)picked[i]->last_ran) / 1000.0);
        fclose(f);
    }
    else
        rebox_log("d3d12_xs: quarantine: can't append to %s", path);

    for (UINT i = 0; i < picked_count; i++)
        if (new_used + 96 < sizeof(new_list))
            new_used += snprintf(new_list + new_used, sizeof(new_list) - new_used,
                                 " %016llx (%s, first ran %.1f s before, this spell from %.1f s);",
                                 (unsigned long long)picked[i]->hash, stage_name(picked[i]->stage),
                                 (now - (ULONGLONG)picked[i]->first_ran) / 1000.0,
                                 (now - (ULONGLONG)picked[i]->spell) / 1000.0);
    rebox_log("d3d12_xs: quarantine: device lost; already quarantined (%u):%s", quarantined,
              quarantined ? old_list : " none");
    rebox_log("d3d12_xs: quarantine: newly quarantined in %s, %u of the %u programs that ran in the last %.0f s:%s",
              path, picked_count, recent, window_ms / 1000.0, picked_count ? new_list : " none");
}
