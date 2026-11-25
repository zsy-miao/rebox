/* Fault reports for first runs of a title.
 *
 * Wine's own backtrace names addresses in the dumped exe only by number. This
 * vectored handler logs the first few faults (access violations, illegal or
 * privileged instructions, integer divide by zero, stack overflow) with the
 * module and RVA of the faulting instruction, its bytes, the registers and a
 * stack walk through the images' .pdata unwind data. An access violation also
 * says whether the address lies in the GPU window (docs/d3d12x.md), where a
 * unified-memory mistake would land.
 *
 * Reports are first-chance: the exception then continues to the title's own
 * handlers unchanged, so a fault the title expects and handles is reported
 * too ("first-chance" in the log). xmem.dll is imported by the exe and most
 * package DLLs, so this is installed before any title code runs.
 *
 * REBOX_CRASH_LOG=N sets how many faults are reported (default 8, 0 off).
 *
 * Hangs: d3d12_xs calls ReboxHeartbeat on every present. When none comes for
 * REBOX_HANG_DUMP seconds (default 60, 0 off), every thread's stack is
 * logged once, with its name, to see what a title is waiting for; after the
 * next present it can happen again (three times at most).
 *
 * Slow frames: REBOX_STACK_SAMPLE=N[,COUNT] logs every thread's stack every
 * N seconds, COUNT times (default 10), to see where a title that presents but
 * slowly spends its time. */

#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "crash.h"

static LONG remaining;
static DWORD busy_slot = TLS_OUT_OF_INDEXES; /* set while this thread is reporting (__thread needs a DLL) */
static ULONG_PTR window_base, window_size;

static const char *code_name(DWORD code)
{
    switch (code)
    {
    case EXCEPTION_ACCESS_VIOLATION: return "access violation";
    case EXCEPTION_ILLEGAL_INSTRUCTION: return "illegal instruction";
    case EXCEPTION_PRIV_INSTRUCTION: return "privileged instruction";
    case EXCEPTION_INT_DIVIDE_BY_ZERO: return "integer divide by zero";
    case EXCEPTION_STACK_OVERFLOW: return "stack overflow";
    case EXCEPTION_BREAKPOINT: return "breakpoint (an assertion?)";
    case 0xc0000409: return "fail-fast (__fastfail or a stack buffer overrun)";
    case 0xc0000420: return "assertion failure";
    case 0xc0000374: return "heap corruption";
    case 0xe06d7363: return "C++ exception";
    default:
        /* Any other error-severity code: a title that writes a minidump and exits on its own says
         * nothing else about why (the target title, after its first placeholder reservation). */
        return (code & 0xf0000000) == 0xc0000000 ? "exception" : NULL;
    }
}

/* "module+0xrva", or the bare address outside any module. */
static void describe(ULONG_PTR addr, char *out, size_t len)
{
    HMODULE module;
    char path[MAX_PATH], *name;

    /* Wine maps address 0 to the main module; nothing lives in the first 64 KiB. */
    if (addr >= 0x10000 && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (const char *)addr, &module)
        && GetModuleFileNameA(module, path, sizeof(path)))
    {
        name = strrchr(path, '\\');
        snprintf(out, len, "%s+%#llx", name ? name + 1 : path, (unsigned long long)(addr - (ULONG_PTR)module));
    }
    else
        snprintf(out, len, "%#llx", (unsigned long long)addr);
}

static BOOL readable(ULONG_PTR addr, SIZE_T len)
{
    MEMORY_BASIC_INFORMATION mbi;

    if (!VirtualQuery((void *)addr, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT
        || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) || !mbi.Protect)
        return FALSE;
    return addr + len <= (ULONG_PTR)mbi.BaseAddress + mbi.RegionSize;
}

static void log_stack_frames(CONTEXT ctx, ULONG_PTR low, ULONG_PTR high, int frames)
{
    char where[MAX_PATH + 32];

    for (int i = 0; i < frames && ctx.Rip; i++)
    {
        DWORD64 image_base;
        RUNTIME_FUNCTION *fn;
        HMODULE module;

        describe(ctx.Rip, where, sizeof(where));
        rebox_log("  #%-2d %s", i, where);
        if (ctx.Rsp < low || ctx.Rsp >= high || !readable(ctx.Rsp, 8))
            break;
        /* Only addresses in a module are unwound: for others Wine's RtlLookupFunctionEntry consults
         * dynamically registered tables, and a stale return address (a UE5 title's render thread gave
         * 0xc040) made it fault inside the hang dump, which killed the title. */
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (const WCHAR *)(ULONG_PTR)ctx.Rip, &module))
            break;
        fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, NULL);
        if (fn)
        {
            void *handler_data;
            DWORD64 frame;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data, &frame, NULL);
        }
        else
        {
            /* Leaf function: the return address is at the top of the stack. */
            ctx.Rip = *(DWORD64 *)ctx.Rsp;
            ctx.Rsp += 8;
        }
    }
}

static void log_stack(CONTEXT ctx)
{
    NT_TIB *tib = (NT_TIB *)NtCurrentTeb();
    log_stack_frames(ctx, (ULONG_PTR)tib->StackLimit, (ULONG_PTR)tib->StackBase, 24);
}

static LONG CALLBACK on_exception(EXCEPTION_POINTERS *ep)
{
    EXCEPTION_RECORD *rec = ep->ExceptionRecord;
    CONTEXT *c = ep->ContextRecord;
    const char *name = code_name(rec->ExceptionCode);
    char where[MAX_PATH + 32], bytes[3 * 16 + 1] = "";

    if (!name || TlsGetValue(busy_slot) || InterlockedDecrement(&remaining) < 0)
        return EXCEPTION_CONTINUE_SEARCH;
    TlsSetValue(busy_slot, (void *)1);

    describe(c->Rip, where, sizeof(where));
    rebox_log("xmem: first-chance %s (%#lx) at %s, thread %#lx", name, rec->ExceptionCode, where,
                 GetCurrentThreadId());
    if (rec->ExceptionCode != EXCEPTION_ACCESS_VIOLATION && rec->NumberParameters)
        rebox_log("  parameters: %#llx %#llx %#llx %#llx", (unsigned long long)rec->ExceptionInformation[0],
                  (unsigned long long)(rec->NumberParameters > 1 ? rec->ExceptionInformation[1] : 0),
                  (unsigned long long)(rec->NumberParameters > 2 ? rec->ExceptionInformation[2] : 0),
                  (unsigned long long)(rec->NumberParameters > 3 ? rec->ExceptionInformation[3] : 0));
    if (rec->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rec->NumberParameters >= 2)
    {
        ULONG_PTR target = rec->ExceptionInformation[1];
        const char *op = rec->ExceptionInformation[0] == 1 ? "writing" : rec->ExceptionInformation[0] == 8
                       ? "executing" : "reading";

        rebox_log("  %s %#llx%s", op, (unsigned long long)target,
                     target - window_base < window_size ? " (inside the GPU window: unmapped or freed GPU memory?)" : "");
    }
    if (!readable(c->Rip, 1) && readable(c->Rsp, 8))
    {
        /* A call through a null or stale pointer: where it was called from. */
        char from[MAX_PATH + 32];
        describe(*(ULONG_PTR *)c->Rsp, from, sizeof(from));
        rebox_log("  called from %s (the return address at rsp)", from);
    }
    if (readable(c->Rip, 16))
        for (int i = 0; i < 16; i++)
            snprintf(bytes + 3 * i, 4, "%02x ", ((BYTE *)c->Rip)[i]);
    rebox_log("  code: %s", bytes);
    rebox_log("  rax %016llx rbx %016llx rcx %016llx rdx %016llx", c->Rax, c->Rbx, c->Rcx, c->Rdx);
    rebox_log("  rsi %016llx rdi %016llx rbp %016llx rsp %016llx", c->Rsi, c->Rdi, c->Rbp, c->Rsp);
    rebox_log("  r8  %016llx r9  %016llx r10 %016llx r11 %016llx", c->R8, c->R9, c->R10, c->R11);
    rebox_log("  r12 %016llx r13 %016llx r14 %016llx r15 %016llx", c->R12, c->R13, c->R14, c->R15);
    if (rec->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
    {
        /* The title memory the registers point at (not the stack): what kind of memory the object
         * being read is. */
        const ULONG_PTR values[] = { c->Rax, c->Rbx, c->Rcx, c->Rdx, c->Rsi, c->Rdi, c->Rbp, c->R8, c->R9, c->R10,
                                     c->R11, c->R12, c->R13, c->R14, c->R15 };
        static const char *const names[] = { "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "r8", "r9", "r10",
                                             "r11", "r12", "r13", "r14", "r15" };
        for (UINT i = 0; i < ARRAYSIZE(values); i++)
            if (values[i] >= 0x100000000ull && values[i] < 0x400000000000ull
                    && values[i] - (c->Rsp & ~(ULONG_PTR)0xfffff) >= 0x200000)
                xmem_describe_address(names[i], values[i]);
    }
    if (rec->ExceptionCode != EXCEPTION_STACK_OVERFLOW)
        log_stack(*c);

    TlsSetValue(busy_slot, NULL);
    return EXCEPTION_CONTINUE_SEARCH;
}

/* -------------------------------------------------------------------- hangs */

static volatile LONG64 heartbeats;

void WINAPI ReboxHeartbeat(void)
{
    InterlockedIncrement64(&heartbeats);
}

/* Another thread's frames from a copy of its stack: rip, then each stack value that is a return
 * address, one just after a call instruction (E8 rel32, or FF /2 of 2, 3 or 6 bytes) in a loaded
 * module. A scan, not an unwind: a stale return address can show up, but nothing can fault. */
static void log_stack_scan(ULONG_PTR rip, const ULONG_PTR *stack, SIZE_T count, int frames)
{
    char where[MAX_PATH + 32];
    int n = 0;

    if (rip)
    {
        describe(rip, where, sizeof(where));
        rebox_log("  #%-2d %s", n++, where);
    }
    for (SIZE_T i = 0; i < count && n < frames; i++)
    {
        ULONG_PTR v = stack[i];
        HMODULE module;
        const BYTE *b = (const BYTE *)v;
        if (v < 0x10000 || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                                                   | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                               (const WCHAR *)v, &module)
                || IsBadReadPtr(b - 6, 6))
            continue;
        if (b[-5] != 0xe8 && !(b[-2] == 0xff && (b[-1] & 0x38) == 0x10) && !(b[-3] == 0xff && (b[-2] & 0x38) == 0x10)
                && !(b[-6] == 0xff && (b[-5] & 0x38) == 0x10))
            continue;
        describe(v, where, sizeof(where));
        rebox_log("  #%-2d %s", n++, where);
    }
}

/* The function table entry for rip, from its module's exception directory: read from the
 * module's mapped image only, without the loader or Wine's unwinder (neither may be entered while
 * a sampled thread could hold their locks, and the unwinder faulted on stale addresses). */
static const RUNTIME_FUNCTION *copy_function_entry(ULONG_PTR rip, ULONG_PTR *image_base)
{
    HMODULE module;
    const IMAGE_DOS_HEADER *dos;
    const IMAGE_NT_HEADERS64 *nt;
    const IMAGE_DATA_DIRECTORY *dir;
    const RUNTIME_FUNCTION *table;
    DWORD rva, lo = 0, hi;

    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (const WCHAR *)rip, &module))
        return NULL;
    dos = (const IMAGE_DOS_HEADER *)module;
    nt = (const IMAGE_NT_HEADERS64 *)((const BYTE *)module + dos->e_lfanew);
    dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
    if (!dir->VirtualAddress || dir->Size < sizeof(RUNTIME_FUNCTION))
        return NULL;
    *image_base = (ULONG_PTR)module;
    table = (const RUNTIME_FUNCTION *)((const BYTE *)module + dir->VirtualAddress);
    hi = dir->Size / sizeof(RUNTIME_FUNCTION);
    rva = (DWORD)(rip - (ULONG_PTR)module);
    while (lo < hi)
    {
        DWORD mid = (lo + hi) / 2;
        if (rva < table[mid].BeginAddress)
            hi = mid;
        else if (rva >= table[mid].EndAddress)
            lo = mid + 1;
        else
            return &table[mid];
    }
    return NULL;
}

/* One frame of an x64 unwind (the PE unwind codes) over a copy of the stack: regs are the 16
 * general registers in CONTEXT order (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8-r15), rsp the
 * live address, stack the copy of [base, base + bytes). FALSE when a read leaves the copy. */
static BOOL copy_unwind_frame(ULONG_PTR *rip, ULONG_PTR regs[16], ULONG_PTR base, const BYTE *stack, SIZE_T bytes)
{
#define COPY_READ(addr, out) \
    do { if ((addr) < base || (addr) + 8 > base + bytes) return FALSE; memcpy(&(out), stack + ((addr) - base), 8); } while (0)
    ULONG_PTR image_base;
    const RUNTIME_FUNCTION *fn = copy_function_entry(*rip, &image_base);
    DWORD offset;

    if (!fn)
    {
        /* A leaf: the return address is at the top of the stack. */
        COPY_READ(regs[4], *rip);
        regs[4] += 8;
        return TRUE;
    }
    offset = (DWORD)(*rip - image_base - fn->BeginAddress);
    for (int depth = 0; depth < 32; depth++)
    {
        const BYTE *info = (const BYTE *)image_base + fn->UnwindData;
        BYTE prolog = info[1], count = info[2], frame_reg = info[3] & 15, frame_off = info[3] >> 4;
        const USHORT *codes = (const USHORT *)(info + 4);
        ULONG_PTR frame = regs[4];

        /* Saves are relative to the frame: the frame register less its offset once the prolog set
         * it up, else rsp in the body (Wine's RtlVirtualUnwind does the same). */
        for (unsigned i = 0; i < count; i++)
        {
            BYTE at = codes[i] & 0xff, op = (codes[i] >> 8) & 15;
            if (op == 3 /* UWOP_SET_FPREG */ && frame_reg && (depth || offset >= at || offset >= prolog))
                frame = regs[frame_reg] - frame_off * 16;
            if (op == 1) i += ((codes[i] >> 12) ? 2 : 1);
            else if (op == 4 || op == 8) i += 1;
            else if (op == 5 || op == 9) i += 2;
        }
        for (unsigned i = 0; i < count; i++)
        {
            BYTE at = codes[i] & 0xff, op = (codes[i] >> 8) & 15, arg = codes[i] >> 12;
            BOOL done = depth || offset >= at || offset >= prolog;
            switch (op)
            {
            case 0: /* UWOP_PUSH_NONVOL */
                if (done) { COPY_READ(regs[4], regs[arg]); regs[4] += 8; }
                break;
            case 1: /* UWOP_ALLOC_LARGE */
            {
                ULONG size = arg ? *(const ULONG *)&codes[i + 1] : codes[i + 1] * 8u;
                if (done) regs[4] += size;
                i += arg ? 2 : 1;
                break;
            }
            case 2: /* UWOP_ALLOC_SMALL */
                if (done) regs[4] += arg * 8u + 8;
                break;
            case 3: /* UWOP_SET_FPREG */
                if (done) regs[4] = frame;
                break;
            case 4: /* UWOP_SAVE_NONVOL */
                if (done) COPY_READ(frame + codes[i + 1] * 8u, regs[arg]);
                i += 1;
                break;
            case 5: /* UWOP_SAVE_NONVOL_FAR */
                if (done) COPY_READ(frame + *(const ULONG *)&codes[i + 1], regs[arg]);
                i += 2;
                break;
            case 8: /* UWOP_SAVE_XMM128 */
                i += 1;
                break;
            case 9: /* UWOP_SAVE_XMM128_FAR */
                i += 2;
                break;
            case 10: /* UWOP_PUSH_MACHFRAME: rip and rsp from the frame */
            {
                ULONG_PTR at_rsp = regs[4] + (arg ? 8 : 0);
                COPY_READ(at_rsp, *rip);
                COPY_READ(at_rsp + 24, regs[4]);
                return TRUE;
            }
            default:
                return FALSE;
            }
        }
        if (!(info[0] >> 3 & 4 /* UNW_FLAG_CHAININFO */))
            break;
        fn = (const RUNTIME_FUNCTION *)(info + 4 + ((count + 1) & ~1) * 2);
        offset = ~0u;
    }
    COPY_READ(regs[4], *rip);
    regs[4] += 8;
    return TRUE;
#undef COPY_READ
}

/* Another thread's frames from a copy of its stack, unwound by the modules' unwind codes; where
 * the unwind leaves the copy or a module, the rest is scanned. */
static void log_stack_copy(const CONTEXT *ctx, const ULONG_PTR *stack, SIZE_T count, int frames)
{
    char where[MAX_PATH + 32];
    ULONG_PTR regs[16], rip = ctx->Rip, base = ctx->Rsp;
    int n = 0;

    memcpy(regs, &ctx->Rax, sizeof(regs));
    while (n < frames && rip)
    {
        describe(rip, where, sizeof(where));
        rebox_log("  #%-2d %s", n++, where);
        if (!copy_unwind_frame(&rip, regs, base, (const BYTE *)stack, count * 8) || regs[4] <= base
                || regs[4] >= base + count * 8)
        {
            if (regs[4] > base && regs[4] < base + count * 8 && n < frames)
            {
                rebox_log("  (unwind stopped; scanning the rest)");
                log_stack_scan(0, stack + (regs[4] - base) / 8, count - (regs[4] - base) / 8, frames - n);
            }
            return;
        }
    }
}

static void log_thread(DWORD tid)
{
    typedef HRESULT (WINAPI *get_description_fn)(HANDLE, WCHAR **);
    static get_description_fn get_description;
    HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION,
                               FALSE, tid);
    CONTEXT ctx = { .ContextFlags = CONTEXT_FULL };
    WCHAR *name = NULL;
    BOOL ok;
    /* The top 32 KiB of its stack, copied while it is suspended, between bounds read from its TEB
     * before: while it is stopped, only GetThreadContext and a copy (it may hold a lock: the
     * heap's, the loader's). */
    ULONG_PTR *stack = HeapAlloc(GetProcessHeap(), 0, 4096 * sizeof(ULONG_PTR));
    ULONG_PTR stack_base = 0, stack_limit = 0;
    SIZE_T copied = 0;

    if (!thread)
    {
        HeapFree(GetProcessHeap(), 0, stack);
        return;
    }
    if (!get_description)
        get_description = (get_description_fn)GetProcAddress(GetModuleHandleA("kernelbase.dll"), "GetThreadDescription");
    {
        struct { LONG status; void *teb; ULONG_PTR pid, tid, affinity; LONG priority, base_priority; } info;
        NTSTATUS (WINAPI *query)(HANDLE, int, void *, ULONG, ULONG *) =
                (void *)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryInformationThread");
        HANDLE q = OpenThread(THREAD_QUERY_INFORMATION, FALSE, tid);
        if (q && query && !query(q, 0 /* ThreadBasicInformation */, &info, sizeof(info), NULL) && info.teb
                && !IsBadReadPtr(info.teb, sizeof(NT_TIB)))
        {
            stack_base = (ULONG_PTR)((NT_TIB *)info.teb)->StackBase;
            stack_limit = (ULONG_PTR)((NT_TIB *)info.teb)->StackLimit;
        }
        if (q)
            CloseHandle(q);
    }

    ok = SuspendThread(thread) != (DWORD)-1;
    if (ok)
    {
        ok = GetThreadContext(thread, &ctx);
        /* Copied while the thread can't change it, with nothing that takes a lock (the thread may
         * hold one); the walk below reads only the copy. Walking the live stack with Wine's
         * unwinder after resuming it faulted inside ntdll and ended the title (a stack sample). */
        if (ok && stack && ctx.Rsp >= stack_limit && ctx.Rsp < stack_base)
        {
            copied = min(stack_base - ctx.Rsp, 4096 * sizeof(ULONG_PTR)) & ~(SIZE_T)7;
            memcpy(stack, (void *)ctx.Rsp, copied);
        }
        ResumeThread(thread);
    }
    if (get_description && FAILED(get_description(thread, &name)))
        name = NULL;
    rebox_log("xmem: thread %#lx%s%ls%s", tid, name && *name ? " \"" : "", name && *name ? name : L"",
                 name && *name ? "\"" : "");
    if (name)
        LocalFree(name);
    if (ok)
        log_stack_copy(&ctx, stack, copied / 8, 24);
    HeapFree(GetProcessHeap(), 0, stack);
    CloseHandle(thread);
}

static void dump_threads(DWORD seconds, int sample)
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te = { sizeof(te) };
    DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();

    if (sample)
        rebox_log("xmem: stack sample %d (%lld presents so far): every thread's stack follows (REBOX_STACK_SAMPLE)",
                     sample, (long long)heartbeats);
    else
        rebox_log("xmem: no frame presented for %lu s: every thread's stack follows (REBOX_HANG_DUMP)", seconds);
    if (snap == INVALID_HANDLE_VALUE)
        return;
    for (BOOL more = Thread32First(snap, &te); more; more = Thread32Next(snap, &te))
        if (te.th32OwnerProcessID == pid && te.th32ThreadID != self)
            log_thread(te.th32ThreadID);
    CloseHandle(snap);
    {
        /* d3d12_xs's fences and slot counts (proxy.c), when it is loaded. */
        HMODULE d3d12 = GetModuleHandleA("d3d12_xs.dll");
        void (WINAPI *report)(void) = d3d12 ? (void *)GetProcAddress(d3d12, "ReboxHangReport") : NULL;
        if (report)
            report();
    }
    rebox_log("xmem: end of thread stacks");
}

static DWORD WINAPI hang_watch(void *param)
{
    DWORD seconds = (DWORD)(ULONG_PTR)param, idle = 0;
    LONG64 last = -1;
    int dumps = 0;
    BOOL armed = TRUE;

    while (dumps < 3)
    {
        LONG64 now;
        Sleep(1000);
        if ((now = heartbeats) != last)
        {
            last = now;
            idle = 0;
            armed = TRUE;
        }
        else if (++idle >= seconds && armed)
        {
            dump_threads(seconds, 0);
            dumps++;
            armed = FALSE;
        }
    }
    return 0;
}

static DWORD WINAPI stack_sampler(void *param)
{
    DWORD every = LOWORD((ULONG_PTR)param), count = HIWORD((ULONG_PTR)param);
    for (DWORD i = 1; i <= count; i++)
    {
        Sleep(every * 1000);
        dump_threads(0, (int)i);
    }
    return 0;
}

/* REBOX_SKIP_ASSERTS=1 (bring-up): a title's assertions are int3s in its own code ("master" builds
 * keep them, and a debugger would step past). Each is stepped over and logged with its caller, so
 * one run lists every assertion a title meets instead of stopping at the first; what follows an
 * int3 is mostly the title handling the failure. Probes (probe.c) are handled before this. */
static LONG CALLBACK skip_assert(EXCEPTION_POINTERS *ep)
{
    static LONG skipped;
    EXCEPTION_RECORD *rec = ep->ExceptionRecord;
    CONTEXT *c = ep->ContextRecord;
    HMODULE exe = GetModuleHandleA(NULL);
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)((BYTE *)exe + ((IMAGE_DOS_HEADER *)exe)->e_lfanew);
    char where[MAX_PATH + 32], caller[MAX_PATH + 32];
    LONG n;

    if (rec->ExceptionCode != EXCEPTION_BREAKPOINT || c->Rip - (ULONG_PTR)exe >= nt->OptionalHeader.SizeOfImage
            || *(const BYTE *)c->Rip != 0xcc)
        return EXCEPTION_CONTINUE_SEARCH;
    n = InterlockedIncrement(&skipped);
    if (n <= 200)
    {
        describe(c->Rip, where, sizeof(where));
        describe(readable(c->Rsp, 8) ? *(ULONG_PTR *)c->Rsp : 0, caller, sizeof(caller));
        rebox_log("xmem: assertion %ld skipped at %s (thread %#lx; rax %#llx, [rsp] %s)", n, where,
                  GetCurrentThreadId(), (unsigned long long)c->Rax, caller);
    }
    c->Rip++;
    return EXCEPTION_CONTINUE_EXECUTION;
}

/* REBOX_SKIP_FAULT=RVA:LENGTH[,...] (bring-up, hex): an access violation at that RVA of the exe
 * skips the faulting instruction (LENGTH bytes) and goes on, logged. For a title's deliberate
 * crash, such as a watchdog writing its signature to address 0, to see what it would do next. */
static struct { ULONG_PTR rva, length; } skip_faults[8];
static UINT skip_fault_count;

static LONG CALLBACK skip_fault(EXCEPTION_POINTERS *ep)
{
    static LONG skipped;
    EXCEPTION_RECORD *rec = ep->ExceptionRecord;
    CONTEXT *c = ep->ContextRecord;
    ULONG_PTR exe = (ULONG_PTR)GetModuleHandleA(NULL);

    if (rec->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;
    for (UINT i = 0; i < skip_fault_count; i++)
        if (c->Rip == exe + skip_faults[i].rva)
        {
            if (InterlockedIncrement(&skipped) <= 32)
                rebox_log("xmem: the fault at exe+%#llx skipped (thread %#lx; REBOX_SKIP_FAULT)",
                          (unsigned long long)skip_faults[i].rva, GetCurrentThreadId());
            c->Rip += skip_faults[i].length;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    return EXCEPTION_CONTINUE_SEARCH;
}

/* Hex digits to bytes: the count, or -1 if odd or not hex. */
static int hex_bytes(const char *s, size_t n, BYTE *out, size_t max)
{
    size_t count = 0;
    if (n % 2)
        return -1;
    for (size_t i = 0; i < n; i += 2)
    {
        unsigned v;
        char pair[3] = { s[i], s[i + 1], 0 };
        if (count == max || !isxdigit((unsigned char)pair[0]) || !isxdigit((unsigned char)pair[1])
                || sscanf(pair, "%x", &v) != 1)
            return -1;
        out[count++] = (BYTE)v;
    }
    return (int)count;
}

/* REBOX_PATCH=RVA:EXPECTED:REPLACEMENT[,...] (bring-up, hex): the exe's bytes at RVA are replaced,
 * only if they are EXPECTED (another build is left alone), logged either way. For a title's own
 * checks that end it (a watchdog's timeout branch, say), to let it run on. */
static void apply_patches(void)
{
    char list[512];
    DWORD len = GetEnvironmentVariableA("REBOX_PATCH", list, sizeof(list));
    BYTE *exe = (BYTE *)GetModuleHandleA(NULL);
    char *at = list;

    if (!len || len >= sizeof(list))
        return;
    while (at && *at)
    {
        char *next = strchr(at, ','), *c1, *c2;
        unsigned long long rva;
        BYTE expected[64], replacement[64];
        int ne, nr;
        if (next)
            *next++ = 0;
        c1 = strchr(at, ':');
        c2 = c1 ? strchr(c1 + 1, ':') : NULL;
        if (!c1 || !c2 || sscanf(at, "%llx", &rva) != 1
                || (ne = hex_bytes(c1 + 1, (size_t)(c2 - c1 - 1), expected, sizeof(expected))) <= 0
                || (nr = hex_bytes(c2 + 1, strlen(c2 + 1), replacement, sizeof(replacement))) != ne)
            rebox_log("xmem: REBOX_PATCH entry \"%s\" not understood (RVA:EXPECTED:REPLACEMENT, same lengths)", at);
        else if (memcmp(exe + rva, expected, ne))
            rebox_log("xmem: REBOX_PATCH at exe+%#llx: the bytes there aren't the expected ones; left alone", rva);
        else
        {
            DWORD old;
            if (VirtualProtect(exe + rva, ne, PAGE_EXECUTE_READWRITE, &old))
            {
                memcpy(exe + rva, replacement, ne);
                VirtualProtect(exe + rva, ne, old, &old);
                FlushInstructionCache(GetCurrentProcess(), exe + rva, ne);
                rebox_log("xmem: REBOX_PATCH: %d bytes at exe+%#llx replaced", ne, rva);
            }
            else
                rebox_log("xmem: REBOX_PATCH at exe+%#llx: can't make it writable (error %lu)", rva, GetLastError());
        }
        at = next;
    }
}

void crash_report_install(ULONG_PTR gpu_window_base, ULONG_PTR gpu_window_size)
{
    apply_patches();
    {
        char list[256];
        DWORD len = GetEnvironmentVariableA("REBOX_SKIP_FAULT", list, sizeof(list));
        const char *at = list;
        unsigned long long rva, length;
        int used;
        while (len && len < sizeof(list) && skip_fault_count < ARRAYSIZE(skip_faults)
                && sscanf(at, "%llx:%llx%n", &rva, &length, &used) == 2 && length)
        {
            skip_faults[skip_fault_count].rva = (ULONG_PTR)rva;
            skip_faults[skip_fault_count++].length = (ULONG_PTR)length;
            rebox_log("xmem: a fault at exe+%#llx is skipped (%llu bytes; REBOX_SKIP_FAULT)", rva, length);
            at += used;
            if (*at != ',')
                break;
            at++;
        }
        if (skip_fault_count)
            AddVectoredExceptionHandler(1, skip_fault);
    }
    char hang[16];
    DWORD hang_len = GetEnvironmentVariableA("REBOX_HANG_DUMP", hang, sizeof(hang));
    int hang_seconds = hang_len && hang_len < sizeof(hang) ? atoi(hang) : 60;
    char buf[16];
    DWORD n = GetEnvironmentVariableA("REBOX_CRASH_LOG", buf, sizeof(buf));

    remaining = n && n < sizeof(buf) ? atoi(buf) : 8;
    window_base = gpu_window_base;
    window_size = gpu_window_size;
    if (remaining > 0 && (busy_slot = TlsAlloc()) != TLS_OUT_OF_INDEXES)
        AddVectoredExceptionHandler(0, on_exception);
    {
        char skip[4];
        if (GetEnvironmentVariableA("REBOX_SKIP_ASSERTS", skip, sizeof(skip)) && skip[0] == '1')
        {
            AddVectoredExceptionHandler(0, skip_assert);
            rebox_log("xmem: the title's assertions (int3) are stepped over and logged (REBOX_SKIP_ASSERTS)");
        }
    }
    if (hang_seconds > 0)
        CloseHandle(CreateThread(NULL, 0, hang_watch, (void *)(ULONG_PTR)hang_seconds, 0, NULL));
    {
        char sample[32];
        DWORD len = GetEnvironmentVariableA("REBOX_STACK_SAMPLE", sample, sizeof(sample));
        unsigned every = 0, count = 10;
        if (len && len < sizeof(sample) && sscanf(sample, "%u,%u", &every, &count) >= 1 && every && count)
            CloseHandle(CreateThread(NULL, 0, stack_sampler,
                                     (void *)(ULONG_PTR)MAKELONG(min(every, 0xffffu), min(count, 0xffffu)), 0, NULL));
    }
}
