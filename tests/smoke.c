/* Smoke test for the shims, run under Wine by tests/run-smoke.sh.
 *
 *   smoke.exe FAKE_CONSOLE_SHADER [nocache]
 *
 * Exercises xmem.dll directly, then creates a device through d3d12_xs.dll and
 * builds compute PSOs from FAKE_CONSOLE_SHADER (bytes vkd3d-proton cannot
 * compile). Those only succeed if the shader cache substitutes a translation. */

#define COBJMACROS
#include <initguid.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <xaudio2.h>
#include <mmdeviceapi.h>
#include <psapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../generated/test_shaders.h"
#include "../anyps5/rebox_anyps5.h"
#include "../shims/d3d12_xs/console_compile.h"

/* The title's allocation hooks, exported as a title does (__XMemAllocationHooks__: alloc, free):
 * xmem uses them for every XMemAlloc. Everything goes to the defaults except a descriptor heap
 * (object type 4, the D3D allocator 128) of a registered size, which gets placement_memory, as the
 * target title places its heap. */
static BYTE placement_memory[64 * 64] __attribute__((aligned(64)));
static SIZE_T placement_size, placement_asked;
static UINT xhook_calls, xhook_heap_calls;
static ULONGLONG hook_last_attributes;
static void *WINAPI hook_alloc(SIZE_T size, ULONGLONG attributes)
{
    xhook_calls++;
    hook_last_attributes = attributes;
    if ((attributes & 0x3fff) == 4)
        xhook_heap_calls++;
    if ((attributes & 0x3fff) == 4 && (attributes >> 16 & 0xff) == 128 && placement_size && size <= sizeof(placement_memory))
    {
        placement_asked = size;
        placement_size = 0;
        return placement_memory;
    }
    return ((void *(WINAPI *)(SIZE_T, ULONGLONG))GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemAllocDefault"))(
            size, attributes);
}
static void WINAPI hook_free(void *address, ULONGLONG attributes)
{
    if (address != placement_memory)
        ((void (WINAPI *)(void *, ULONGLONG))GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemFreeDefault"))(
                address, attributes);
}
__declspec(dllexport) void *__XMemAllocationHooks__[4] = { (void *)hook_alloc, (void *)hook_free, NULL, NULL };

static int failures;

/* Heap slot 10 reports shadow-heap title addresses unless REBOX_HEAP_VA says real or virtual;
 * the shadow-heap and deferred-shader tests need them. */
static BOOL title_heap_va(void)
{
    const char *mode = getenv("REBOX_HEAP_VA"), *legacy = getenv("REBOX_REAL_HEAP_VA");
    if (mode)
        return !strcmp(mode, "title");
    return !(legacy && legacy[0] == '0');
}

#define CHECK(cond, ...) do { \
    if (cond) printf("ok   " __VA_ARGS__); \
    else { printf("FAIL " __VA_ARGS__); failures++; } \
    printf("\n"); } while (0)

typedef void *(WINAPI *alloc_fn)(SIZE_T, ULONGLONG);
typedef void (WINAPI *free_fn)(void *, ULONGLONG);
typedef void (WINAPI *hooks_fn)(alloc_fn, free_fn);
/* Layout as XboxBvhBuilder.dll uses it: MEMORY_BASIC_INFORMATION, then GPU access bits. */
typedef struct { MEMORY_BASIC_INFORMATION mbi; ULONG gpu_access, reserved[3]; } xmem_info;
typedef BOOL (WINAPI *vquery_fn)(HANDLE, const void *, xmem_info *);
typedef BOOL (WINAPI *wss_fn)(ULONG, void *);

static alloc_fn alloc_default;
static free_fn free_default;
static int hook_calls;

static void *WINAPI my_alloc(SIZE_T size, ULONGLONG attr) { hook_calls++; return alloc_default(size, attr); }
static void WINAPI my_free(void *p, ULONGLONG attr) { hook_calls++; free_default(p, attr); }

/* Faults xmem's reporter should describe (crash.c): a read of 0x10, then one
 * inside the GPU window. This handler runs after xmem's and skips the
 * 2-byte "mov eax, [rax]" so the test continues; run-smoke.sh checks the
 * report in the log. */
static LONG CALLBACK skip_test_fault(EXCEPTION_POINTERS *ep)
{
    BYTE *rip = (BYTE *)ep->ContextRecord->Rip;

    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rip[0] == 0x8b && rip[1] == 0x00
        && (ep->ContextRecord->Rax == 0x10 || ep->ContextRecord->Rax == 0x20ff00000040ull))
    {
        ep->ContextRecord->Rip += 2;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static void __attribute__((noinline)) fault_read(ULONG_PTR addr)
{
    __asm__ volatile("movl (%%rax), %%eax" : "+a"(addr) :: "memory");
}

static void test_crash_report(HMODULE xmem)
{
    typedef BOOL (WINAPI *vquery_fn2)(HANDLE, const void *, void *);
    vquery_fn2 vquery = (vquery_fn2)(void *)GetProcAddress(xmem, "XMemVirtualQuery");
    void *veh = AddVectoredExceptionHandler(0, skip_test_fault);
    BOOL ok;

    fault_read(0x10);
    fault_read(0x20ff00000040ull);
    RemoveVectoredExceptionHandler(veh);
    printf("ok   resumed after two reported faults\n");

    SetLastError(0);
    ok = vquery(GetCurrentProcess(), &ok, NULL);
    CHECK(!ok && GetLastError() == ERROR_NOACCESS, "XMemVirtualQuery with a null buffer fails with ERROR_NOACCESS (%lu)",
          GetLastError());
}

static void test_probe(HMODULE xmem);

/* Closing a title's window suspends it (its app state routines hear "quiesced") before the
 * process ends; REBOX_QUIT_EXIT=0 stops short of ending it here. */
static HANDLE quit_suspended;
static BOOLEAN quit_quiesced;
static void CALLBACK quit_app_state(BOOLEAN quiesced, void *context)
{
    quit_quiesced = quiesced;
    SetEvent(context);
}

static void test_quit(HMODULE xmem)
{
    ULONG (WINAPI *reg)(void *, void *, void **) = (void *)GetProcAddress(xmem, "ReboxAppStateRegister");
    HWND (WINAPI *subclass)(HWND) = (void *)GetProcAddress(xmem, "ReboxQuitSubclass");
    void *registration = NULL;
    HWND hwnd;

    CHECK(reg && subclass, "xmem exports ReboxAppStateRegister and ReboxQuitSubclass");
    if (!reg || !subclass) return;
    SetEnvironmentVariableA("REBOX_QUIT_EXIT", "0");
    quit_suspended = CreateEventA(NULL, TRUE, FALSE, NULL);
    CHECK(reg(quit_app_state, quit_suspended, &registration) == ERROR_SUCCESS && registration,
          "RegisterAppStateChangeNotification (ours) registers");
    {
        WNDCLASSW wc = { .lpfnWndProc = DefWindowProcW, .hInstance = GetModuleHandleW(NULL), .lpszClassName = L"SmokeTitle" };
        RegisterClassW(&wc);
    }
    /* Message-only: there's no display here. */
    hwnd = subclass(CreateWindowExW(0, L"SmokeTitle", L"title", 0, 0, 0, 64, 64, HWND_MESSAGE, NULL,
                                    GetModuleHandleW(NULL), NULL));
    CHECK(hwnd != NULL, "a title window");
    if (!hwnd) return;
    CHECK(WaitForSingleObject(quit_suspended, 0) == WAIT_TIMEOUT, "the title isn't suspended before it's closed");
    SendMessageW(hwnd, WM_CLOSE, 0, 0);
    CHECK(WaitForSingleObject(quit_suspended, 5000) == WAIT_OBJECT_0 && quit_quiesced,
          "closing the title's window suspends the title (quiesced)");
    CHECK(IsWindow(hwnd), "the window stays until the process ends");
    DestroyWindow(hwnd);
    SetEnvironmentVariableA("REBOX_QUIT_EXIT", NULL);
}

static void test_audio_state_monitor(void);

static void test_xmem(void)
{
    HMODULE m = LoadLibraryA("xmem.dll");
    alloc_fn xalloc;
    free_fn xfree;
    hooks_fn set_hooks;
    BYTE *a, *b;

    CHECK(m != NULL, "load xmem.dll");
    if (!m) return;
    xalloc = (alloc_fn)(void *)GetProcAddress(m, "XMemAlloc");
    xfree = (free_fn)(void *)GetProcAddress(m, "XMemFree");
    set_hooks = (hooks_fn)(void *)GetProcAddress(m, "XMemSetAllocationHooks");
    alloc_default = (alloc_fn)(void *)GetProcAddress(m, "XMemAllocDefault");
    free_default = (free_fn)(void *)GetProcAddress(m, "XMemFreeDefault");
    CHECK(xalloc && xfree && set_hooks && alloc_default && free_default, "xmem exports present");
    CHECK(GetProcAddress(m, "ReboxHeartbeat") != NULL, "xmem exports ReboxHeartbeat (the hang watchdog's)");
    test_probe(m);
    test_quit(m);
    test_audio_state_monitor();
    if (!(xalloc && xfree && set_hooks && alloc_default && free_default)) return;
    test_crash_report(m);

    /* alignment field (bits 24-28) = 7 -> 2^7 = 128-byte alignment, heap path */
    a = xalloc(100, 7ull << 24);
    CHECK(a && !((ULONG_PTR)a & 127), "XMemAlloc aligned heap allocation %p", a);
    if (a) { a[0] = a[99] = 1; xfree(a, 7ull << 24); }

    /* Blocks come back zeroed even when the allocator reuses a dirty address
     * (Xbox2Windows' era_heap_reuse): heap path and GPU-visible path. */
    {
        static const ULONGLONG kinds[] = { 5ull << 24, 0xa8004000 };
        for (int k = 0; k < 2; k++)
        {
            BYTE *held[32] = { 0 }, *dirty = xalloc(4096, kinds[k]);
            BOOL zero = dirty != NULL, reused = FALSE;
            int n;

            if (dirty) { memset(dirty, 0xa5, 4096); xfree(dirty, kinds[k]); }
            for (n = 0; n < 32 && zero && !reused; n++)
            {
                if (!(held[n] = xalloc(4096, kinds[k]))) { zero = FALSE; break; }
                for (int i = 0; i < 4096; i++)
                    zero &= held[n][i] == 0;
                reused = held[n] == dirty;
            }
            CHECK(zero, "XMemAlloc %#llx blocks are zeroed (dirty address reused: %s)",
                  kinds[k], reused ? "yes" : "no");
            while (n-- > 0)
                if (held[n]) xfree(held[n], kinds[k]);
        }
    }

    /* memory_type field (bits 29-32) != 0 -> VirtualAlloc path */
    b = xalloc(1 << 20, 1ull << 29);
    CHECK(b && !((ULONG_PTR)b & 0xffff), "XMemAlloc page allocation %p", b);
    if (b) { b[0] = b[(1 << 20) - 1] = 1; xfree(b, 1ull << 29); }

    /* Attribute words real callers pass (Xbox2Windows call-site notes; not checked against a
     * disassembly here). Bink's fallback allocator asks for len + 64 with 0x05e50000, steps
     * the base forward 33-64 bytes to a 32-byte boundary, stores the step at [-1] and a tag
     * at [-2], and later frees base = user - step with the same word. Here: alignment 5
     * (32 bytes), memory_type 0, so the heap path. */
    {
        const ULONGLONG bink = 0x05e50000;
        BYTE *base = xalloc(1000 + 64, bink), *user;

        CHECK(base && !((ULONG_PTR)base & 31), "XMemAlloc Bink attributes %#llx -> %p", bink, base);
        if (base)
        {
            user = base + 64 - ((ULONG_PTR)base & 31);
            user[-1] = (BYTE)(user - base);
            user[-2] = 0xb1;
            memset(user, 0x5a, 1000);
            CHECK(!((ULONG_PTR)user & 31) && user + 1000 <= base + 1064 && user - user[-1] == base,
                  "Bink-style 32-byte user pointer fits and recovers its base");
            xfree(user - user[-1], bink);
        }
    }

    /* The AMD upscaler rounds sizes up to 256 and passes 0xa8004000 (its on-disk default),
     * then copies shader bytes in and hands the pointer to a compute PSO, so the GPU reads
     * it. Here: alignment 8 (256 bytes, matching its rounding), page_size 1 (64 KiB pages, so a
     * 64 KiB boundary, as Xbox2Windows' xmem.dll gives), memory_type 5, so GPU-visible memory,
     * which XMemVirtualQuery must report as GPU-accessible. */
    {
        const ULONGLONG amd = 0xa8004000;
        vquery_fn vquery = (vquery_fn)(void *)GetProcAddress(m, "XMemVirtualQuery");
        BYTE *p = xalloc(0x1100, amd);
        xmem_info info;

        CHECK(p && !((ULONG_PTR)p & 0xffff), "XMemAlloc AMD attributes %#llx -> %p", amd, p);
        if (p)
        {
            memset(p, 0xa5, 0x1100);
            memset(&info, 0xcc, sizeof(info));
            CHECK(vquery && vquery(GetCurrentProcess(), p + 0x10ff, &info) && info.mbi.State == MEM_COMMIT
                  && (info.gpu_access & 3) == 3,
                  "AMD allocation is committed and GPU-accessible (state %#lx gpu_access %#lx)",
                  info.mbi.State, info.gpu_access);
            xfree(p, amd);
        }
    }

    /* 4 MiB pages (page_size 2) start on a 4 MiB boundary, GPU-visible
     * (memory_type 1) and CPU heap pages alike. */
    {
        static const ULONGLONG words[] = { (1ull << 29) | (2ull << 14), 2ull << 14 };
        for (int k = 0; k < 2; k++)
        {
            BYTE *p = xalloc(100 << 10, words[k]);
            CHECK(p && !((ULONG_PTR)p & ((4 << 20) - 1)), "XMemAlloc 4 MiB pages %#llx -> %p", words[k], p);
            if (p) { p[0] = p[(100 << 10) - 1] = 1; xfree(p, words[k]); }
        }
    }

    set_hooks(my_alloc, my_free);
    xfree(xalloc(16, 0), 0);
    set_hooks(NULL, NULL);
    CHECK(hook_calls == 2, "allocation hooks called (%d)", hook_calls);

    /* Called as XboxBvhBuilder.dll does: (process, address, info), 0x40-byte buffer. */
    {
        vquery_fn vquery = (vquery_fn)(void *)GetProcAddress(m, "XMemVirtualQuery");
        BYTE *p = xalloc(1 << 16, 1ull << 29);
        xmem_info info;

        CHECK(vquery && p, "XMemVirtualQuery present");
        if (vquery && p)
        {
            memset(&info, 0xcc, sizeof(info));
            CHECK(vquery(GetCurrentProcess(), p + 100, &info) && info.mbi.State == MEM_COMMIT
                  && (info.mbi.Protect & PAGE_READWRITE) && (info.gpu_access & 3) == 3,
                  "XMemVirtualQuery state %#lx protect %#lx gpu_access %#lx",
                  info.mbi.State, info.mbi.Protect, info.gpu_access);
            xfree(p, 1ull << 29);
        }
    }

    /* Called as the AMD upscaler does: (0x20000001, &local), reading the u64 at +8. */
    {
        wss_fn wss = (wss_fn)(void *)GetProcAddress(m, "XMemGetWorkingSetStatistics");
        ULONGLONG out[4], before, after;
        BYTE *p;

        CHECK(wss != NULL, "XMemGetWorkingSetStatistics present");
        if (wss)
        {
            memset(out, 0xcc, sizeof(out));
            CHECK(wss(0x20000001, out) && out[1] != 0xccccccccccccccccull
                  && out[0] == 0xccccccccccccccccull && out[2] == 0xccccccccccccccccull,
                  "XMemGetWorkingSetStatistics writes only +8 (%#llx)", out[1]);
            before = out[1];
            p = xalloc(16 << 20, 1ull << 29);
            if (p) memset(p, 1, 16 << 20); /* Wine counts only touched pages */
            wss(0x20000001, out);
            after = out[1];
            CHECK(p && after >= before + (16 << 20),
                  "XMemGetWorkingSetStatistics tracks allocations (+%lld bytes)", (long long)(after - before));
            if (p) xfree(p, 1ull << 29);
            memset(out, 0xcc, sizeof(out));
            CHECK(!wss(0x1234, out) && out[1] == 0xccccccccccccccccull,
                  "XMemGetWorkingSetStatistics rejects unknown selector");
        }
    }

    /* XMemVirtualAlloc (flags before the protection) and physical pages: allocate tokens, map
     * them into a reserved range, write through it, unmap. */
    {
        typedef void *(WINAPI *valloc_fn)(void *, SIZE_T, DWORD, ULONGLONG, DWORD);
        typedef BOOL (WINAPI *palloc_fn)(ULONGLONG, ULONG_PTR *, ULONG_PTR *);
        typedef BOOL (WINAPI *pmap_fn)(void *, ULONG_PTR, ULONG_PTR *);
        typedef BOOL (WINAPI *pfree_fn)(ULONG_PTR, ULONG_PTR *);
        valloc_fn valloc = (valloc_fn)(void *)GetProcAddress(m, "XMemVirtualAlloc");
        palloc_fn palloc = (palloc_fn)(void *)GetProcAddress(m, "XMemAllocatePhysicalPages");
        pmap_fn pmap = (pmap_fn)(void *)GetProcAddress(m, "XMemMapPhysicalPages");
        pfree_fn pfree = (pfree_fn)(void *)GetProcAddress(m, "XMemFreePhysicalPages");
        ULONG_PTR pages[16], count = 16;
        BYTE *range, *v;

        CHECK(valloc && palloc && pmap && pfree, "xmem virtual and physical page exports present");
        if (valloc && palloc && pmap && pfree)
        {
            v = valloc(NULL, 1 << 20, MEM_RESERVE | MEM_COMMIT | 0x20000000 /* MEM_LARGE_PAGES */,
                       0x1234567800000000ull, PAGE_READWRITE);
            CHECK(v != NULL, "XMemVirtualAlloc with large pages and XMem flags -> %p", v);
            /* The console's addresses are 46-bit: big reservations, the title's own VirtualAlloc
             * included (an import xmem replaces), stay below 2^46. */
            {
                void *big = valloc(NULL, 16ull << 30, MEM_RESERVE, 0, PAGE_NOACCESS);
                void *own = VirtualAlloc(NULL, 16ull << 30, MEM_RESERVE, PAGE_NOACCESS);
                CHECK(big && (ULONG_PTR)big < (1ull << 46) && own && (ULONG_PTR)own < (1ull << 46),
                      "16 GiB reservations through XMemVirtualAlloc and the title's VirtualAlloc are below 2^46 "
                      "(%p, %p)", big, own);
                if (big) VirtualFree(big, 0, MEM_RELEASE);
                if (own) VirtualFree(own, 0, MEM_RELEASE);
            }
            if (v) { v[(1 << 20) - 1] = 1; VirtualFree(v, 0, MEM_RELEASE); }
            /* A UE5 title's two calls: reserve 112 MiB for physical pages with console protection
             * bits, then commit 64 KiB of it the same way. */
            v = valloc(NULL, 0x7000000, 0x20402000, 0x23, 0x10404);
            CHECK(v != NULL, "XMemVirtualAlloc reserving for physical pages (MEM_PHYSICAL, protect 0x10404) -> %p", v);
            if (v)
            {
                BYTE *c = valloc(v + 0x10000, 0x10000, 0x20401000, 0x3, 0x2404);
                CHECK(c == v + 0x10000, "XMemVirtualAlloc committing in it (MEM_PHYSICAL, protect 0x2404) -> %p", c);
                if (c) c[0xffff] = 1;
                VirtualFree(v, 0, MEM_RELEASE);
            }
            /* The target title's pair: a reservation, then one of the same size with no CPU access and
             * GPU bits (0x801), its GPU view: an address in the second maps to the first. */
            {
                ULONGLONG (WINAPI *to_cpu)(ULONGLONG) = (void *)GetProcAddress(m, "ReboxGpuViewToCpu");
                BOOL (WINAPI *xfree)(void *, SIZE_T, DWORD) = (void *)GetProcAddress(m, "XMemVirtualFree");
                BYTE *cpu = valloc(NULL, 0x4000000, 0x20402000, 0x21, PAGE_READWRITE);
                BYTE *gpu = valloc(NULL, 0x4000000, 0x20402000, 0x3, 0x801);
                CHECK(to_cpu && cpu && gpu && to_cpu((ULONG_PTR)gpu + 0x12340) == (ULONG_PTR)cpu + 0x12340
                      && to_cpu((ULONG_PTR)cpu + 0x10) == (ULONG_PTR)cpu + 0x10,
                      "a no-access GPU reservation after a reservation of its size is its GPU view (%p, %p)", cpu, gpu);
                if (gpu && xfree) xfree(gpu, 0, MEM_RELEASE);
                if (cpu && xfree) xfree(cpu, 0, MEM_RELEASE);
                CHECK(!to_cpu || !gpu || !xfree || to_cpu((ULONG_PTR)gpu) == (ULONG_PTR)gpu,
                      "the GPU view is forgotten when released");
            }
            /* A placeholder reservation (the target title's: 185 MiB, type 0x42000, PAGE_NOACCESS),
             * split with VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) as the title does, then one
             * piece replaced by committed memory and another given physical pages. */
            {
                typedef BOOL (WINAPI *vfree_fn)(void *, SIZE_T, DWORD);
                vfree_fn vfree = (vfree_fn)(void *)GetProcAddress(m, "XMemVirtualFree");
                BYTE *ph = valloc(NULL, 0xb900000, 0x42000, 0x7, PAGE_NOACCESS), *in;
                ULONG_PTR one = 1, page[1];
                CHECK(ph != NULL, "XMemVirtualAlloc reserves a placeholder (type 0x42000) -> %p", ph);
                CHECK((ULONG_PTR)ph < (1ull << 46), "the placeholder is below 2^46, the console's address width (%p)", ph);
                if (ph && vfree)
                {
                    CHECK(VirtualFree(ph, 1 << 16, MEM_RELEASE | 0x2 /* PRESERVE_PLACEHOLDER */)
                          && VirtualFree(ph + (1 << 16), 1 << 16, MEM_RELEASE | 0x2),
                          "the placeholder splits with VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)");
                    in = valloc(ph, 1 << 16, MEM_RESERVE | MEM_COMMIT | 0x4000 /* REPLACE_PLACEHOLDER */, 0,
                                PAGE_READWRITE);
                    CHECK(in == ph, "XMemVirtualAlloc replaces a split placeholder with committed memory -> %p", in);
                    if (in) in[0xffff] = 1;
                    CHECK(palloc(0, &one, page) && pmap(ph + (1 << 16), 1, page),
                          "XMemMapPhysicalPages fills a split placeholder");
                    if (in) ph[(1 << 16) + 0xffff] = 2;
                    vfree(ph, 0, MEM_RELEASE);
                    vfree(ph + (1 << 16), 0, MEM_RELEASE);
                    vfree(ph + (2 << 16), 0, MEM_RELEASE);
                }
            }
            range = VirtualAlloc(NULL, 16 << 16, MEM_RESERVE, PAGE_NOACCESS);
            CHECK(palloc(0x20000000, &count, pages) && count == 16 && pages[0] != pages[15],
                  "XMemAllocatePhysicalPages hands out 16 distinct pages");
            /* 4 KiB frame numbers: a title packs them as >> 4 and back as (packed << 4) & 0x1fffff0. */
            CHECK(!(pages[0] & 15) && !(pages[15] & 15) && ((pages[3] >> 4) << 4 & 0x1fffff0) == pages[3]
                  && pages[1] - pages[0] == 16,
                  "physical page numbers are 4 KiB frames, 16 a page (%#llx, %#llx)", (unsigned long long)pages[0],
                  (unsigned long long)pages[1]);
            CHECK(range && pmap(range, 16, pages), "XMemMapPhysicalPages maps them into a reserved range");
            if (range)
            {
                range[0] = 0x5a;
                range[(16 << 16) - 1] = 0xa5;
                CHECK(range[0] == 0x5a && range[(16 << 16) - 1] == 0xa5, "mapped physical pages are writable");
                CHECK(pmap(range, 16, NULL), "XMemMapPhysicalPages with no pages unmaps");
                VirtualFree(range, 0, MEM_RELEASE);
            }
            /* Scattered: page i to addresses[i] (pages 3 and 9 of a fresh range), then unmapped. */
            {
                BOOL (WINAPI *scatter)(void **, ULONG_PTR, ULONG_PTR *) =
                        (void *)GetProcAddress(m, "XMemMapPhysicalPagesScatter");
                BYTE *r2 = VirtualAlloc(NULL, 16 << 16, MEM_RESERVE, PAGE_NOACCESS);
                void *where[2] = { r2 + (3 << 16), r2 + (9 << 16) };
                CHECK(scatter && r2 && scatter(where, 2, pages), "XMemMapPhysicalPagesScatter maps two pages apart");
                if (scatter && r2)
                {
                    /* Page 0 was written at range and unmapped there: what it held comes with it. */
                    CHECK(r2[3 << 16] == 0x5a, "a physical page's contents follow it to a new address (%#x)",
                          r2[3 << 16]);
                    MEMORY_BASIC_INFORMATION mbi = { 0 };
                    r2[(3 << 16) + 1] = 1;
                    r2[(9 << 16) + 0xffff] = 2;
                    CHECK(scatter(where, 2, NULL) && VirtualQuery(where[1], &mbi, sizeof(mbi))
                          && mbi.State != MEM_COMMIT, "XMemMapPhysicalPagesScatter with no pages unmaps them");
                }
                if (r2) VirtualFree(r2, 0, MEM_RELEASE);
            }
            /* Two pages mapped twice at once into a range reserved for physical pages (MEM_PHYSICAL),
             * as a page pool and streaming do: both addresses are the same memory. */
            {
                typedef BOOL (WINAPI *vfree_fn)(void *, SIZE_T, DWORD);
                vfree_fn vfree = (vfree_fn)(void *)GetProcAddress(m, "XMemVirtualFree");
                BYTE *r3 = valloc(NULL, 4 << 16, MEM_RESERVE | 0x400000 /* MEM_PHYSICAL */, 0x3, PAGE_READWRITE);
                BOOL shared = r3 && vfree && pmap(r3, 2, pages + 4) && pmap(r3 + (2 << 16), 2, pages + 4);
                if (shared)
                {
                    r3[(1 << 16) + 7] = 0x3c;
                    r3[(2 << 16) + 9] = 0xc3;
                    shared = r3[(3 << 16) + 7] == 0x3c && r3[9] == 0xc3;
                }
                CHECK(shared, "a physical page mapped at two addresses is the same memory at both");
                /* Four pages in a row (one view), the middle two unmapped: the outer two stay. */
                if (shared)
                {
                    shared = pmap(r3, 4, NULL) && pmap(r3, 4, pages + 4);
                    if (shared)
                    {
                        r3[5] = 0x11;
                        r3[(3 << 16) + 5] = 0x33;
                        shared = pmap(r3 + (1 << 16), 2, NULL) && r3[5] == 0x11 && r3[(3 << 16) + 5] == 0x33;
                    }
                    CHECK(shared, "unmapping the middle of four pages mapped together keeps the outer two");
                }
                /* A page mapped as a copy (ordinary memory) and as a view at once: what is written
                 * through the view survives unmapping the copy, which gives back only its own writes. */
                if (r3)
                {
                    BYTE *plain = VirtualAlloc(NULL, 1 << 16, MEM_RESERVE, PAGE_READWRITE);
                    BOOL kept = plain && pmap(r3, 4, NULL) && pmap(plain, 1, pages + 8) && pmap(r3, 1, pages + 8);
                    if (kept)
                    {
                        r3[100] = 0x77;   /* through the view */
                        plain[200] = 0x88; /* through the copy */
                        kept = pmap(plain, 1, NULL) && r3[100] == 0x77 && r3[200] == 0x88;
                    }
                    CHECK(kept, "unmapping a copy of a physical page keeps what was written through its view");
                    if (plain) VirtualFree(plain, 0, MEM_RELEASE);
                }
                /* A copy's snapshot is a hash per 128 bytes (REBOX_COPY_SNAPSHOT), not a copy of
                 * its 64 KiB: a write into the page (ReboxWritePhysicalPages) part way into a block
                 * and one through the copy beside it both reach the page; one through the copy and
                 * one through a view into the same block are a conflict, and the page's stays. */
                if (r3)
                {
                    void (WINAPI *stats)(UINT64 *, UINT64 *, UINT64 *, UINT64 *) =
                            (void *)GetProcAddress(m, "ReboxPhysicalMemoryStats");
                    BOOL (WINAPI *pwrite)(const ULONG_PTR *, ULONG_PTR, SIZE_T, const void *, SIZE_T) =
                            (void *)GetProcAddress(m, "ReboxWritePhysicalPages");
                    BYTE *plain = VirtualAlloc(NULL, 1 << 16, MEM_RESERVE, PAGE_READWRITE);
                    UINT64 allocated = 0, copies = 0, before = 0, during = 0, conflicts0 = 0, conflicts = 0, x;
                    static const BYTE ds[3] = { 0xd1, 0xd2, 0xd3 };
                    BOOL ok = stats && pwrite && plain && pmap(r3, 4, NULL);
                    if (ok)
                        stats(&allocated, &copies, &before, &conflicts0);
                    ok = ok && pmap(plain, 1, pages + 9) && pmap(r3, 1, pages + 9);
                    if (ok)
                    {
                        stats(&allocated, &copies, &during, &x);
                        ok = copies >= 1 && during - before == (64 << 10) / 128 * 8 && allocated >= 16 << 16;
                        CHECK(ok, "a copy of a 64 KiB physical page keeps a 4 KiB snapshot (%llu bytes more, %llu copies, "
                              "%llu allocated)", (unsigned long long)(during - before), (unsigned long long)copies,
                              (unsigned long long)allocated);
                        /* Bytes 0x404-0x406 written into the page (and its copy), 0x410 through the
                         * copy: one block. 0x500 through the copy, 0x508 through the view: another. */
                        ok = pwrite(pages + 9, 1, 0x404, ds, sizeof(ds));
                        plain[0x410] = 0x41;
                        plain[0x500] = 0x50;
                        r3[0x508] = 0x58;
                        ok = ok && pmap(plain, 1, NULL);
                        stats(&allocated, &copies, &x, &conflicts);
                        CHECK(ok && r3[0x404] == 0xd1 && r3[0x406] == 0xd3 && r3[0x410] == 0x41,
                              "a write into a copied page and one through the copy beside it both reach the page");
                        CHECK(ok && r3[0x508] == 0x58 && r3[0x500] != 0x50 && conflicts == conflicts0 + 1 && x == before,
                              "a block written through the copy and the page's view keeps the page's (%llu conflicts)",
                              (unsigned long long)(conflicts - conflicts0));
                    }
                    else
                        CHECK(ok, "a physical page mapped as a copy and a view for the snapshot test");
                    if (plain) VirtualFree(plain, 0, MEM_RELEASE);
                }
                /* CopyPageMappingsBatchX's mapping (ReboxMapPhysicalAlias): the pages at a pool
                 * address mapped at another as well, the same memory at both. */
                {
                    BOOL (WINAPI *alias)(ULONGLONG, ULONGLONG, ULONGLONG) =
                            (void *)GetProcAddress(m, "ReboxMapPhysicalAlias");
                    BYTE *heap = valloc(NULL, 2 << 16, MEM_RESERVE | 0x400000 /* MEM_PHYSICAL */, 0x3, PAGE_READWRITE);
                    BOOL aliased = alias && heap && r3 && pmap(r3, 4, NULL) && pmap(r3, 2, pages + 10)
                            && alias((ULONG_PTR)heap, (ULONG_PTR)r3, 2 << 16);
                    if (aliased)
                    {
                        r3[(1 << 16) + 3] = 0x5e;
                        aliased = heap[(1 << 16) + 3] == 0x5e && alias((ULONG_PTR)heap, 0, 2 << 16);
                    }
                    CHECK(aliased, "ReboxMapPhysicalAlias maps a pool's pages at another address, then unmaps them");
                    if (heap && vfree) vfree(heap, 0, MEM_RELEASE);
                }
                CHECK(r3 && pmap(r3, 4, NULL) && vfree && vfree(r3, 0, MEM_RELEASE),
                      "a range reserved for physical pages unmaps and releases");
            }
            CHECK(pfree(16, pages), "XMemFreePhysicalPages");
        }
    }
}

static BYTE *read_file(const char *path, SIZE_T *size)
{
    FILE *f = fopen(path, "rb");
    BYTE *data;
    long len;

    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = malloc(len);
    fread(data, 1, len, f);
    fclose(f);
    *size = len;
    return data;
}

static void test_d3d12(const char *fake_path, BOOL expect_hit)
{
    static const char *forwarded[] = {
        "D3D12GetDebugInterface", "D3D12SerializeRootSignature", "D3D12SerializeVersionedRootSignature",
        "D3D12CreateRootSignatureDeserializer", "D3D12CreateVersionedRootSignatureDeserializer",
        "D3D12EnableExperimentalFeatures", "D3D12GetInterface",
    };
    HMODULE m = LoadLibraryA("d3d12_xs.dll");
    PFN_D3D12_CREATE_DEVICE create_device;
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize;
    D3D12_COMPUTE_PIPELINE_STATE_DESC cdesc = {0};
    D3D12_ROOT_SIGNATURE_DESC rs_desc = {0};
    ID3D10Blob *rs_blob = NULL, *err = NULL;
    ID3D12RootSignature *rs = NULL;
    ID3D12PipelineState *pso;
    ID3D12Device2 *device;
    SIZE_T fake_size;
    BYTE *fake;
    unsigned int i;
    HRESULT hr;

    CHECK(m != NULL, "load d3d12_xs.dll");
    if (!m) return;
    for (i = 0; i < ARRAYSIZE(forwarded); i++)
        CHECK(GetProcAddress(m, forwarded[i]) != NULL, "forwarded export %s resolves", forwarded[i]);
    CHECK(GetProcAddress(m, "D3D12XboxCreateDevice") != NULL, "D3D12XboxCreateDevice present");

    create_device = (PFN_D3D12_CREATE_DEVICE)(void *)GetProcAddress(m, "D3D12CreateDevice");
    serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)GetProcAddress(m, "D3D12SerializeRootSignature");
    hr = create_device(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device2, (void **)&device);
    if (FAILED(hr))
    {
        printf("skip D3D12CreateDevice failed %#lx (no usable Vulkan device?)\n", hr);
        return;
    }
    CHECK(1, "D3D12CreateDevice through d3d12_xs");

    hr = serialize(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &rs_blob, &err);
    CHECK(SUCCEEDED(hr), "serialize root signature via forwarder (%#lx)", hr);
    if (FAILED(hr)) return;
    hr = ID3D12Device2_CreateRootSignature(device, 0, ID3D10Blob_GetBufferPointer(rs_blob),
            ID3D10Blob_GetBufferSize(rs_blob), &IID_ID3D12RootSignature, (void **)&rs);
    CHECK(SUCCEEDED(hr), "create root signature (%#lx)", hr);
    if (FAILED(hr)) return;

    /* Console root signatures carry flags 0x03000000 (XboxBvhBuilder.dll adds them at desc +0x28
     * before serializing; its 52 embedded RTS0 parts all have them). Windows' runtime rejects these
     * bits with E_INVALIDARG. vkd3d-proton ignores them, so d3d12_xs passes them through. If this
     * starts failing, mask them in the forwarders and in CreateRootSignature. */
    {
        PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE serialize_v = (PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE)
                (void *)GetProcAddress(m, "D3D12SerializeVersionedRootSignature");
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC vdesc = { .Version = D3D_ROOT_SIGNATURE_VERSION_1_1,
                .Desc_1_1.Flags = 0x03000000 };
        ID3D12RootSignature *console_rs = NULL;
        ID3D10Blob *blob = NULL;

        hr = serialize_v(&vdesc, &blob, NULL);
        CHECK(SUCCEEDED(hr), "serialize root signature with console flags 0x03000000 (%#lx)", hr);
        if (SUCCEEDED(hr))
        {
            hr = ID3D12Device2_CreateRootSignature(device, 0, ID3D10Blob_GetBufferPointer(blob),
                    ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&console_rs);
            CHECK(SUCCEEDED(hr), "create root signature with console flags (%#lx)", hr);
            if (SUCCEEDED(hr)) ID3D12RootSignature_Release(console_rs);
            ID3D10Blob_Release(blob);
        }
    }

    if (!(fake = read_file(fake_path, &fake_size)))
    {
        CHECK(0, "read %s", fake_path);
        return;
    }

    cdesc.pRootSignature = rs;
    cdesc.CS.pShaderBytecode = fake;
    cdesc.CS.BytecodeLength = fake_size;
    hr = ID3D12Device2_CreateComputePipelineState(device, &cdesc, &IID_ID3D12PipelineState, (void **)&pso);
    CHECK(SUCCEEDED(hr) == expect_hit, "CreateComputePipelineState with console bytecode %s (%#lx)", expect_hit ? "substituted" : "rejected without cache entry", hr);
    if (SUCCEEDED(hr)) ID3D12PipelineState_Release(pso);

    {
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; ID3D12RootSignature *v; } __attribute__((aligned(8))) s_rs;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; D3D12_SHADER_BYTECODE v; } __attribute__((aligned(8))) s_cs;
        struct { __typeof__(s_rs) rs; __typeof__(s_cs) cs; } stream;
        D3D12_PIPELINE_STATE_STREAM_DESC sdesc = { sizeof(stream), &stream };

        stream.rs.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
        stream.rs.v = rs;
        stream.cs.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS;
        stream.cs.v = cdesc.CS;
        hr = ID3D12Device2_CreatePipelineState(device, &sdesc, &IID_ID3D12PipelineState, (void **)&pso);
        CHECK(SUCCEEDED(hr) == expect_hit, "CreatePipelineState stream with console bytecode %s (%#lx)", expect_hit ? "substituted" : "rejected without cache entry", hr);
        if (SUCCEEDED(hr)) ID3D12PipelineState_Release(pso);
    }

    /* A graphics stream with the newer subobjects (DEPTH_STENCIL2 = 26,
     * RASTERIZER2 = 28) and untranslated console VS/PS: d3d12_xs must parse
     * past them to substitute fallbacks; strict mode must still fail. */
    {
        const char *fb = getenv("REBOX_SHADER_FALLBACK");
        BOOL strict = fb && *fb == '0';
        static const char vs_bytes[] = "REBOX-FAKE-CONSOLE-VS", ps_bytes[] = "REBOX-FAKE-CONSOLE-PS";
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; ID3D12RootSignature *v; } __attribute__((aligned(8))) s_rs;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; D3D12_SHADER_BYTECODE v; } __attribute__((aligned(8))) s_bc;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; UINT v[10]; } __attribute__((aligned(8))) s_rast2;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; UINT v[15]; } __attribute__((aligned(8))) s_ds2;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; struct D3D12_RT_FORMAT_ARRAY v; } __attribute__((aligned(8))) s_rt;
        struct { __typeof__(s_rs) rs; __typeof__(s_bc) vs; __typeof__(s_rast2) rast; __typeof__(s_ds2) ds;
                 __typeof__(s_bc) ps; __typeof__(s_rt) rt; } gstream;
        D3D12_PIPELINE_STATE_STREAM_DESC gdesc = { sizeof(gstream), &gstream };

        memset(&gstream, 0, sizeof(gstream));
        gstream.rs.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
        gstream.rs.v = rs;
        gstream.vs.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
        gstream.vs.v.pShaderBytecode = vs_bytes;
        gstream.vs.v.BytecodeLength = sizeof(vs_bytes);
        gstream.rast.t = 28; /* RASTERIZER2: solid, cull back, depth clip on */
        gstream.rast.v[0] = D3D12_FILL_MODE_SOLID;
        gstream.rast.v[1] = D3D12_CULL_MODE_BACK;
        gstream.rast.v[6] = TRUE;
        gstream.ds.t = 26; /* DEPTH_STENCIL2: depth and stencil off */
        gstream.ds.v[1] = D3D12_DEPTH_WRITE_MASK_ALL;
        gstream.ds.v[2] = D3D12_COMPARISON_FUNC_LESS;
        gstream.ps.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
        gstream.ps.v.pShaderBytecode = ps_bytes;
        gstream.ps.v.BytecodeLength = sizeof(ps_bytes);
        gstream.rt.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
        gstream.rt.v.NumRenderTargets = 1;
        gstream.rt.v.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        hr = ID3D12Device2_CreatePipelineState(device, &gdesc, &IID_ID3D12PipelineState, (void **)&pso);
        CHECK(SUCCEEDED(hr) != strict, "graphics stream with DEPTH_STENCIL2/RASTERIZER2 and console VS/PS %s (%#lx)",
              strict ? "rejected in strict mode" : "gets fallbacks", hr);
        if (SUCCEEDED(hr)) ID3D12PipelineState_Release(pso);
    }

    /* A CachedPSO blob vkd3d-proton didn't make (as a console PSO cache entry
     * would be) must not fail creation: d3d12_xs retries without it. */
    {
        static const BYTE foreign_blob[64] = { 'P', 'S', '0', '9' };
        D3D12_COMPUTE_PIPELINE_STATE_DESC cached_desc = cdesc;

        cached_desc.CS.pShaderBytecode = test_empty_cs;
        cached_desc.CS.BytecodeLength = sizeof(test_empty_cs);
        cached_desc.CachedPSO.pCachedBlob = foreign_blob;
        cached_desc.CachedPSO.CachedBlobSizeInBytes = sizeof(foreign_blob);
        hr = ID3D12Device2_CreateComputePipelineState(device, &cached_desc, &IID_ID3D12PipelineState, (void **)&pso);
        CHECK(SUCCEEDED(hr), "compute PSO with a foreign CachedPSO blob is created without it (%#lx)", hr);
        if (SUCCEEDED(hr)) ID3D12PipelineState_Release(pso);
    }

    /* PC-format bytecode that doesn't compile (Xbox DXIL with @xbox.* intrinsics, as in
     * XboxBvhBuilder) gets fallbacks too: here a DXIL container with its bitcode scribbled on. */
    {
        const char *fb = getenv("REBOX_SHADER_FALLBACK");
        BOOL strict = fb && *fb == '0';
        D3D12_COMPUTE_PIPELINE_STATE_DESC bad_desc = cdesc;
        BYTE *bad = malloc(sizeof(test_empty_cs));

        memcpy(bad, test_empty_cs, sizeof(test_empty_cs));
        for (size_t i = sizeof(test_empty_cs) / 2; i < sizeof(test_empty_cs); i++)
            bad[i] ^= 0x5a;
        bad_desc.CS.pShaderBytecode = bad;
        bad_desc.CS.BytecodeLength = sizeof(test_empty_cs);
        pso = NULL;
        hr = ID3D12Device2_CreateComputePipelineState(device, &bad_desc, &IID_ID3D12PipelineState, (void **)&pso);
        CHECK(SUCCEEDED(hr) != strict, "compute PSO whose PC bytecode doesn't compile %s (%#lx)",
              strict ? "fails in strict mode" : "gets the fallback shader", hr);
        if (SUCCEEDED(hr)) ID3D12PipelineState_Release(pso);

        /* The same through a pipeline stream. */
        {
            struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; ID3D12RootSignature *v; } __attribute__((aligned(8))) s_rs;
            struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; D3D12_SHADER_BYTECODE v; } __attribute__((aligned(8))) s_cs;
            struct { __typeof__(s_rs) rs; __typeof__(s_cs) cs; } cstream;
            D3D12_PIPELINE_STATE_STREAM_DESC sdesc = { sizeof(cstream), &cstream };

            cstream.rs.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
            cstream.rs.v = cdesc.pRootSignature;
            cstream.cs.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS;
            cstream.cs.v = bad_desc.CS;
            pso = NULL;
            hr = ID3D12Device2_CreatePipelineState(device, &sdesc, &IID_ID3D12PipelineState, (void **)&pso);
            CHECK(SUCCEEDED(hr) != strict, "compute stream whose PC bytecode doesn't compile %s (%#lx)",
                  strict ? "fails in strict mode" : "gets the fallback shader", hr);
            if (SUCCEEDED(hr)) ID3D12PipelineState_Release(pso);
        }
        free(bad);
    }

    /* A mesh pipeline from untranslated console AS/MS/PS gets the mesh
     * fallbacks (when the device has mesh shaders and fallbacks are on). */
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7 = { 0 };
        const char *fb = getenv("REBOX_SHADER_FALLBACK");
        static const char as_bytes[] = "REBOX-FAKE-CONSOLE-AS", ms_bytes[] = "REBOX-FAKE-CONSOLE-MS",
                          ps_bytes[] = "REBOX-FAKE-CONSOLE-MESH-PS";
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; ID3D12RootSignature *v; } __attribute__((aligned(8))) s_rs;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; D3D12_SHADER_BYTECODE v; } __attribute__((aligned(8))) s_as, s_ms, s_ps;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t; struct D3D12_RT_FORMAT_ARRAY v; } __attribute__((aligned(8))) s_rt;
        struct { __typeof__(s_rs) rs; __typeof__(s_as) as, ms, ps; __typeof__(s_rt) rt; } mstream;
        D3D12_PIPELINE_STATE_STREAM_DESC mdesc = { sizeof(mstream), &mstream };

        if (FAILED(ID3D12Device2_CheckFeatureSupport(device, D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7)))
            || o7.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED || (fb && *fb == '0'))
            printf("skip mesh fallback check (mesh tier %u)\n", o7.MeshShaderTier);
        else
        {
            memset(&mstream, 0, sizeof(mstream));
            mstream.rs.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
            mstream.rs.v = rs;
            mstream.as.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS;
            mstream.as.v.pShaderBytecode = as_bytes;
            mstream.as.v.BytecodeLength = sizeof(as_bytes);
            mstream.ms.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
            mstream.ms.v.pShaderBytecode = ms_bytes;
            mstream.ms.v.BytecodeLength = sizeof(ms_bytes);
            mstream.ps.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
            mstream.ps.v.pShaderBytecode = ps_bytes;
            mstream.ps.v.BytecodeLength = sizeof(ps_bytes);
            mstream.rt.t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
            mstream.rt.v.NumRenderTargets = 1;
            mstream.rt.v.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            hr = ID3D12Device2_CreatePipelineState(device, &mdesc, &IID_ID3D12PipelineState, (void **)&pso);
            CHECK(SUCCEEDED(hr), "mesh PSO from untranslated console AS/MS/PS gets fallbacks (%#lx)", hr);
            if (SUCCEEDED(hr)) ID3D12PipelineState_Release(pso);
        }
    }

    ID3D12RootSignature_Release(rs);
    ID3D12Device2_Release(device);
}

/* Call an Xbox vtable slot by number, as title code does. */
#define XSLOT(obj, n, ret, ...) ((ret (STDMETHODCALLTYPE *)(void *, ##__VA_ARGS__))((*(void ***)(obj))[n]))

typedef HRESULT (WINAPI *xbox_create_fn)(IUnknown *, const void *, REFIID, void **);
typedef HRESULT (WINAPI *gpu_alloc_fn)(SIZE_T, SIZE_T, void **, UINT64 *);
typedef void (WINAPI *gpu_free_fn)(void *);

static BOOL gpu_identity; /* set by test_gpu_memory */

static void test_gpu_memory(HMODULE m)
{
    gpu_alloc_fn galloc = (gpu_alloc_fn)(void *)GetProcAddress(m, "ReboxGpuAlloc");
    gpu_free_fn gfree = (gpu_free_fn)(void *)GetProcAddress(m, "ReboxGpuFree");
    void *cpu = NULL;
    UINT64 gpu = 0;
    HRESULT hr;

    CHECK(galloc && gfree, "ReboxGpuAlloc/Free exported");
    if (!galloc || !gfree) return;
    hr = galloc(1 << 20, 64 << 10, &cpu, &gpu);
    CHECK(SUCCEEDED(hr) && cpu && gpu && !((ULONG_PTR)cpu & 0xffff), "GPU memory %p, GPU VA %#llx (%#lx)",
          cpu, (unsigned long long)gpu, hr);
    if (FAILED(hr)) return;
    memset(cpu, 0x5a, 1 << 20);
    /* Identity depends on the driver and layer; lavapipe gives it for free. */
    gpu_identity = (UINT64)(ULONG_PTR)cpu == gpu;
    printf("info GPU memory is %sidentity-mapped\n", gpu_identity ? "" : "NOT ");

    /* A second heap (32 MiB gets its own): if the first was identity-mapped,
     * later ones must be too. RADV pads each allocation's VA range, which once
     * made the next address collide. */
    {
        void *cpu2 = NULL;
        UINT64 gpu2 = 0;
        BOOL first = (UINT64)(ULONG_PTR)cpu == gpu;

        hr = galloc(32 << 20, 64 << 10, &cpu2, &gpu2);
        CHECK(SUCCEEDED(hr) && cpu2, "second GPU heap %p, GPU VA %#llx (%#lx)", cpu2, (unsigned long long)gpu2, hr);
        if (SUCCEEDED(hr))
        {
            memset(cpu2, 0x5a, 32 << 20);
            if (first)
                CHECK((UINT64)(ULONG_PTR)cpu2 == gpu2, "second GPU heap is identity-mapped too");
            gfree(cpu2);
        }
        /* Right after a free: RADV releases VA ranges lazily, so an
         * allocator that reuses the freed address gets refused. */
        hr = galloc(32 << 20, 64 << 10, &cpu2, &gpu2);
        CHECK(SUCCEEDED(hr) && cpu2, "GPU heap after a free %p, GPU VA %#llx (%#lx)", cpu2,
              (unsigned long long)gpu2, hr);
        if (SUCCEEDED(hr))
        {
            if (first)
                CHECK((UINT64)(ULONG_PTR)cpu2 == gpu2, "GPU heap after a free is identity-mapped too");
            gfree(cpu2);
        }
    }
    /* A V# in GPU memory whose records run past it (into title memory that isn't GPU memory) is
     * cut where that memory ends: the shader reads it at its address, bounded only by its length,
     * and read the rest unmapped. */
    {
        UINT64 (WINAPI *length)(UINT64, UINT64) = (void *)GetProcAddress(m, "ReboxGuestBufferLength");
        CHECK(length != NULL, "ReboxGuestBufferLength exported");
        /* Only title addresses (4 GiB to 2^46) are checked; others are vkd3d-proton's own, kept whole. */
        if (length && gpu >= 0x400000000000ull)
            CHECK(length(gpu, 1ull << 32) == 1ull << 32, "a V# at a non-title address keeps its length");
        if (length && gpu_identity && gpu >= 0x100000000ull && gpu + (1ull << 32) <= 0x200000000000ull)
        {
            UINT64 whole = length(gpu, 256), cut = length(gpu, 1ull << 32);
            CHECK(whole == 256, "a V# inside GPU memory keeps its length (%#llx)", (unsigned long long)whole);
            CHECK(cut >= 1 << 20 && cut < 1ull << 32, "a V# running past GPU memory is cut at its end (%#llx)",
                  (unsigned long long)cut);
        }
        else if (length)
            printf("skip guest buffer length: GPU memory at %#llx isn't identity-mapped at a title address\n",
                   (unsigned long long)gpu);
    }
    gfree(cpu);
}

/* Texels a title writes through a texture's GPU VA (console textures are their memory) reach the
 * texture: d3d12_xs backs the VA with GPU memory it detiles into the texture before the queue
 * runs. One colour everywhere, so the result doesn't depend on which tiling is assumed. by_map:
 * through Map/Unmap instead (a GDK title maps a DEFAULT heap texture, which vkd3d-proton refuses). */
static void test_texture_va_upload(void *dev, void *queue, BOOL by_map)
{
    const char *how = by_map ? "Map" : "GPU VA";
    D3D12_HEAP_PROPERTIES def = { D3D12_HEAP_TYPE_DEFAULT }, rb = { D3D12_HEAP_TYPE_READBACK };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 256, 256, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 1024 * 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_RESOURCE_BARRIER barrier = { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
    D3D12_TEXTURE_COPY_LOCATION dst = { 0 }, src = { 0 };
    ID3D12Resource *tex = NULL, *readback = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *list = NULL, *mapped;
    UINT32 *texels;
    UINT64 va = 0;
    HRESULT hr;

    hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
            D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &def, D3D12_HEAP_FLAG_NONE,
            &td, D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&tex);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &rb, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&readback);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList,
                &list);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr) && by_map)
    {
        void *p = NULL;
        hr = ID3D12Resource_Map(tex, 0, NULL, &p);
        va = (UINT64)(ULONG_PTR)p;
    }
    else if (SUCCEEDED(hr))
        va = ID3D12Resource_GetGPUVirtualAddress(tex);
    if (SUCCEEDED(hr) && !va)
        hr = E_FAIL;
    CHECK(SUCCEEDED(hr), "texture, readback and list for texels written through a %s (%#lx)", how, hr);
    if (FAILED(hr))
        goto done;
    texels = (UINT32 *)(ULONG_PTR)va;
    for (UINT i = 0; i < 256 * 256; i++)
        texels[i] = 0xff20c040;
    if (by_map)
        ID3D12Resource_Unmap(tex, 0, NULL);

    barrier.Transition.pResource = tex;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, &barrier);
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = (D3D12_SUBRESOURCE_FOOTPRINT){ DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, 1024 };
    src.pResource = tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
          const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
    for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
        Sleep(10);
    if (SUCCEEDED(ID3D12Resource_Map(readback, 0, NULL, &mapped)))
    {
        UINT32 a = ((UINT32 *)mapped)[0], b = ((UINT32 *)mapped)[100 * 256 + 37];
        CHECK(a == 0xff20c040 && b == 0xff20c040,
              "texels written through a texture's %s reach the texture (%08x %08x, want ff20c040)", how, a, b);
        ID3D12Resource_Unmap(readback, 0, NULL);
    }

    /* One texel changed later, between the points a sampled fingerprint reads (Unreal's font
     * atlas gains glyphs so): the next submission uploads it. Wherever the tiling puts it. */
    if (by_map)
    {
        void *p = NULL;
        ID3D12Resource_Map(tex, 0, NULL, &p);
        texels = p ? p : texels;
    }
    texels[1000] = 0xff0000ff;
    if (by_map)
        ID3D12Resource_Unmap(tex, 0, NULL);
    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, &barrier);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, &barrier);
    XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
          const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 2);
    for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 2; i++)
        Sleep(10);
    if (SUCCEEDED(ID3D12Resource_Map(readback, 0, NULL, &mapped)))
    {
        UINT changed = 0, other = 0;
        for (UINT i = 0; i < 256 * 256; i++)
        {
            UINT32 v = ((UINT32 *)mapped)[i];
            changed += v == 0xff0000ff;
            other += v != 0xff0000ff && v != 0xff20c040;
        }
        CHECK(changed == 1 && !other, "a one-texel edit through the %s reaches the texture (%u changed, %u other)",
              how, changed, other);
        ID3D12Resource_Unmap(readback, 0, NULL);
    }

done:
    if (fence) ID3D12Fence_Release(fence);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (readback) ID3D12Resource_Release(readback);
    if (tex) ID3D12Resource_Release(tex);
}

/* REBOX_PROBE's mechanism (xmem probe.c): an int3 on a function logs each call and the function
 * still runs, every time (the original byte single-stepped, then rearmed). */
static __attribute__((noinline)) int probed_function(int a)
{
    return a * 3 + 1;
}

static void test_probe(HMODULE xmem)
{
    BOOL (WINAPI *add)(void *, const char *) = (void *)GetProcAddress(xmem, "ReboxProbeAdd");
    LONG (WINAPI *hits)(const void *) = (void *)GetProcAddress(xmem, "ReboxProbeHits");
    int (*volatile fn)(int) = probed_function;
    int sum = 0;

    CHECK(add && hits, "xmem exports ReboxProbeAdd/ReboxProbeHits");
    if (!add || !hits || !add((void *)probed_function, "smoke probed_function"))
        return;
    for (int i = 0; i < 3; i++)
        sum += fn(i);
    CHECK(sum == 1 + 4 + 7 && hits((void *)probed_function) == 3,
          "a probed function logs each call and still runs (sum %d, want 12; %ld hits, want 3)", sum,
          hits((void *)probed_function));
}

/* Reads pixel (0,0) of a 32-bit BMP written by PresentX's frame dump. */
static BOOL read_dump_pixel(const char *dir, int frame, BYTE bgra[4])
{
    char path[MAX_PATH];
    BYTE header[58];
    FILE *f;
    BOOL ok;

    snprintf(path, sizeof(path), "%s\\frame_%d.bmp", dir, frame);
    if (!(f = fopen(path, "rb")))
        return FALSE;
    ok = fread(header, 1, sizeof(header), f) == sizeof(header) && header[0] == 'B' && header[1] == 'M';
    fclose(f);
    if (ok)
        memcpy(bgra, header + 54, 4);
    return ok;
}

/* Render a cleared target through Xbox slots, then PresentX it (queue slot
 * 30), once with the resource in the plane params and once with a pointer
 * to it. The frame dumps (run-smoke.sh sets REBOX_DUMP_FRAMES) show
 * whether the right pixels were copied out; a window is shown only if
 * there's a display. */
/* A console container (DXBC, DXIL part, XBCH, zlib FSSM) around graphics FSSM in a UE5 title's older
 * layout (console_fssm.c): code at 0x160 (the u32 at 0xC0 plus 0x28), its size at 0xC8, RSRC1 at
 * 0x108, float mode at 0xFC. zlib in stored blocks, which puff reads. Free with free(). */
/* With an interface table (FSSM sub-version 5: the pixel table's records at 0x130, the vertex
 * table's at 0x1D4) and a signature part (sig_tag ISG1 or OSG1, its chunk), when given. */
static BYTE *console_container_around(BYTE *fssm, UINT fssm_size, const char *sig_tag, const BYTE *sig,
                                      UINT sig_size, SIZE_T *size);

static BYTE *console_graphics_container_ex(const UINT32 *code, UINT dwords, const BYTE *table, UINT table_at,
                                           UINT table_size, const char *sig_tag, const BYTE *sig, UINT sig_size, SIZE_T *size)
{
    UINT code_at = table ? 0x300 : 0x160;
    UINT fssm_size = code_at + dwords * 4;
    BYTE *fssm = calloc(1, fssm_size);

    memcpy(fssm, "FSSM", 4);
    *(UINT32 *)(fssm + 0xc0) = code_at - 0x28;
    *(UINT32 *)(fssm + 0xc8) = dwords * 4;
    *(UINT32 *)(fssm + 0xfc) = 0xc0;
    *(UINT32 *)(fssm + 0x108) = 0x002c0003;
    if (table)
    {
        *(UINT32 *)(fssm + 0x10) = 5;
        memcpy(fssm + table_at, table, table_size);
    }
    memcpy(fssm + code_at, code, dwords * 4);
    return console_container_around(fssm, fssm_size, sig_tag, sig, sig_size, size);
}

/* The container (DXBC, DXIL part, XBCH, zlib in stored blocks) around an FSSM, which it frees; with a
 * signature part when given. */
static BYTE *console_container_around(BYTE *fssm, UINT fssm_size, const char *sig_tag, const BYTE *sig,
                                      UINT sig_size, SIZE_T *size)
{
    UINT zsize = 2 + 5 + fssm_size + 4, part, total, parts = sig ? 2 : 1;
    UINT dxil_at = 0x20 + 4 * parts;
    UINT32 a = 1, b = 0;
    BYTE *out;

    part = 4 + 0x60 + 4 + 0x14 + zsize;
    part = (part + 3) & ~3u;
    total = dxil_at + 8 + part + (sig ? 8 + sig_size : 0);
    out = calloc(1, total);
    memcpy(out, "DXBC", 4);
    *(UINT32 *)(out + 20) = 1;
    *(UINT32 *)(out + 24) = total;
    *(UINT32 *)(out + 28) = parts;
    *(UINT32 *)(out + 32) = dxil_at;
    memcpy(out + dxil_at, "DXIL", 4);
    *(UINT32 *)(out + dxil_at + 4) = part;
    if (sig)
    {
        UINT at = dxil_at + 8 + part;
        *(UINT32 *)(out + 36) = at;
        memcpy(out + at + 8, sig, sig_size);
        *(UINT32 *)(out + at + 4) = sig_size;
        memcpy(out + at, sig_tag, 4);
    }
    {
        BYTE *p = out + dxil_at + 8;
        memcpy(p, "XBCH", 4);
        memcpy(p + 4 + 0x60, "NOBC", 4);
        p += 4 + 0x60 + 4 + 0x14;
        p[0] = 0x78;
        p[1] = 0x01;
        p[2] = 0x01; /* final stored block */
        p[3] = fssm_size & 0xff;
        p[4] = fssm_size >> 8;
        p[5] = ~fssm_size & 0xff;
        p[6] = (~fssm_size >> 8) & 0xff;
        memcpy(p + 7, fssm, fssm_size);
        for (UINT i = 0; i < fssm_size; i++)
        {
            a = (a + fssm[i]) % 65521;
            b = (b + a) % 65521;
        }
        p[7 + fssm_size] = b >> 8;
        p[8 + fssm_size] = b & 0xff;
        p[9 + fssm_size] = a >> 8;
        p[10 + fssm_size] = a & 0xff;
    }
    free(fssm);
    *size = total;
    return out;
}

/* A signature chunk (ISG1/OSG1 layout: count, 8, then 32-byte elements from the stream on) naming
 * register regs[k] semantic names[k]0, a system value where sv[k]. */
static UINT signature_chunk(const char *const *names, const UINT *regs, const UINT *sv, UINT n, BYTE out[512])
{
    UINT strings = 8 + 32 * n;
    memset(out, 0, 512);
    *(UINT32 *)out = n;
    *(UINT32 *)(out + 4) = 8;
    for (UINT k = 0; k < n; k++)
    {
        UINT32 *e = (UINT32 *)(out + 8 + 32 * k);
        e[1] = strings;          /* name, from the chunk body */
        e[3] = sv[k];
        e[4] = 3;                /* float */
        e[5] = regs[k];
        e[6] = 0x0f0f;
        strcpy((char *)out + strings, names[k]);
        strings += (UINT)strlen(names[k]) + 1;
    }
    return (strings + 3) & ~3u;
}

static BYTE *console_graphics_container(const UINT32 *code, UINT dwords, SIZE_T *size)
{
    return console_graphics_container_ex(code, dwords, NULL, 0, 0, NULL, NULL, 0, size);
}

/* Deferred console graphics (patch 0006): a PSO whose stages are console vertex and pixel shaders
 * in their containers draws with AnyPS5's compiles of them. The vertex shader calls the fetch
 * shader as a UE5 title's do (s_swappc_b64 s[4:5], s[4:5]) and exports input 0 as the position, its x
 * and y scaled by a constant buffer's first value (1.0), whose V# it loads through descriptor table
 * 0 in s8 as a UE5 title's load theirs through s12: past the 5-bit RSRC2 user SGPR count. The pixel
 * shader exports red, as 32-bit floats or (packed) as two FP16 pairs, which every UE5 pixel
 * shader exports. One full-screen triangle over a blue clear: the centre turns red (blue: fell
 * back, or the constant read 0 and the triangle collapsed). */
/* rect: 1 a rect list drawn by AnyPS5's rect-list stages (vkd3d-proton patch 0011), 2 the same made
 * two triangles a rectangle on the CPU (REBOX_RECT_LISTS=cpu). */
static void test_console_draw(void *dev, void *queue, BOOL packed, BOOL linked, int rect)
{
    /* Linked: the vertex shader writes black to param0 and red to param1, whose registers are
     * TEXCOORD0 and COLOR0 (the FSSM table and OSG1); the pixel shader reads its one input, COLOR0,
     * as attr0 (flat). Red only if attr0 is linked to param1 by semantic, as the console driver
     * does and as a UE5 title's UI needs. */
    static const UINT32 link_vs[] = {
        0xbe842104,             /* s_swappc_b64 s[4:5], s[4:5] */
        0x7e100280,             /* v_mov_b32 v8, 0 */
        0x7e1202f2,             /* v_mov_b32 v9, 1.0 */
        0xf80008cf, 0x07060504, /* exp pos0 v4, v5, v6, v7 done */
        0xf800020f, 0x08080808, /* exp param0 v8, v8, v8, v8 */
        0xf800021f, 0x09080809, /* exp param1 v9, v8, v8, v9 */
        0xbf810000,             /* s_endpgm */
    };
    static const UINT32 link_ps[] = {
        0xc8020002,             /* v_interp_mov_f32 v0, p0, attr0.x */
        0xc8060102,             /* v_interp_mov_f32 v1, p0, attr0.y */
        0xc80a0202,             /* v_interp_mov_f32 v2, p0, attr0.z */
        0xc80e0302,             /* v_interp_mov_f32 v3, p0, attr0.w */
        0xf800180f, 0x03020100, /* exp mrt0 v0, v1, v2, v3 done vm */
        0xbf810000,             /* s_endpgm */
    };
    static const UINT32 vs_code[] = {
        0xbe842104,             /* s_swappc_b64 s[4:5], s[4:5]: the fetch, input 0 in v4-v7 */
        0xbe890380,             /* s_mov_b32 s9, 0 */
        0xf4080304, 0xfa000000, /* s_load_dwordx4 s[12:15], s[8:9], 0x0: table 0's V# */
        0xbf8cc07f,             /* s_waitcnt lgkmcnt(0) */
        0xf4200406, 0xfa000000, /* s_buffer_load_dword s16, s[12:15], 0x0 */
        0xbf8cc07f,             /* s_waitcnt lgkmcnt(0) */
        0x10080810,             /* v_mul_f32 v4, s16, v4 */
        0x100a0a10,             /* v_mul_f32 v5, s16, v5 */
        0xf80008cf, 0x07060504, /* exp pos0 v4, v5, v6, v7 done */
        0xbf810000,             /* s_endpgm */
    };
    static const float scale[64] = { 1.0f };
    static const UINT32 ps_code[] = {
        0x7e0002f2,             /* v_mov_b32 v0, 1.0 */
        0x7e020280,             /* v_mov_b32 v1, 0 */
        0xf800180f, 0x00010100, /* exp mrt0 v0, v1, v1, v0 done vm */
        0xbf810000,             /* s_endpgm */
    };
    static const UINT32 ps_packed_code[] = {
        0xd52f0000, 0x000100f2, /* v_cvt_pkrtz_f16_f32 v0, 1.0, 0: red, green */
        0xd52f0001, 0x0001e480, /* v_cvt_pkrtz_f16_f32 v1, 0, 1.0: blue, alpha */
        0xf8001c0f, 0x00000100, /* exp mrt0 v0, v0, v1, v1 done compr vm */
        0xbf810000,             /* s_endpgm */
    };
    static const float full[3][4] = { { -1, -1, 0, 1 }, { -1, 3, 0, 1 }, { 3, -1, 0, 1 } };
    /* rect: the console's rectangle list as a retail title's full-screen pass gives it, three corners;
     * its first triangle is the lower left half, the top right is the fourth corner's. */
    static const float corners[3][4] = { { -1, 1, 0, 1 }, { -1, -1, 0, 1 }, { 1, -1, 0, 1 } };
    const float (*tri)[4] = rect ? corners : full;
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    D3D12_HEAP_PROPERTIES def = { D3D12_HEAP_TYPE_DEFAULT }, up = { D3D12_HEAP_TYPE_UPLOAD },
                          rb = { D3D12_HEAP_TYPE_READBACK };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 48, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256 * 48, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_CLEAR_VALUE clear = { DXGI_FORMAT_R8G8B8A8_UNORM, { { 0.0f, 0.0f, 1.0f, 1.0f } } };
    D3D12_INPUT_ELEMENT_DESC element = { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,
                                         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 };
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "D3D12SerializeRootSignature");
    D3D12_DESCRIPTOR_RANGE range = { D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, 0 };
    D3D12_ROOT_PARAMETER param = { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, .DescriptorTable = { 1, &range },
                                   .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL };
    D3D12_ROOT_SIGNATURE_DESC rsd = { 1, &param, 0, NULL, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT };
    D3D12_DESCRIPTOR_HEAP_DESC cbv_hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gd = { 0 };
    D3D12_VIEWPORT vp = { 0, 0, 64, 48, 0, 1 };
    D3D12_RECT sc = { 0, 0, 64, 48 };
    D3D12_RESOURCE_BARRIER barrier = { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
    D3D12_TEXTURE_COPY_LOCATION dst = { 0 }, src = { 0 };
    D3D12_VERTEX_BUFFER_VIEW vbv;
    ID3D12Resource *tex = NULL, *vb = NULL, *cb = NULL, *readback = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *rtv_heap = NULL, *cbv_heap = NULL, *list = NULL, *rs = NULL, *pso = NULL, *mapped;
    BYTE *vs = NULL, *ps = NULL, *px;
    SIZE_T vs_size, ps_size, rtv;
    ID3D10Blob *blob = NULL;
    HRESULT hr;

    if (getenv("REBOX_SHADER_FALLBACK") && *getenv("REBOX_SHADER_FALLBACK") == '0')
        return;
    if (linked)
    {
        static const char *const vs_names[] = { "SV_Position", "TEXCOORD", "COLOR" }, *const ps_names[] = {
            "SV_Position", "COLOR" };
        static const UINT vs_regs[] = { 0, 1, 2 }, vs_sv[] = { 1, 0, 0 }, ps_regs[] = { 0, 1 }, ps_sv[] = { 1, 0 };
        BYTE vs_table[0xc4] = { 6, 1, 0, 0, 0, 0, 6, 2, 1, 0, 0, 0 }, ps_table[0x114] = { 6, 1, 0xf, 0, 0, 0, 1, 1 };
        BYTE sig[512];
        UINT n;
        vs_table[0xc0] = 2;
        ps_table[0x100] = 1;
        ps_table[0x110] = 2;
        n = signature_chunk(vs_names, vs_regs, vs_sv, 3, sig);
        vs = console_graphics_container_ex(link_vs, ARRAYSIZE(link_vs), vs_table, 0x1d4, sizeof(vs_table), "OSG1", sig,
                                           n, &vs_size);
        n = signature_chunk(ps_names, ps_regs, ps_sv, 2, sig);
        ps = console_graphics_container_ex(link_ps, ARRAYSIZE(link_ps), ps_table, 0x130, sizeof(ps_table), "ISG1", sig,
                                           n, &ps_size);
    }
    else
    {
        vs = console_graphics_container(vs_code, ARRAYSIZE(vs_code), &vs_size);
        ps = packed ? console_graphics_container(ps_packed_code, ARRAYSIZE(ps_packed_code), &ps_size)
                    : console_graphics_container(ps_code, ARRAYSIZE(ps_code), &ps_size);
    }
    hr = serialize && SUCCEEDED(serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL)) ? S_OK : E_FAIL;
    if (SUCCEEDED(hr))
    {
        hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
        ID3D10Blob_Release(blob);
    }
    gd.pRootSignature = rs;
    gd.VS.pShaderBytecode = vs;
    gd.VS.BytecodeLength = vs_size;
    gd.PS.pShaderBytecode = ps;
    gd.PS.BytecodeLength = ps_size;
    gd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gd.SampleMask = ~0u;
    gd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gd.InputLayout.pInputElementDescs = &element;
    gd.InputLayout.NumElements = 1;
    gd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gd.NumRenderTargets = 1;
    gd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    gd.SampleDesc.Count = 1;
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 10, HRESULT, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &gd,
                &IID_ID3D12PipelineState, &pso);
    CHECK(SUCCEEDED(hr), "graphics PSO from console vertex and pixel containers (%#lx)", hr);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
                &IID_ID3D12DescriptorHeap, &rtv_heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &def, D3D12_HEAP_FLAG_NONE,
                &td, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, &IID_ID3D12Resource, (void **)&tex);
    bd.Width = sizeof(full);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &up, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void **)&vb);
    bd.Width = sizeof(scale);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &up, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void **)&cb);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &cbv_hd,
                &IID_ID3D12DescriptorHeap, &cbv_heap);
    bd.Width = 256 * 48;
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &rb, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&readback);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList,
                &list);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                &IID_ID3D12Fence, (void **)&fence);
    CHECK(SUCCEEDED(hr), "render target, vertex buffer and list for a console draw (%#lx)", hr);
    if (FAILED(hr))
        goto done;
    if (SUCCEEDED(ID3D12Resource_Map(vb, 0, NULL, &mapped)))
    {
        memcpy(mapped, tri, sizeof(full));
        ID3D12Resource_Unmap(vb, 0, NULL);
    }
    if (SUCCEEDED(ID3D12Resource_Map(cb, 0, NULL, &mapped)))
    {
        memcpy(mapped, scale, sizeof(scale));
        ID3D12Resource_Unmap(cb, 0, NULL);
    }
    cbv.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(cb);
    cbv.SizeInBytes = sizeof(scale);
    XSLOT(dev, 17, void, const D3D12_CONSTANT_BUFFER_VIEW_DESC *, SIZE_T)(dev, &cbv, XSLOT(cbv_heap, 9, SIZE_T)(cbv_heap));

    rtv = XSLOT(rtv_heap, 9, SIZE_T)(rtv_heap);
    XSLOT(dev, 20, void, ID3D12Resource *, const void *, SIZE_T)(dev, tex, NULL, rtv);
    XSLOT(list, 52, void, SIZE_T, const FLOAT *, UINT, const D3D12_RECT *)(list, rtv, clear.Color, 0, NULL);
    XSLOT(list, 50, void, UINT, const SIZE_T *, BOOL, const SIZE_T *)(list, 1, &rtv, FALSE, NULL);
    XSLOT(list, 21, void, UINT, const D3D12_VIEWPORT *)(list, 1, &vp);
    XSLOT(list, 22, void, UINT, const D3D12_RECT *)(list, 1, &sc);
    XSLOT(list, 20, void, D3D12_PRIMITIVE_TOPOLOGY)(list, rect ? (D3D12_PRIMITIVE_TOPOLOGY)17
                                                                : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    XSLOT(list, 32, void, UINT, void *const *)(list, 1, &cbv_heap);
    XSLOT(list, 34, void, void *)(list, rs);
    XSLOT(list, 36, void, UINT, UINT64)(list, 0, XSLOT(cbv_heap, 10, UINT64)(cbv_heap));
    XSLOT(list, 25, void, void *)(list, pso);
    vbv.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(vb);
    vbv.SizeInBytes = sizeof(full);
    vbv.StrideInBytes = sizeof(full[0]);
    XSLOT(list, 48, void, UINT, UINT, const D3D12_VERTEX_BUFFER_VIEW *)(list, 0, 1, &vbv); /* PC 44 */
    XSLOT(list, 12, void, UINT, UINT, UINT, UINT)(list, 3, 1, 0, 0);
    barrier.Transition.pResource = tex;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, &barrier);
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = (D3D12_SUBRESOURCE_FOOTPRINT){ DXGI_FORMAT_R8G8B8A8_UNORM, 64, 48, 1, 256 };
    src.pResource = tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
          const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    hr = XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
    for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
        Sleep(10);
    if (SUCCEEDED(ID3D12Resource_Map(readback, 0, NULL, &mapped)))
    {
        px = (BYTE *)mapped + 24 * 256 + 32 * 4;
        if (rect)
        {
            const BYTE *top_right = (BYTE *)mapped + 6 * 256 + 56 * 4, *bottom_left = (BYTE *)mapped + 42 * 256 + 8 * 4;
            UINT missed = 0, first_x = 0, first_y = 0;
            /* Every pixel: the two triangles share their diagonal, with no seam between them. */
            for (UINT y = 0; y < 48; y++)
                for (UINT x = 0; x < 64; x++)
                {
                    const BYTE *q = (BYTE *)mapped + y * 256 + x * 4;
                    if (!(q[0] == 255 && q[1] == 0 && q[2] == 0) && !missed++)
                        first_x = x, first_y = y;
                }
            CHECK(!missed, "a console rect list covers every pixel %s: no seam on its diagonal (%u missed, first at %u,%u)",
                  rect == 2 ? "on the CPU" : "with the rect-list stages", missed, first_x, first_y);
            CHECK(bottom_left[0] == 255 && bottom_left[1] == 0 && top_right[0] == 255 && top_right[1] == 0
                  && top_right[2] == 0,
                  "a console rect list draws the whole rectangle: its fourth corner is made %s (bottom left RGB "
                  "%u %u %u, top right %u %u %u)", rect == 2 ? "on the CPU" : "by the rect-list stages",
                  bottom_left[0], bottom_left[1], bottom_left[2], top_right[0], top_right[1], top_right[2]);
        }
        else
        CHECK(px[0] == 255 && px[1] == 0 && px[2] == 0,
              "console vertex and pixel shaders draw through AnyPS5, %s (centre RGB %u %u %u; blue: fell back)",
              linked ? "pixel attr0 linked to vertex param1 by semantic (black: attr n = param n)"
              : packed ? "FP16 (compr) export" : "32-bit export", px[0], px[1], px[2]);
        ID3D12Resource_Unmap(readback, 0, NULL);
    }

done:
    if (fence) ID3D12Fence_Release(fence);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (readback) ID3D12Resource_Release(readback);
    if (vb) ID3D12Resource_Release(vb);
    if (cb) ID3D12Resource_Release(cb);
    if (tex) ID3D12Resource_Release(tex);
    if (rtv_heap) XSLOT(rtv_heap, 2, ULONG)(rtv_heap);
    if (cbv_heap) XSLOT(cbv_heap, 2, ULONG)(cbv_heap);
    if (pso) XSLOT(pso, 2, ULONG)(pso);
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    free(vs);
    free(ps);
}

static void test_present(void *dev, void *queue)
{
    const char *dump_dir = getenv("REBOX_DUMP_FRAMES");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    D3D12_HEAP_PROPERTIES heap_props = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 48, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                 { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET };
    D3D12_CLEAR_VALUE clear = { DXGI_FORMAT_R8G8B8A8_UNORM, { { 1.0f, 0.5f, 0.25f, 1.0f } } };
    D3D12_RESOURCE_BARRIER to_present = { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Resource *tex = NULL;
    void *rtv_heap = NULL, *list = NULL, *rs = NULL, *pso = NULL; /* alive until the GPU is done */
    UINT64 plane_params[6] = { 0 }, scratch_before = 0;
    SIZE_T rtv;
    BYTE px[4];
    HRESULT hr;

    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &rtv_heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &heap_props,
                D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, &IID_ID3D12Resource,
                (void **)&tex);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList,
                &list);
    CHECK(SUCCEEDED(hr), "render target, RTV heap and list for PresentX (%#lx)", hr);
    if (FAILED(hr)) goto done;

    /* A title's own top-level window (a GDK title makes one): the swapchain goes there. Without a
     * display there is none, and PresentX makes a window of its own. */
    {
        WNDCLASSW wc = { .lpfnWndProc = DefWindowProcW, .hInstance = GetModuleHandleW(NULL),
                         .lpszClassName = L"SmokeTitleWindow" };
        RegisterClassW(&wc);
        if (CreateWindowExW(0, wc.lpszClassName, L"title", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 0, 0, 320, 240,
                            NULL, NULL, wc.hInstance, NULL))
            printf("info a title window exists for PresentX\n");
    }

    /* Called before the device is made; title memory is identity-mapped, so it only succeeds. */
    {
        HRESULT (WINAPI *configure)(UINT64, UINT64, UINT64, UINT64) =
                (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "D3D12ConfigureVirtualMemoryX");
        CHECK(configure && configure(0, 0, 0, 0) == S_OK, "D3D12ConfigureVirtualMemoryX succeeds");
    }

    /* A descriptor heap in the title's memory: the title's hook places it (as the target title's
     * does), its CPU start is that memory, and a view written at a handle there reaches the
     * console-format shadow heap. */
    {
        D3D12_DESCRIPTOR_HEAP_DESC phd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
        HRESULT (WINAPI *shadow_heap)(SIZE_T, UINT64 *, void **, UINT *, UINT *) =
                (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxShadowHeap");
        UINT inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        void *pheap = NULL, *pdata = NULL;
        UINT pstride = 0, pcount = 0;
        SIZE_T start = 0;
        UINT64 pva = 0;

        /* test_xmem cleared the hooks after trying XMemSetAllocationHooks: the exported ones again. */
        void (WINAPI *set_hooks)(void *, void *) = (void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemSetAllocationHooks");
        if (set_hooks)
            set_hooks((void *)hook_alloc, (void *)hook_free);
        placement_size = (SIZE_T)64 * inc;
        if (placement_size <= sizeof(placement_memory)
                && SUCCEEDED(hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &phd,
                &IID_ID3D12DescriptorHeap, &pheap)))
            start = XSLOT(pheap, 9, SIZE_T)(pheap);
        CHECK(start == (SIZE_T)placement_memory && placement_asked == 64 * inc, "a descriptor heap placed by the title's "
              "allocation hook starts in its memory (%#Ix, the hook's %p; %Iu bytes asked, 64 x %u; hook calls %u, heaps %u, "
              "last attributes %#llx)", start, (void *)placement_memory, placement_asked, inc, xhook_calls, xhook_heap_calls,
              hook_last_attributes);
        if (start == (SIZE_T)placement_memory)
        {
            D3D12_HEAP_PROPERTIES up = { D3D12_HEAP_TYPE_UPLOAD };
            D3D12_RESOURCE_DESC cb = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
            ID3D12Resource *cbuf = NULL;
            D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = { 0, 256 };
            const UINT32 *words;
            if (SUCCEEDED(XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &up, D3D12_HEAP_FLAG_NONE, &cb,
                    D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void **)&cbuf)))
            {
                cbv.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(cbuf);
                XSLOT(dev, 17, void, const D3D12_CONSTANT_BUFFER_VIEW_DESC *, SIZE_T)(dev, &cbv, start + 3 * inc);
            }
            hr = shadow_heap ? shadow_heap(start, &pva, &pdata, &pstride, &pcount) : E_FAIL;
            words = pdata ? (const UINT32 *)((BYTE *)pdata + 3 * pstride) : NULL;
            CHECK(SUCCEEDED(hr) && words && (words[0] || words[1]), "a CBV written at a title handle in it reaches the "
                  "shadow heap (%#lx: %08x %08x)", hr, words ? words[0] : 0, words ? words[1] : 0);
            /* As the console's runtime writes one: 16-byte rows, an index-only bounds check. */
            CHECK(words && (words[1] >> 16 & 0x3fff) == 16 && words[2] == 256 / 16 && (words[3] >> 28 & 3) == 1,
                  "the CBV's V# has stride 16 and its size in rows (%08x %08x %08x)", words ? words[1] : 0,
                  words ? words[2] : 0, words ? words[3] : 0);
            if (cbuf) ID3D12Resource_Release(cbuf);
        }
        if (pheap) XSLOT(pheap, 2, ULONG)(pheap);
        /* A CPU-only heap placed so keeps its console descriptors in that memory: bytes the title
         * writes there itself are the shadow's. */
        {
            void *cheap = NULL;
            phd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            pdata = NULL;
            start = 0;
            if (SUCCEEDED(XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &phd,
                    &IID_ID3D12DescriptorHeap, &cheap)))
                start = XSLOT(cheap, 9, SIZE_T)(cheap);
            if (start)
                ((UINT32 *)(start + 5 * inc))[0] = 0x12345678;
            hr = shadow_heap && start ? shadow_heap(start, &pva, &pdata, &pstride, &pcount) : E_FAIL;
            CHECK(SUCCEEDED(hr) && pdata == (void *)start && ((UINT32 *)((BYTE *)pdata + 5 * pstride))[0] == 0x12345678,
                  "a CPU-only heap placed in title memory keeps its console descriptors there (%#lx, %p)", hr, pdata);
            if (cheap) XSLOT(cheap, 2, ULONG)(cheap);
        }
        placement_size = 0;
    }

    /* Newer-layout device methods a retail title uses (after xcl): 127 CreateCommittedResource with a
     * barrier layout, 108 GetResourceAllocationInfo2, 70 a shader resource view of memory. */
    {
        D3D12_HEAP_PROPERTIES dh = { D3D12_HEAP_TYPE_DEFAULT };
        D3D12_RESOURCE_DESC b1 = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 1 << 16, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
        BYTE descs[2][0x48] = { { 0 } };
        D3D12_RESOURCE_ALLOCATION_INFO1 infos[2];
        D3D12_RESOURCE_ALLOCATION_INFO all = { 0 };
        ID3D12Resource *committed = NULL;
        HRESULT (WINAPI *gpu_alloc)(SIZE_T, SIZE_T, void **, UINT64 *) =
                (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxGpuAlloc");

        hr = XSLOT(dev, 127, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *, UINT32,
                   const D3D12_CLEAR_VALUE *, void *, UINT32, const DXGI_FORMAT *, REFIID, void **)(dev, &dh,
                   D3D12_HEAP_FLAG_NONE, &b1, 0xffffffff, NULL, NULL, 0, NULL, &IID_ID3D12Resource, (void **)&committed);
        CHECK(SUCCEEDED(hr) && committed, "device slot 127 makes a committed resource from a barrier layout %#lx", hr);
        if (committed) ID3D12Resource_Release(committed);

        memcpy(descs[0], &b1, sizeof(b1));
        memcpy(descs[1], &b1, sizeof(b1));
        XSLOT(dev, 108, D3D12_RESOURCE_ALLOCATION_INFO *, D3D12_RESOURCE_ALLOCATION_INFO *, UINT, UINT, const void *,
              D3D12_RESOURCE_ALLOCATION_INFO1 *)(dev, &all, 0, 2, descs, infos);
        CHECK(all.SizeInBytes >= 2 << 16 && infos[1].Offset >= infos[0].SizeInBytes,
              "device slot 108 sums two buffers (%llu bytes, the second at %llu)", (unsigned long long)all.SizeInBytes,
              (unsigned long long)infos[1].Offset);

        if (gpu_alloc)
        {
            D3D12_DESCRIPTOR_HEAP_DESC vh = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
            HRESULT (WINAPI *shadow_heap)(SIZE_T, UINT64 *, void **, UINT *, UINT *) =
                    (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxShadowHeap");
            struct { D3D12_SHADER_RESOURCE_VIEW_DESC v; UINT64 address; BYTE pad[8]; } view = { { DXGI_FORMAT_R32_TYPELESS,
                    D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING } };
            void *mem = NULL, *vheap = NULL, *vdata = NULL;
            UINT64 va = 0, vva;
            UINT vstride, vcount;
            SIZE_T vstart;
            view.v.Buffer.NumElements = 16;
            view.v.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            if (SUCCEEDED(gpu_alloc(1 << 16, 1 << 16, &mem, &va)) && SUCCEEDED(XSLOT(dev, 14, HRESULT,
                    const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &vh, &IID_ID3D12DescriptorHeap, &vheap)))
            {
                view.address = va;
                vstart = XSLOT(vheap, 9, SIZE_T)(vheap);
                XSLOT(dev, 70, void, const D3D12_RESOURCE_DESC *, const void *, SIZE_T)(dev, &b1, &view, vstart);
                hr = shadow_heap ? shadow_heap(vstart, &vva, &vdata, &vstride, &vcount) : E_FAIL;
                CHECK(SUCCEEDED(hr) && vdata && (((UINT32 *)vdata)[0] || ((UINT32 *)vdata)[1]),
                      "device slot 70 writes a view of memory at %#llx into the heap (%#lx)", (unsigned long long)va, hr);
                XSLOT(vheap, 2, ULONG)(vheap);
            }
        }
    }

    /* Command allocators are proxies: the console's slot 9 takes an allocation callback (the target
     * title registers one), and a list made and reset with one records and closes. */
    {
        void *palloc = NULL, *plist = NULL;
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, &palloc);
        if (SUCCEEDED(hr))
            hr = XSLOT(palloc, 9, HRESULT, void *, void *)(palloc, (void *)GetProcAddress, NULL);
        if (SUCCEEDED(hr))
            hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, void *, void *, REFIID, void **)(dev, 0,
                       D3D12_COMMAND_LIST_TYPE_DIRECT, palloc, NULL, &IID_ID3D12GraphicsCommandList, &plist);
        if (SUCCEEDED(hr))
            hr = XSLOT(plist, 9, HRESULT)(plist);
        if (SUCCEEDED(hr))
            hr = XSLOT(palloc, 8, HRESULT)(palloc);
        if (SUCCEEDED(hr))
            hr = XSLOT(plist, 10, HRESULT, void *, void *)(plist, palloc, NULL);
        if (SUCCEEDED(hr))
            hr = XSLOT(plist, 9, HRESULT)(plist);
        CHECK(SUCCEEDED(hr), "a command allocator takes the console's slot 9 callback and serves a list's creation "
              "and Reset (%#lx)", hr);
        if (plist) XSLOT(plist, 2, ULONG)(plist);
        if (palloc) XSLOT(palloc, 2, ULONG)(palloc);
    }

    /* Device slot 133 (SDK 0x20004): CreatePlacedResourceX, as the target title places a buffer over
     * its physical pages, with state 0xffffffff. Past the old end of the device vtable (128). */
    {
        D3D12_RESOURCE_DESC bd133 = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 1 << 16, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                      D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
        void *mem = VirtualAlloc(NULL, 1 << 16, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        ID3D12Resource *placed = NULL;
        hr = XSLOT(dev, 133, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
                   const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, (UINT64)(ULONG_PTR)mem, &bd133,
                   (D3D12_RESOURCE_STATES)0xffffffff, NULL, &IID_ID3D12Resource, (void **)&placed);
        CHECK(SUCCEEDED(hr) && placed, "device slot 133 places a buffer (CreatePlacedResourceX, state 0xffffffff) %#lx", hr);
        if (placed) ID3D12Resource_Release(placed);
        if (mem) VirtualFree(mem, 0, MEM_RELEASE);
    }

    /* A texture's GPU VA is 0 under vkd3d-proton; titles write texels
     * through it, so d3d12_xs hands out scratch GPU memory instead. */
    {
        UINT64 (WINAPI *usage)(void) = (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxGpuBytesInUse");
        UINT64 va;

        scratch_before = usage ? usage() : 0;
        va = ID3D12Resource_GetGPUVirtualAddress(tex);
        CHECK(va != 0, "texture GPU VA is backed by scratch memory (%#llx)", (unsigned long long)va);
        if (va)
        {
            memset((void *)(ULONG_PTR)va, 0x11, 64 * 48 * 4);
            CHECK(ID3D12Resource_GetGPUVirtualAddress(tex) == va && (!usage || usage() > scratch_before),
                  "texel writes through it land, and the address is stable");
        }
    }

    /* CreateRenderTargetView (device slot 20), ClearRenderTargetView (list
     * slot 52 = PC 48), ResourceBarrier (slot 26) to PRESENT. */
    rtv = XSLOT(rtv_heap, 9, SIZE_T)(rtv_heap);
    XSLOT(dev, 20, void, ID3D12Resource *, const void *, SIZE_T)(dev, tex, NULL, rtv);
    XSLOT(list, 52, void, SIZE_T, const FLOAT *, UINT, const D3D12_RECT *)(list, rtv, clear.Color, 0, NULL);

    /* A graphics PSO from console bytes with no translation gets fallback
     * shaders, which must draw nothing: the frame stays the clear colour.
     * Skipped when fallbacks are off (the strict run). */
    if (!getenv("REBOX_SHADER_FALLBACK") || *getenv("REBOX_SHADER_FALLBACK") != '0')
    {
        static const char console_vs[] = "REBOX-FAKE-CONSOLE-VS", console_ps[] = "REBOX-FAKE-CONSOLE-PS";
        PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
                GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "D3D12SerializeRootSignature");
        D3D12_ROOT_SIGNATURE_DESC rsd = { 0, NULL, 0, NULL, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT };
        D3D12_GRAPHICS_PIPELINE_STATE_DESC gd = { 0 };
        D3D12_VIEWPORT vp = { 0, 0, 64, 48, 0, 1 };
        D3D12_RECT sc = { 0, 0, 64, 48 };
        ID3D10Blob *blob = NULL;

        if (serialize && SUCCEEDED(serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL)))
        {
            XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                    ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
            ID3D10Blob_Release(blob);
        }
        gd.pRootSignature = rs;
        gd.VS.pShaderBytecode = console_vs;
        gd.VS.BytecodeLength = sizeof(console_vs);
        gd.PS.pShaderBytecode = console_ps;
        gd.PS.BytecodeLength = sizeof(console_ps);
        gd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        gd.SampleMask = ~0u;
        gd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        gd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        gd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        gd.NumRenderTargets = 1;
        gd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        gd.SampleDesc.Count = 1;
        hr = rs ? XSLOT(dev, 10, HRESULT, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &gd,
                &IID_ID3D12PipelineState, &pso) : E_FAIL;
        CHECK(SUCCEEDED(hr), "graphics PSO from untranslated console shaders gets fallbacks (%#lx)", hr);
        {
            /* vkd3d-proton (patch 0018) asks during creation and makes no pipeline for it; the draw
             * below is left out rather than drawn with one that draws nothing. */
            BOOL (WINAPI *fallback_pso)(void) = (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"),
                    "ReboxFallbackPso");
            CHECK(fallback_pso && !fallback_pso(), "d3d12_xs exports ReboxFallbackPso, FALSE outside PSO creation");
        }
        if (pso)
        {
            /* RSSetViewports 21, RSSetScissorRects 22, IASetPrimitiveTopology
             * 20, SetGraphicsRootSignature 34, SetPipelineState 25,
             * OMSetRenderTargets 50, DrawInstanced 12. */
            XSLOT(list, 21, void, UINT, const D3D12_VIEWPORT *)(list, 1, &vp);
            XSLOT(list, 22, void, UINT, const D3D12_RECT *)(list, 1, &sc);
            /* 17: the console rectangle list, drawn as a triangle list. */
            XSLOT(list, 20, void, D3D12_PRIMITIVE_TOPOLOGY)(list, (D3D12_PRIMITIVE_TOPOLOGY)17);
            XSLOT(list, 34, void, void *)(list, rs);
            XSLOT(list, 25, void, void *)(list, pso);
            XSLOT(list, 50, void, UINT, const SIZE_T *, BOOL, const SIZE_T *)(list, 1, &rtv, FALSE, NULL);
            /* IASetIndexBuffer 47, IASetVertexBuffers 48 (PC 43, 44) over the title's own memory:
             * made GPU memory in place when bound, so vkd3d-proton knows the address. */
            {
                static BYTE *ia_memory;
                D3D12_VERTEX_BUFFER_VIEW vbv;
                D3D12_INDEX_BUFFER_VIEW ibv;
                if (!ia_memory)
                    ia_memory = VirtualAlloc(NULL, 1 << 21, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                vbv.BufferLocation = (UINT64)(ULONG_PTR)ia_memory;
                vbv.SizeInBytes = 1 << 16;
                vbv.StrideInBytes = 16;
                ibv.BufferLocation = (UINT64)(ULONG_PTR)ia_memory + (1 << 16);
                ibv.SizeInBytes = 1 << 12;
                ibv.Format = DXGI_FORMAT_R16_UINT;
                XSLOT(list, 48, void, UINT, UINT, const D3D12_VERTEX_BUFFER_VIEW *)(list, 0, 1, &vbv);
                XSLOT(list, 47, void, const D3D12_INDEX_BUFFER_VIEW *)(list, &ibv);
            }
            XSLOT(list, 12, void, UINT, UINT, UINT, UINT)(list, 3, 1, 0, 0);
        }
    }

    to_present.Transition.pResource = tex;
    to_present.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, &to_present);
    hr = XSLOT(list, 9, HRESULT)(list);
    CHECK(SUCCEEDED(hr), "clear recorded through Xbox slots (%#lx)", hr);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);

    plane_params[1] = 1;
    plane_params[2] = (UINT64)(ULONG_PTR)tex;
    hr = XSLOT(queue, 30, HRESULT, UINT, const void *, const void *)(queue, 1, plane_params, NULL);
    CHECK(hr == S_OK, "PresentX with the plane resource in the params (%#lx)", hr);
    plane_params[2] = (UINT64)(ULONG_PTR)&tex;
    hr = XSLOT(queue, 30, HRESULT, UINT, const void *, const void *)(queue, 1, plane_params, NULL);
    CHECK(hr == S_OK, "PresentX with a pointer to the plane resource (%#lx)", hr);

    if (!dump_dir)
    {
        printf("skip frame dump checks (REBOX_DUMP_FRAMES not set)\n");
        goto done;
    }
    /* Clear colour (1, 0.5, 0.25) is RGB 255/128/64; BMP stores BGRA. */
    for (int frame = 1; frame <= 2; frame++)
    {
        BOOL found = read_dump_pixel(dump_dir, frame, px);
        CHECK(found && px[2] == 255 && abs(px[1] - 128) <= 1 && abs(px[0] - 64) <= 1,
              "presented frame %d has the cleared pixels (RGB %u %u %u)", frame, found ? px[2] : 0,
              found ? px[1] : 0, found ? px[0] : 0);
    }

done:
    /* The draw must finish before its PSO and root signature go. */
    if (pso || rs)
    {
        ID3D12Fence *fence = NULL;
        if (SUCCEEDED(XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0,
                D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&fence)))
        {
            XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
            for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
                Sleep(10);
            ID3D12Fence_Release(fence);
        }
    }
    if (pso) XSLOT(pso, 2, ULONG)(pso);
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (tex)
    {
        UINT64 (WINAPI *usage)(void) = (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxGpuBytesInUse");
        ID3D12Resource_Release(tex);
        if (usage && scratch_before)
            CHECK(usage() == scratch_before, "texture scratch memory freed with the texture");
    }
    if (rtv_heap) XSLOT(rtv_heap, 2, ULONG)(rtv_heap);
}

/* The FidelityFX stub (amd_fidelityfx_loader_xbox.dll): FSR-style queries,
 * then an upscale dispatch through d3d12_xs's bilinear blit. A 32x24 green
 * render is scaled into a 64x48 output, which is presented as frame 3 and
 * read back from the frame dump. */
static void test_ffx(void *dev, void *queue)
{
    HMODULE ffx = LoadLibraryA("amd_fidelityfx_loader_xbox.dll");
    UINT32 (*create)(void **, void *, const void *) = ffx ? (void *)GetProcAddress(ffx, "ffxCreateContext") : NULL;
    UINT32 (*query)(void **, void *) = ffx ? (void *)GetProcAddress(ffx, "ffxQuery") : NULL;
    UINT32 (*dispatch)(void **, const void *) = ffx ? (void *)GetProcAddress(ffx, "ffxDispatch") : NULL;
    UINT32 (*destroy)(void **, const void *) = ffx ? (void *)GetProcAddress(ffx, "ffxDestroyContext") : NULL;
    const char *dump_dir = getenv("REBOX_DUMP_FRAMES");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    D3D12_HEAP_PROPERTIES heap_props = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC cd = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 32, 24, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET };
    D3D12_RESOURCE_DESC od = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 48, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    D3D12_CLEAR_VALUE green = { DXGI_FORMAT_R8G8B8A8_UNORM, { { 0.0f, 1.0f, 0.0f, 1.0f } } };
    D3D12_RESOURCE_BARRIER b[2] = { { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION }, { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION } };
    struct { UINT64 type; void *next; void *device; } backend = { 0x1000002, NULL, dev };
    BYTE create_desc[0x30] = { 0 }, query_desc[0x30] = { 0 }, dispatch_desc[0x1b0] = { 0 };
    ID3D12Resource *color = NULL, *output = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *rtv_heap = NULL, *list = NULL, *ctx = NULL;
    UINT64 plane_params[6] = { 0 };
    UINT32 rw = 0, rh = 0, rc;
    SIZE_T rtv;
    BYTE px[4];
    HRESULT hr;

    CHECK(create && query && dispatch && destroy, "FidelityFX stub exports the FFX API");
    if (!create || !query || !dispatch || !destroy) return;

    /* GETRENDERRESOLUTIONFROMQUALITYMODE: 64x48 at PERFORMANCE (2x) -> 32x24. */
    *(UINT64 *)query_desc = 0x10004;
    *(UINT32 *)(query_desc + 0x10) = 64;
    *(UINT32 *)(query_desc + 0x14) = 48;
    *(UINT32 *)(query_desc + 0x18) = 3;
    *(UINT32 **)(query_desc + 0x20) = &rw;
    *(UINT32 **)(query_desc + 0x28) = &rh;
    rc = query(NULL, query_desc);
    CHECK(rc == 0 && rw == 32 && rh == 24, "FFX render resolution query (%u: %ux%u)", rc, rw, rh);

    *(UINT64 *)create_desc = 0x10000;
    *(void **)(create_desc + 8) = &backend;
    *(UINT32 *)(create_desc + 0x14) = 32;
    *(UINT32 *)(create_desc + 0x18) = 24;
    *(UINT32 *)(create_desc + 0x1c) = 64;
    *(UINT32 *)(create_desc + 0x20) = 48;
    rc = create(&ctx, create_desc, NULL);
    CHECK(rc == 0 && ctx, "FFX upscale context (%u)", rc);
    if (rc) return;

    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &rtv_heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &heap_props,
                D3D12_HEAP_FLAG_NONE, &cd, D3D12_RESOURCE_STATE_RENDER_TARGET, &green, &IID_ID3D12Resource,
                (void **)&color);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &heap_props,
                D3D12_HEAP_FLAG_NONE, &od, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource,
                (void **)&output);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList,
                &list);
    CHECK(SUCCEEDED(hr), "resources for the FFX dispatch (%#lx)", hr);
    if (FAILED(hr)) goto done;

    rtv = XSLOT(rtv_heap, 9, SIZE_T)(rtv_heap);
    XSLOT(dev, 20, void, ID3D12Resource *, const void *, SIZE_T)(dev, color, NULL, rtv);
    XSLOT(list, 52, void, SIZE_T, const FLOAT *, UINT, const D3D12_RECT *)(list, rtv, green.Color, 0, NULL);
    b[0].Transition.pResource = color;
    b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, b);

    *(UINT64 *)dispatch_desc = 0x10001;
    *(void **)(dispatch_desc + 0x10) = list;
    *(void **)(dispatch_desc + 0x18) = color;
    *(void **)(dispatch_desc + 0x138) = output;
    *(UINT32 *)(dispatch_desc + 0x178) = 32;
    *(UINT32 *)(dispatch_desc + 0x17c) = 24;
    *(UINT32 *)(dispatch_desc + 0x180) = 64;
    *(UINT32 *)(dispatch_desc + 0x184) = 48;
    rc = dispatch(&ctx, dispatch_desc);
    CHECK(rc == 0, "FFX upscale dispatch (%u)", rc);

    b[0].Transition.pResource = output;
    b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, b);
    hr = XSLOT(list, 9, HRESULT)(list);
    CHECK(SUCCEEDED(hr), "list with the FFX dispatch closes (%#lx)", hr);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    plane_params[1] = 1;
    plane_params[2] = (UINT64)(ULONG_PTR)output;
    XSLOT(queue, 30, HRESULT, UINT, const void *, const void *)(queue, 1, plane_params, NULL);
    if (dump_dir)
    {
        BOOL found = read_dump_pixel(dump_dir, 3, px);
        CHECK(found && px[2] == 0 && px[1] == 255 && px[0] == 0, "upscaled frame is the render's green (RGB %u %u %u)",
              found ? px[2] : 0, found ? px[1] : 0, found ? px[0] : 0);
    }

done:
    if (SUCCEEDED(XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
            &IID_ID3D12Fence, (void **)&fence)))
    {
        XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
        for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
            Sleep(10);
        ID3D12Fence_Release(fence);
    }
    destroy(&ctx, NULL);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (color) ID3D12Resource_Release(color);
    if (output) ID3D12Resource_Release(output);
    if (rtv_heap) XSLOT(rtv_heap, 2, ULONG)(rtv_heap);
}

/* A plane in a format no swapchain takes (R11G11B10_FLOAT) is converted to
 * RGBA8 by PresentX; presented as frame 4. */
static void test_present_format(void *dev, void *queue)
{
    const char *dump_dir = getenv("REBOX_DUMP_FRAMES");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    D3D12_HEAP_PROPERTIES heap_props = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 48, 1, 1, DXGI_FORMAT_R11G11B10_FLOAT,
                                 { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET };
    D3D12_CLEAR_VALUE clear = { DXGI_FORMAT_R11G11B10_FLOAT, { { 1.0f, 0.5f, 0.25f, 1.0f } } };
    D3D12_RESOURCE_BARRIER b = { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Resource *tex = NULL;
    ID3D12Fence *fence = NULL;
    void *rtv_heap = NULL, *list = NULL;
    UINT64 plane_params[6] = { 0 };
    SIZE_T rtv;
    BYTE px[4];
    HRESULT hr;

    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &rtv_heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &heap_props,
                D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear, &IID_ID3D12Resource,
                (void **)&tex);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList,
                &list);
    CHECK(SUCCEEDED(hr), "R11G11B10 plane for PresentX (%#lx)", hr);
    if (FAILED(hr)) goto done;

    rtv = XSLOT(rtv_heap, 9, SIZE_T)(rtv_heap);
    XSLOT(dev, 20, void, ID3D12Resource *, const void *, SIZE_T)(dev, tex, NULL, rtv);
    XSLOT(list, 52, void, SIZE_T, const FLOAT *, UINT, const D3D12_RECT *)(list, rtv, clear.Color, 0, NULL);
    b.Transition.pResource = tex;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    XSLOT(list, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list, 1, &b);
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    plane_params[1] = 1;
    plane_params[2] = (UINT64)(ULONG_PTR)tex;
    hr = XSLOT(queue, 30, HRESULT, UINT, const void *, const void *)(queue, 1, plane_params, NULL);
    CHECK(hr == S_OK, "PresentX of an R11G11B10 plane (%#lx)", hr);
    if (dump_dir)
    {
        BOOL found = read_dump_pixel(dump_dir, 4, px);
        CHECK(found && px[2] == 255 && abs(px[1] - 128) <= 1 && abs(px[0] - 64) <= 1,
              "R11G11B10 plane presented as RGBA8 (RGB %u %u %u)", found ? px[2] : 0, found ? px[1] : 0,
              found ? px[0] : 0);
    }

done:
    if (SUCCEEDED(XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
            &IID_ID3D12Fence, (void **)&fence)))
    {
        XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
        for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
            Sleep(10);
        ID3D12Fence_Release(fence);
    }
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (tex) ID3D12Resource_Release(tex);
    if (rtv_heap) XSLOT(rtv_heap, 2, ULONG)(rtv_heap);
}

/* Frame pacing through device slots 78-80, as the traced retail title uses
 * them: SetFrameIntervalX, then WaitFrameEventX(ORIGIN, INFINITE) per frame. */
static double ms_since(LARGE_INTEGER a)
{
    LARGE_INTEGER b, f;
    QueryPerformanceCounter(&b);
    QueryPerformanceFrequency(&f);
    return (b.QuadPart - a.QuadPart) * 1000.0 / f.QuadPart;
}

/* Xbox-only root signature flags (0x01000000, 0x02000000, 0x80000000; Leonida found the PC
 * deserializer rejects them): serializing, creating and deserializing must all work. */
/* Console indirect argument types: Unreal 5 (a UE5 sample) makes its dispatch-indirect
 * signature with type 105, a dispatch of three group counts on the console. */
static void test_console_command_signature(void *dev)
{
    D3D12_INDIRECT_ARGUMENT_DESC arg = { (D3D12_INDIRECT_ARGUMENT_TYPE)105 };
    D3D12_COMMAND_SIGNATURE_DESC desc = { sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &arg, 0 };
    void *signature = NULL;
    HRESULT hr = XSLOT(dev, 41, HRESULT, const D3D12_COMMAND_SIGNATURE_DESC *, void *, REFIID, void **)(dev, &desc, NULL,
            &IID_ID3D12CommandSignature, &signature);
    CHECK(SUCCEEDED(hr) && signature, "CreateCommandSignature takes the console's dispatch argument type 105 (%#lx)",
          (unsigned long)hr);
    CHECK(arg.Type == 105, "and leaves the title's description as it was");
    if (signature)
        IUnknown_Release((IUnknown *)signature);
    /* A root signature with arguments that don't change root arguments (the console takes it). */
    {
        PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
                GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "D3D12SerializeRootSignature");
        D3D12_ROOT_SIGNATURE_DESC rsd = { 0 };
        ID3D10Blob *blob = NULL;
        void *rs = NULL;
        if (serialize && SUCCEEDED(serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL)))
        {
            XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0, ID3D10Blob_GetBufferPointer(blob),
                    ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
            ID3D10Blob_Release(blob);
        }
        signature = NULL;
        hr = rs ? XSLOT(dev, 41, HRESULT, const D3D12_COMMAND_SIGNATURE_DESC *, void *, REFIID, void **)(dev, &desc, rs,
                &IID_ID3D12CommandSignature, &signature) : E_FAIL;
        CHECK(SUCCEEDED(hr) && signature, "CreateCommandSignature takes a root signature the arguments don't need (%#lx)",
              (unsigned long)hr);
        if (signature)
            IUnknown_Release((IUnknown *)signature);
        if (rs)
            IUnknown_Release((IUnknown *)rs);
    }
}

/* Device slot 115, DeserializeStateObjectX: a placeholder ray tracing state object for a console
 * blob (Unreal 5 stops if it fails), with its properties, zero identifiers and the device. */
static void test_state_object_placeholder(void *dev)
{
    static const UINT blob[8] = { 2 };
    void *so = NULL, *props = NULL, *device = NULL;
    const BYTE *id;
    BOOL zero = TRUE;
    HRESULT hr = XSLOT(dev, 115, HRESULT, const void *, SIZE_T, void *, UINT, REFIID, void **)(dev, blob, sizeof(blob),
            NULL, 0, &IID_ID3D12StateObject, &so);

    CHECK(SUCCEEDED(hr) && so, "DeserializeStateObjectX gives a state object (%#lx)", (unsigned long)hr);
    if (!so) return;
    CHECK(SUCCEEDED(XSLOT(so, 0, HRESULT, REFIID, void **)(so, &IID_ID3D12StateObjectProperties, &props)) && props,
          "the state object has properties");
    if (props)
    {
        id = XSLOT(props, 3, const BYTE *, const WCHAR *)(props, L"RayGen");
        for (int i = 0; id && i < D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES; i++)
            zero &= !id[i];
        CHECK(id && zero && XSLOT(props, 4, UINT64, const WCHAR *)(props, L"RayGen") == 0,
              "its shader identifiers are zero (no shader), stack sizes 0");
        {
            /* Unreal's inspection of the native programs: none here. */
            void *info0[2] = { (void *)1, (void *)1 }, *info1[2] = { NULL, NULL }, *programs = NULL, *table = NULL;
            const BYTE *count1 = NULL, *count2 = NULL;
            XSLOT(props, 7, HRESULT, UINT, void *)(props, 0, info0);
            XSLOT(props, 7, HRESULT, UINT, void *)(props, 1, info1);
            if (info1[0])
                count1 = XSLOT(info1[0], 3, const BYTE *)(info1[0]);
            XSLOT(so, 8, HRESULT, void **)(so, &programs);
            if (programs)
                XSLOT(programs, 9, HRESULT, UINT64, UINT64, void **)(programs, 1, 0, &table);
            if (table)
                count2 = XSLOT(table, 3, const BYTE *)(table);
            CHECK(!info0[0] && !info0[1] && count1 && !*(const UINT *)(count1 + 8) && count2
                      && !*(const UINT *)(count2 + 0x10),
                  "the console's inspection slots (properties 7, state object 8) describe no programs");
        }
        XSLOT(props, 2, ULONG)(props);
    }
    CHECK(SUCCEEDED(XSLOT(so, 7, HRESULT, REFIID, void **)(so, &IID_IUnknown, &device)) && device,
          "GetDevice gives the device");
    if (device)
        IUnknown_Release((IUnknown *)device);
    CHECK(XSLOT(so, 2, ULONG)(so) == 0, "the state object is freed on its last release");
}

/* Device slot 86: ID3D12Device2::CreatePipelineState from a stream (Unreal 5's Xbox RHI). */
static void test_stream_pso(void *dev)
{
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "D3D12SerializeRootSignature");
    D3D12_ROOT_SIGNATURE_DESC rsd = { 0 };
    ID3D10Blob *blob = NULL;
    void *rs = NULL, *pso = NULL;
    struct
    {
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; void *rs; } __attribute__((aligned(8))) root;
        struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_SHADER_BYTECODE cs; } __attribute__((aligned(8))) cs;
    } stream = { { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, NULL },
                 { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, { test_empty_cs, sizeof(test_empty_cs) } } };
    D3D12_PIPELINE_STATE_STREAM_DESC desc = { sizeof(stream), &stream };
    HRESULT hr;

    if (serialize && SUCCEEDED(serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL)))
    {
        XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0, ID3D10Blob_GetBufferPointer(blob),
                ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
        ID3D10Blob_Release(blob);
    }
    stream.root.rs = rs;
    hr = rs ? XSLOT(dev, 86, HRESULT, const D3D12_PIPELINE_STATE_STREAM_DESC *, REFIID, void **)(dev, &desc,
            &IID_ID3D12PipelineState, &pso) : E_FAIL;
    CHECK(SUCCEEDED(hr) && pso, "CreatePipelineState from a stream (device slot 86) makes a compute pipeline (%#lx)",
          (unsigned long)hr);
    if (pso)
        IUnknown_Release((IUnknown *)pso);
    /* Ending with the console's type 26 (24 bytes: four dwords, 0xffff, a pointer), as The Matrix
     * Awakens' streams do: dropped, not refused. */
    {
        struct
        {
            struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; void *rs; } __attribute__((aligned(8))) root;
            struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; D3D12_SHADER_BYTECODE cs; } __attribute__((aligned(8))) cs;
            struct { UINT type; UINT pad; UINT words[4]; void *pointer; } __attribute__((aligned(8))) console;
        } console_stream = { { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, rs },
                             { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, { test_empty_cs, sizeof(test_empty_cs) } },
                             { 26, 0, { 0, 0, 0, 0xffff }, &console_stream } };
        D3D12_PIPELINE_STATE_STREAM_DESC console_desc = { sizeof(console_stream), &console_stream };
        pso = NULL;
        hr = rs ? XSLOT(dev, 86, HRESULT, const D3D12_PIPELINE_STATE_STREAM_DESC *, REFIID, void **)(dev, &console_desc,
                &IID_ID3D12PipelineState, &pso) : E_FAIL;
        CHECK(SUCCEEDED(hr) && pso, "a stream ending with the console's subobject type 26 makes its pipeline (%#lx)",
              (unsigned long)hr);
        if (pso)
            IUnknown_Release((IUnknown *)pso);
    }
    if (rs)
        IUnknown_Release((IUnknown *)rs);
}

/* Device slot 102: GetRaytracingAccelerationStructurePrebuildInfo (PC 63), forwarded: the sizes
 * come back written (Unreal's ray tracing used them unset and crashed). */
static void test_prebuild_info(void *dev)
{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE, 1,
        D3D12_ELEMENTS_LAYOUT_ARRAY };
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
    memset(&info, 0xcc, sizeof(info));
    XSLOT(dev, 102, void, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS *,
          D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO *)(dev, &inputs, &info);
    CHECK(info.ResultDataMaxSizeInBytes != 0xccccccccccccccccull && info.ScratchDataSizeInBytes != 0xccccccccccccccccull,
          "GetRaytracingAccelerationStructurePrebuildInfo (device slot 102) answers (%llu, %llu bytes)",
          (unsigned long long)info.ResultDataMaxSizeInBytes, (unsigned long long)info.ScratchDataSizeInBytes);
}

/* DEFAULT buffer heaps are CPU-visible custom heaps here; the title still sees DEFAULT (Unreal 5
 * takes a CUSTOM heap's path that's broken on the console). */
static void test_title_heap_properties(void *dev)
{
    D3D12_HEAP_PROPERTIES props = { D3D12_HEAP_TYPE_DEFAULT }, got;
    D3D12_RESOURCE_DESC buffer = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 4096, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
    D3D12_HEAP_DESC heap_desc = { 64 << 10, { D3D12_HEAP_TYPE_DEFAULT }, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS }, desc;
    D3D12_HEAP_FLAGS flags;
    void *resource = NULL, *heap = NULL;

    XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
          D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &props, D3D12_HEAP_FLAG_NONE, &buffer,
          D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, &resource);
    memset(&got, 0xcc, sizeof(got));
    CHECK(resource && SUCCEEDED(XSLOT(resource, 14, HRESULT, D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS *)(resource, &got, &flags))
              && got.Type == D3D12_HEAP_TYPE_DEFAULT,
          "a committed DEFAULT buffer reports a DEFAULT heap (type %u)", got.Type);
    if (resource)
        IUnknown_Release((IUnknown *)resource);
    XSLOT(dev, 28, HRESULT, const D3D12_HEAP_DESC *, REFIID, void **)(dev, &heap_desc, &IID_ID3D12Heap, &heap);
    memset(&desc, 0xcc, sizeof(desc));
    if (heap)
        XSLOT(heap, 8, D3D12_HEAP_DESC *, D3D12_HEAP_DESC *)(heap, &desc);
    CHECK(heap && desc.Properties.Type == D3D12_HEAP_TYPE_DEFAULT && desc.SizeInBytes == (64 << 10),
          "a DEFAULT buffer heap's GetDesc says DEFAULT (type %u)", desc.Properties.Type);
    if (heap)
        IUnknown_Release((IUnknown *)heap);
}

/* A console depth texture that's also a UAV (Unreal 5's, flags 0x1000006): made, as a depth target. */
static void test_depth_uav_texture(void *dev)
{
    D3D12_HEAP_PROPERTIES props = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC tex = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R32G8X24_TYPELESS, { 1, 0 },
                                D3D12_TEXTURE_LAYOUT_UNKNOWN, (D3D12_RESOURCE_FLAGS)0x1000006 };
    D3D12_CLEAR_VALUE clear = { DXGI_FORMAT_D32_FLOAT_S8X24_UINT, .DepthStencil = { 1.0f, 0 } };
    void *resource = NULL;
    HRESULT hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
            D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &props, D3D12_HEAP_FLAG_NONE, &tex,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, &IID_ID3D12Resource, &resource);
    CHECK(SUCCEEDED(hr) && resource, "a depth texture with unordered access (console flags 0x1000006) is made (%#lx)",
          (unsigned long)hr);
    if (resource)
        IUnknown_Release((IUnknown *)resource);
}

/* The console numbers mesh stages one lower (23 AS, 24 MS) and has a 26 of its own (a UE5
 * sample's Nanite streams): the desktop gets 24 and 25, and no 26. */
static void test_console_mesh_stream(void)
{
    UINT (WINAPI *types_of)(const void *, SIZE_T, UINT *, UINT) = (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"),
            "ReboxDesktopStreamTypes");
    static const char ms_bytes[] = "MS", ps_bytes[] = "PS";
    struct
    {
        struct { UINT type; void *rs; } __attribute__((aligned(8))) root;
        struct { UINT type; D3D12_SHADER_BYTECODE bc; } __attribute__((aligned(8))) as, ms, ps;
        struct { UINT type; UINT pad; UINT words[4]; UINT64 value; } __attribute__((aligned(8))) console;
    } stream = { { 0, NULL }, { 23, { NULL, 0 } }, { 24, { ms_bytes, sizeof(ms_bytes) } }, { 2, { ps_bytes, sizeof(ps_bytes) } },
                 { 26, 0, { 0, 0, 0, 0xffff }, 1 } };
    UINT types[8] = { 0 }, n;

    CHECK(types_of != NULL, "d3d12_xs exports ReboxDesktopStreamTypes");
    if (!types_of) return;
    n = types_of(&stream, sizeof(stream), types, ARRAYSIZE(types));
    CHECK(n == 4 && types[0] == 0 && types[1] == 24 && types[2] == 25 && types[3] == 2,
          "a console mesh stream reaches the desktop as AS 24, MS 25, without its 26 (%u types: %u %u %u %u)", n, types[0],
          types[1], types[2], types[3]);
    /* The target title's: a 24 (its mesh shader) and a pixel shader, no 23; and a pipeline of
     * rectangles (topology type 0x13), which becomes a triangle one. */
    {
        SIZE_T (WINAPI *desktop)(const void *, SIZE_T, void *, SIZE_T) = (void *)GetProcAddress(
                GetModuleHandleA("d3d12_xs.dll"), "ReboxDesktopStream");
        struct
        {
            struct { UINT type; void *rs; } __attribute__((aligned(8))) root;
            struct { UINT type; D3D12_SHADER_BYTECODE bc; } __attribute__((aligned(8))) ms, ps;
            struct { UINT type; UINT value; } __attribute__((aligned(8))) topology;
        } mesh = { { 0, NULL }, { 24, { ms_bytes, sizeof(ms_bytes) } }, { 2, { ps_bytes, sizeof(ps_bytes) } }, { 14, 0x13 } },
          out = { { 0 } };
        n = types_of(&mesh, sizeof(mesh), types, ARRAYSIZE(types));
        CHECK(n == 4 && types[1] == 25 && types[2] == 2,
              "a console mesh stream without a 23 reaches the desktop with its 24 as MS 25 (%u types: %u %u %u)", n,
              types[0], types[1], types[2]);
        CHECK(desktop && desktop(&mesh, sizeof(mesh), &out, sizeof(out)) == sizeof(mesh) && out.topology.type == 14
              && out.topology.value == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
              "a stream's rectangle topology type 0x13 reaches the desktop as TRIANGLE (%u)", out.topology.value);
    }
}

/* Guest buffer table entries ({address, length in dwords}), which translated shaders read with a
 * bounds check on the length only: a V# with records but no base (or a base in the first 64 KiB)
 * must be empty, or every load through it reads address 0 (the GPU page fault of the first
 * indirect mesh draws). */
static void test_guest_buffer_entry(void)
{
    void (WINAPI *entry_of)(const UINT32 *, UINT32 *) =
            (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxGuestBufferEntry");
    static const UINT32 no_base[4] = { 0, 16u << 16, 64, 0x1104dfac };
    static const UINT32 first_page[4] = { 0x40, 0, 256, 0x1104dfac };
    static const UINT32 empty[4] = { 0 };
    static const UINT32 raw[4] = { 0x1000, 0x2000, 256, 0x1104dfac };
    UINT32 e[4];

    CHECK(entry_of != NULL, "d3d12_xs exports ReboxGuestBufferEntry");
    if (!entry_of)
        return;
    memset(e, 0xff, sizeof(e));
    entry_of(no_base, e);
    CHECK(e[0] == 0 && e[1] == 0 && e[2] == 0, "a V# with 64 records of 16 bytes and no base is empty (%u dwords)", e[2]);
    entry_of(first_page, e);
    CHECK(e[2] == 0, "a V# whose base is in the first page is empty (%u dwords)", e[2]);
    entry_of(empty, e);
    CHECK(e[2] == 0 && e[3] == 0, "an all-zero V# is empty (%u dwords)", e[2]);
    /* 0x200000001000: in the GPU memory window, but in no buffer the title knows (none made a view
     * of it or asked its address) and no Rebox GPU memory: read as empty, its address kept. */
    entry_of(raw, e);
    CHECK(e[0] == 0x1000 && e[1] == 0x2000 && e[2] == 0,
          "a raw V# of 256 bytes at %#x%08x, in no known buffer, is empty (%u dwords)", e[1], e[0], e[2]);
}

/* A console mesh shader in its container, as the target title's FSSMs have it (console_fssm.c: the
 * slot table at 0x10, the mesh record and stage data, a 24-byte shader ID before the code), made a
 * deferred mesh stage and compiled through ReboxConsoleCompile as at a mesh dispatch (patch 0013):
 * AnyPS5's mesh path, then sa5_finalize_draw with the draw block's address at push byte 96. No
 * pipeline is made (lavapipe has no mesh shaders here). The program: three vertices and one
 * triangle, lane n at (n, 0, 0, 1) with param0 = 1. */
/* The target title's FSSM layout around a console mesh program (see test_console_mesh_compile). */
static BYTE *mesh_container(const UINT32 *code, UINT code_size, SIZE_T *size)
{
    /* Payload at 0x10: the slot count, 12 record offsets, the mesh record (0x54 bytes) at 0x68, its
     * stage data to the info at 0x448, then the shader ID and code at 0x45c. */
    const UINT payload = 0x10, record = 0x68, data = record + 0x54, code_at = data + 0x45c;
    UINT fssm_size = payload + code_at + 24 + code_size;
    BYTE *fssm = calloc(1, fssm_size), *container;

    if (!fssm)
        return NULL;
    memcpy(fssm, "FSSM", 4);
    *(UINT32 *)(fssm + 4) = 0x101;
    *(UINT64 *)(fssm + 8) = fssm_size - payload;
    *(UINT64 *)(fssm + payload) = 6;
    for (UINT at = 0x08; at < 0x68; at += 8)
        *(UINT64 *)(fssm + payload + at) = at == 0x48 ? record : 0xdeadbeefdeadbeefull;
    *(UINT64 *)(fssm + payload + record) = code_at;
    *(UINT64 *)(fssm + payload + record + 8) = 24 + code_size;
    *(UINT32 *)(fssm + payload + record + 0x14) = 17u << 16;        /* wave64, 17 user SGPRs */
    *(UINT32 *)(fssm + payload + record + 0x44) = 0xc0;             /* float mode */
    fssm[payload + data + 0xc8 + 1] = 1;                            /* param0 writes OSG1 register 1 */
    *(UINT32 *)(fssm + payload + data + 0x188) = 1;
    *(UINT16 *)(fssm + payload + data + 0x448) = 128;               /* lanes */
    *(UINT16 *)(fssm + payload + data + 0x44a) = 128;               /* vertices */
    *(UINT16 *)(fssm + payload + data + 0x44c) = 128;               /* primitives */
    fssm[payload + data + 0x44e] = 17;
    fssm[payload + data + 0x44f] = 2;                               /* triangles */
    memset(fssm + payload + code_at, 0x5a, 24);                     /* the shader ID */
    memcpy(fssm + payload + code_at + 24, code, code_size);
    container = console_container_around(fssm, fssm_size, NULL, NULL, 0, size);
    return container;
}

static void test_console_mesh_compile(void)
{
    static const UINT32 code[] = {
        0xbfa00003,                         /* s_inst_prefetch 3, as the title's open */
        0xd7650000, 0x000100c1,             /* v_mbcnt_lo_u32_b32 v0, -1, 0 */
        0xd7660000, 0x000200c1,             /* v_mbcnt_hi_u32_b32 v0, -1, v0 */
        0xb07c1003,                         /* s_movk_i32 m0, 0x1003: 3 vertices, 1 primitive */
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
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    void *(WINAPI *from_container)(const void *, SIZE_T, UINT32, SIZE_T *) = (void *)GetProcAddress(m,
            "ReboxConsoleGraphicsFromContainer");
    HRESULT (WINAPI *compile)(const struct rebox_compile_in *, struct rebox_compile_out *) = (void *)GetProcAddress(m,
            "ReboxConsoleCompile");
    void (WINAPI *release)(void *) = (void *)GetProcAddress(m, "ReboxConsoleFree");
    BYTE *container;
    const struct rebox_deferred_shader *h;
    const struct rebox_deferred_graphics *g;
    struct rebox_compile_in in;
    struct rebox_compile_out out;
    UINT32 sgprs[32] = { 0 };
    SIZE_T size, blob_size = 0;
    HRESULT hr;

    CHECK(from_container && compile && release, "d3d12_xs exports ReboxConsoleGraphicsFromContainer");
    if (!from_container || !compile || !release || !(container = mesh_container(code, sizeof(code), &size)))
        return;

    h = from_container(container, size, 0x80, &blob_size);
    g = h ? (const struct rebox_deferred_graphics *)(h + 1) : NULL;
    CHECK(h && h->version == 2 && h->stage == 0x80 && h->wave_size == 64 && h->code_dwords == ARRAYSIZE(code)
          && !memcmp((const BYTE *)h + h->code_offset, code, sizeof(code)) && g->mesh.threads == 128
          && g->mesh.group_stride == 128 && g->mesh.primitive_vertices == 3 && g->register_count == 1
          && g->registers[0] == 1, "a console mesh container becomes a deferred mesh stage (%u dwords of code, "
          "stride %u)", h ? h->code_dwords : 0, g ? g->mesh.group_stride : 0);
    if (h)
    {
        memset(&in, 0, sizeof(in));
        memset(&out, 0, sizeof(out));
        memset(in.bindless, 0xff, sizeof(in.bindless));
        in.shader = h;
        in.shader_size = (UINT32)blob_size;
        in.user_sgpr_count = 32;
        in.user_sgprs = sgprs;
        hr = compile(&in, &out);
        CHECK(SUCCEEDED(hr) && out.spirv_words && out.subgroup_size == 64 && (out.push[24] | out.push[25]),
              "a deferred mesh stage compiles through AnyPS5's mesh path and finalizes with its draw block at "
              "push byte 96 (%#lx, %u SPIR-V words, subgroup %u)", hr, out.spirv_words, out.subgroup_size);
        release((void *)h);
    }
    free(container);

    /* A mesh stage's root parameters are from s16 (8 system SGPRs, then user data as a vertex stage's):
     * the same program, its param from a V# in the table its first root parameter (s8 of the call)
     * points at, read through s[16:17]. With them at s8, s16 held 0 (or the sampler heap's address)
     * and the table walk read memory no region holds. */
    {
        static const UINT32 table_code[] = {
            0xbfa00003,                         /* s_inst_prefetch 3 */
            0xf4080508, 0xfa000000,             /* s_load_dwordx4 s[20:23], s[16:17], 0x0 */
            0xbf8cc07f,                         /* s_waitcnt lgkmcnt(0) */
            0xf420060a, 0xfa000000,             /* s_buffer_load_dword s24, s[20:23], 0x0 */
            0xbf8cc07f,                         /* s_waitcnt lgkmcnt(0) */
            0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xb07c1003, 0xbf900009, 0x7d880083, 0xbefe046a,
            0x7e020d00, 0x7e040280,
            0x7e060218,                         /* v_mov_b32 v3, s24 */
            0xf80008cf, 0x03020201, 0xf800020f, 0x03030303, 0x7d840080, 0xbefe046a, 0x7e0802ff, 0x00200400,
            0xf8000941, 0x00000004, 0xbf810000,
        };
        HRESULT (WINAPI *galloc)(SIZE_T, SIZE_T, void **, UINT64 *) = (void *)GetProcAddress(m, "ReboxGpuAlloc");
        void (WINAPI *gfree)(void *) = (void *)GetProcAddress(m, "ReboxGpuFree");
        UINT32 *mem = NULL;
        UINT64 mem_gpu = 0;

        if (!galloc || FAILED(galloc(256, 256, (void **)&mem, &mem_gpu)) || !(container = mesh_container(table_code,
                sizeof(table_code), &size)))
        {
            CHECK(0, "GPU memory and a container for the mesh table test");
            return;
        }
        /* The table: a V# of the 16 bytes of constants after it. */
        mem[0] = (UINT32)(mem_gpu + 0x80);
        mem[1] = (UINT32)((mem_gpu + 0x80) >> 32) & 0xffff;
        mem[2] = 16;
        mem[3] = (77u << 12) | 0xfac;
        mem[0x20] = 0x3f800000;
        if ((h = from_container(container, size, 0x80, &blob_size)))
        {
            memset(&in, 0, sizeof(in));
            memset(&out, 0, sizeof(out));
            memset(in.bindless, 0xff, sizeof(in.bindless));
            memset(sgprs, 0, sizeof(sgprs));
            sgprs[8] = (UINT32)mem_gpu;
            sgprs[9] = (UINT32)(mem_gpu >> 32);
            in.shader = h;
            in.shader_size = (UINT32)blob_size;
            in.user_sgpr_count = 32;
            in.user_sgprs = sgprs;
            hr = compile(&in, &out);
            CHECK(SUCCEEDED(hr) && out.spirv_words, "a mesh stage reads its first root parameter's table through "
                  "s[16:17] (%#lx, %u SPIR-V words)", hr, out.spirv_words);
            release((void *)h);
        }
        free(container);
        if (gfree)
            gfree(mem);
    }
}

static void test_xbox_root_flags(void *dev)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(m, "D3D12SerializeRootSignature");
    PFN_D3D12_CREATE_ROOT_SIGNATURE_DESERIALIZER deserialize = (PFN_D3D12_CREATE_ROOT_SIGNATURE_DESERIALIZER)(void *)
            GetProcAddress(m, "D3D12CreateRootSignatureDeserializer");
    D3D12_ROOT_PARAMETER param = { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, .Constants = { 0, 0, 1 } };
    D3D12_ROOT_SIGNATURE_DESC rsd = { 1, &param, 0, NULL, 0x83000000 };
    ID3D12RootSignatureDeserializer *des = NULL;
    ID3D10Blob *blob = NULL, *err = NULL;
    void *rs = NULL;
    HRESULT hr;

    hr = serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
    CHECK(SUCCEEDED(hr), "D3D12SerializeRootSignature takes Xbox-only flags 0x83000000 (%#lx)", hr);
    if (FAILED(hr))
        goto done;
    hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0, ID3D10Blob_GetBufferPointer(blob),
            ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
    CHECK(SUCCEEDED(hr), "CreateRootSignature (device slot 16) takes a blob with Xbox-only flags (%#lx)", hr);
    hr = deserialize(ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob),
            &IID_ID3D12RootSignatureDeserializer, (void **)&des);
    CHECK(SUCCEEDED(hr), "D3D12CreateRootSignatureDeserializer takes a blob with Xbox-only flags (%#lx)", hr);
    if (des)
    {
        const D3D12_ROOT_SIGNATURE_DESC *d = ID3D12RootSignatureDeserializer_GetRootSignatureDesc(des);
        printf("info deserialized root signature flags %#x\n", d ? d->Flags : 0);
        ID3D12RootSignatureDeserializer_Release(des);
    }
done:
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    if (blob) ID3D10Blob_Release(blob);
    if (err) ID3D10Blob_Release(err);
}

/* Titles go device -> IDXGIDevice -> GetAdapter (+0x38) -> EnumOutputs (+0x38), hand the output to
 * SetFrameIntervalX, and take the adapter's parent IDXGIFactory2, checking nothing. */
static void test_dxgi_device(void *dev)
{
    IDXGIDevice *dxgi = NULL;
    IDXGIAdapter *adapter = NULL;
    IDXGIOutput *output = NULL;
    IDXGIFactory2 *factory = NULL;
    void *back = NULL;
    HRESULT hr;

    {
        /* The console's own ID3D12Device8 IID (not PC's) and IGraphicsUnknown. */
        static const GUID device8 = { 0x728f1fa2, 0xea11, 0x4a9f, { 0x9b, 0x52, 0x9b, 0x42, 0xa4, 0x84, 0xd5, 0xc9 } };
        static const GUID graphics_unknown = { 0xaceeea63, 0xe0a9, 0x4a1c, { 0xbb, 0xec, 0x71, 0xb2, 0xf4, 0x85, 0xf7, 0x58 } };
        void *d8 = NULL, *gu = NULL;

        hr = XSLOT(dev, 0, HRESULT, REFIID, void **)(dev, &device8, &d8);
        CHECK(SUCCEEDED(hr) && d8 == dev, "device QueryInterface(console ID3D12Device8) %#lx", hr);
        hr = XSLOT(dev, 0, HRESULT, REFIID, void **)(dev, &graphics_unknown, &gu);
        CHECK(SUCCEEDED(hr) && gu == dev, "device QueryInterface(IGraphicsUnknown) %#lx", hr);
        if (d8)
            XSLOT(d8, 2, ULONG)(d8);
        if (gu)
            XSLOT(gu, 2, ULONG)(gu);
    }
    hr = XSLOT(dev, 0, HRESULT, REFIID, void **)(dev, &IID_IDXGIDevice, (void **)&dxgi);
    CHECK(SUCCEEDED(hr) && dxgi, "device QueryInterface(IDXGIDevice) %#lx", hr);
    if (!dxgi)
        return;
    hr = XSLOT(dxgi, 0, HRESULT, REFIID, void **)(dxgi, &IID_ID3D12Device, &back);
    CHECK(SUCCEEDED(hr) && back == dev, "IDXGIDevice QueryInterface(ID3D12Device) gives the device back %#lx", hr);
    if (back)
        XSLOT(back, 2, ULONG)(back);
    hr = XSLOT(dxgi, 7, HRESULT, IDXGIAdapter **)(dxgi, &adapter);
    CHECK(SUCCEEDED(hr) && adapter, "IDXGIDevice::GetAdapter (+0x38) %#lx", hr);
    if (!adapter)
        return;
    hr = XSLOT(adapter, 7, HRESULT, UINT, IDXGIOutput **)(adapter, 0, &output);
    CHECK(SUCCEEDED(hr) && output, "IDXGIAdapter::EnumOutputs(0) (+0x38) %#lx", hr);
    hr = XSLOT(dev, 78, HRESULT, void *, UINT, UINT)(dev, output, 16667, 1);
    CHECK(SUCCEEDED(hr), "SetFrameIntervalX with the adapter's output %#lx", hr);
    hr = IDXGIAdapter_GetParent(adapter, &IID_IDXGIFactory2, (void **)&factory);
    CHECK(SUCCEEDED(hr) && factory, "adapter's parent IDXGIFactory2 %#lx", hr);
    if (factory)
        IDXGIFactory2_Release(factory);
    if (output)
        IDXGIOutput_Release(output);
    IDXGIAdapter_Release(adapter);
    IDXGIDevice_Release(dxgi);
}

static void test_pacing(void *dev, void *queue)
{
    typedef HRESULT (STDMETHODCALLTYPE *wait_fn)(void *, UINT, UINT, void *, UINT, UINT64 *);
    wait_fn wait = (wait_fn)(*(void ***)dev)[80];
    UINT64 tok[6] = { 0 }, late = 0;
    LARGE_INTEGER t0;
    HANDLE event = CreateEventA(NULL, FALSE, FALSE, NULL);
    HRESULT hr;
    double ms;
    BOOL ok = TRUE;
    int i;

    /* 20 ms, periodicity 3 (not a multiplier): an origin every 20 ms. */
    hr = XSLOT(dev, 78, HRESULT, void *, UINT, UINT)(dev, NULL, 20000, 3);
    CHECK(SUCCEEDED(hr), "SetFrameIntervalX (slot 78) %#lx", hr);
    wait(dev, 0, INFINITE, NULL, 0, &tok[0]);
    QueryPerformanceCounter(&t0);
    for (i = 1; i < 6; i++)
    {
        hr = wait(dev, 0, INFINITE, NULL, 0, &tok[i]);
        ok &= SUCCEEDED(hr) && tok[i] == tok[i - 1] + 1;
    }
    ms = ms_since(t0);
    CHECK(ok && ms > 90 && ms < 200, "WaitFrameEventX (slot 80) paces 5 frames at 20 ms: %.1f ms, tokens %llu..%llu",
          ms, (unsigned long long)tok[0], (unsigned long long)tok[5]);

    /* A late title gets the newest missed frame at once, not a burst; tokens stay consecutive
     * (one per call, as a UE5 title needs: XValkyrie). */
    Sleep(70);
    QueryPerformanceCounter(&t0);
    wait(dev, 0, INFINITE, NULL, 0, &late);
    ms = ms_since(t0);
    CHECK(ms < 10 && late == tok[5] + 1, "late WaitFrameEventX returns at once (%.1f ms) with the next token %llu",
          ms, (unsigned long long)late);

    hr = wait(dev, 0, 0, NULL, 0, &late);
    CHECK(hr == HRESULT_FROM_WIN32(ERROR_TIMEOUT), "WaitFrameEventX with timeout 0 between frames times out (%#lx)", hr);
    /* Flag 2 with no timeout, right after a wait (the target title): the event that happened. */
    {
        UINT64 now_token = 0;
        hr = wait(dev, 0, 0, NULL, 2, &now_token);
        CHECK(hr == S_OK && now_token == late, "WaitFrameEventX(timeout 0, flag 2) gives the last event's token "
              "(%#lx, %llu)", hr, (unsigned long long)now_token);
    }

    /* ScheduleFrameEventX(type, offset, list, flags): the list is { count, HANDLE * }, a UE5 title's
     * pointing at its own flip-event field (TomBox), which is read at each occurrence. An event
     * scheduled 5 ms after the origin is set once per frame. */
    {
        static HANDLE field;
        HANDLE other = CreateEventA(NULL, FALSE, FALSE, NULL);

        field = event;
        {
            /* A temporary, as a UE5 title's is: only the field it points at lives on. */
            struct { UINT32 count, pad; HANDLE *handles; } list = { 1, 0, &field };
            hr = XSLOT(dev, 79, HRESULT, UINT, UINT_PTR, const void *, UINT)(dev, 1, 5000, &list, 0);
            memset(&list, 0, sizeof(list));
        }
        ok = SUCCEEDED(hr);
        QueryPerformanceCounter(&t0);
        for (i = 0; i < 3; i++)
            ok &= WaitForSingleObject(event, 200) == WAIT_OBJECT_0;
        ms = ms_since(t0);
        CHECK(ok && ms > 25 && ms < 120, "ScheduleFrameEventX (slot 79) sets the list's event each frame: 3 in %.1f ms",
              ms);
        field = other;
        CHECK(WaitForSingleObject(other, 200) == WAIT_OBJECT_0,
              "ScheduleFrameEventX reads the title's handle field at each occurrence");
        XSLOT(dev, 79, HRESULT, UINT, UINT_PTR, const void *, UINT)(dev, 1, 0, NULL, 1);
        Sleep(50);
        ResetEvent(other);
        CHECK(WaitForSingleObject(other, 100) == WAIT_TIMEOUT, "ScheduleFrameEventX flag 1 stops the signals");
        CloseHandle(other);
    }

    /* GetFrameStatisticsX (slot 81): interval and frame event at once, rendering, presenting and
     * display once a PresentX named the token (the first plane's first field). */
    {
        BYTE stats[5][56];
        UINT count = 5, types[5];
        UINT64 token = 0, plane[8] = { 0 };
        wait(dev, 0, INFINITE, NULL, 0, &token);
        hr = XSLOT(dev, 81, HRESULT, UINT64, UINT, UINT *, void *)(dev, token, 0x11103, &count, stats);
        CHECK(SUCCEEDED(hr) && count == 2 && *(UINT *)stats[0] == 1 && *(UINT *)(stats[0] + 8) == 20000,
              "GetFrameStatisticsX (slot 81) right after the frame event: %u entries", count);
        Sleep(45);
        count = 5;
        hr = XSLOT(dev, 81, HRESULT, UINT64, UINT, UINT *, void *)(dev, token, 0x11103, &count, stats);
        CHECK(SUCCEEDED(hr) && count == 2, "GetFrameStatisticsX before the frame's PresentX: %u entries", count);
        count = 5;
        hr = XSLOT(dev, 81, HRESULT, UINT64, UINT, UINT *, void *)(dev, token + 50, 0x11103, &count, stats);
        CHECK(SUCCEEDED(hr) && count == 0, "GetFrameStatisticsX for a token never given: %u entries", count);
        plane[0] = token;
        XSLOT(queue, 30, HRESULT, UINT, const void *, const void *)(queue, 1, plane, NULL);
        count = 5;
        hr = XSLOT(dev, 81, HRESULT, UINT64, UINT, UINT *, void *)(dev, token, 0x11103, &count, stats);
        for (i = 0; i < 5; i++)
            types[i] = *(UINT *)stats[i];
        CHECK(SUCCEEDED(hr) && count == 5 && types[2] == 0x100 && types[3] == 0x1000 && types[4] == 0x10000,
              "GetFrameStatisticsX after the frame's PresentX: %u entries", count);
        count = 2;
        hr = XSLOT(dev, 81, HRESULT, UINT64, UINT, UINT *, void *)(dev, token, 0x11103, &count, stats);
        CHECK(SUCCEEDED(hr) && count == 2, "GetFrameStatisticsX fills no more than the room it has (%u)", count);
    }

    XSLOT(dev, 79, HRESULT, UINT, UINT_PTR, const void *, UINT)(dev, 1, 0, NULL, 1);
    XSLOT(dev, 78, HRESULT, void *, UINT, UINT)(dev, NULL, 16667, 1);
    CloseHandle(event);
}

/* Real heap addresses (patched vkd3d-proton, docs/shaders.md): heap slot 10
 * may return the descriptors' real GPU address. A descriptor table set from it
 * (list slot 35) must still reach the right UAV, which a compute shader then
 * writes; vkd3d-proton would misread an untranslated address. */
static void test_real_heap_va(void *dev, void *queue)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(m, "D3D12SerializeRootSignature");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_DESCRIPTOR_RANGE range = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0 };
    D3D12_ROOT_PARAMETER param = { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                                   .DescriptorTable = { 1, &range } };
    D3D12_ROOT_SIGNATURE_DESC rsd = { 1, &param };
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = { .Format = DXGI_FORMAT_R32_TYPELESS,
            .ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = { 0, 64, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW } };
    D3D12_COMPUTE_PIPELINE_STATE_DESC cd = { 0 };
    void *heap = NULL, *rs = NULL, *pso = NULL, *list = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Resource *buf = NULL;
    ID3D12Fence *fence = NULL;
    ID3D10Blob *blob = NULL;
    SIZE_T cpu;
    UINT64 gpu;
    UINT inc, *mapped = NULL;
    HRESULT hr;

    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr) && SUCCEEDED(hr = serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL)))
        hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
    if (SUCCEEDED(hr))
    {
        cd.pRootSignature = rs;
        cd.CS.pShaderBytecode = test_uav_write_cs;
        cd.CS.BytecodeLength = sizeof(test_uav_write_cs);
        hr = XSLOT(dev, 11, HRESULT, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &cd,
                &IID_ID3D12PipelineState, &pso);
    }
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                &IID_ID3D12Fence, (void **)&fence);
    CHECK(SUCCEEDED(hr), "objects for the descriptor table test (%#lx)", hr);
    if (FAILED(hr)) goto done;

    cpu = XSLOT(heap, 9, SIZE_T)(heap);
    gpu = XSLOT(heap, 10, UINT64)(heap);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    printf("info heap GPU start %#llx, CPU start %#llx, increment %u: %s, %s 4 GiB\n", (unsigned long long)gpu,
           (unsigned long long)cpu, inc, gpu == cpu ? "real and identity-mapped" : "not the CPU address",
           gpu < (1ull << 32) ? "below" : "above");
    /* The view goes in entry 1, so the table address carries an offset. */
    XSLOT(dev, 19, void, ID3D12Resource *, ID3D12Resource *, const D3D12_UNORDERED_ACCESS_VIEW_DESC *, SIZE_T)(dev,
            buf, NULL, &uav, cpu + inc);

    XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
    XSLOT(list, 33, void, void *)(list, rs);
    XSLOT(list, 25, void, void *)(list, pso);
    XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
    XSLOT(list, 14, void, UINT, UINT, UINT)(list, 1, 1, 1);
    hr = XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
    for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
        Sleep(10);
    if (SUCCEEDED(hr = ID3D12Resource_Map(buf, 0, NULL, (void **)&mapped)))
    {
        CHECK(mapped[0] == 0x5ca71e77, "compute shader wrote through a descriptor table set from heap slot 10 "
              "(%#x)", mapped[0]);
        mapped[0] = 0;
        ID3D12Resource_Unmap(buf, 0, NULL);
    }
    else
        CHECK(0, "map the UAV buffer (%#lx)", hr);

    /* The same dispatch through list slot 61 as ExecuteIndirect (Leonida, from the retail FSR DLL),
     * with a command signature made through device slot 41. */
    {
        D3D12_INDIRECT_ARGUMENT_DESC arg = { D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH };
        D3D12_COMMAND_SIGNATURE_DESC sd = { sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &arg, 0 };
        D3D12_HEAP_PROPERTIES up = { D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC ad = bd;
        ID3D12CommandSignature *sig = NULL;
        ID3D12Resource *args = NULL;
        D3D12_DISPATCH_ARGUMENTS *dispatch = NULL;

        ad.Flags = D3D12_RESOURCE_FLAG_NONE;
        hr = XSLOT(dev, 41, HRESULT, const D3D12_COMMAND_SIGNATURE_DESC *, void *, REFIID, void **)(dev, &sd, NULL,
                &IID_ID3D12CommandSignature, (void **)&sig);
        if (SUCCEEDED(hr))
            hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &up, D3D12_HEAP_FLAG_NONE,
                    &ad, D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void **)&args);
        if (SUCCEEDED(hr) && SUCCEEDED(hr = ID3D12Resource_Map(args, 0, NULL, (void **)&dispatch)))
        {
            *dispatch = (D3D12_DISPATCH_ARGUMENTS){ 1, 1, 1 };
            ID3D12Resource_Unmap(args, 0, NULL);
        }
        if (SUCCEEDED(hr))
            hr = ID3D12CommandAllocator_Reset(alloc);
        if (SUCCEEDED(hr))
            hr = XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
        CHECK(SUCCEEDED(hr), "command signature (device slot 41) and arguments for list slot 61 (%#lx)", hr);
        if (SUCCEEDED(hr))
        {
            XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
            XSLOT(list, 33, void, void *)(list, rs);
            XSLOT(list, 25, void, void *)(list, pso);
            XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
            XSLOT(list, 61, void, void *, UINT, void *, UINT64, void *, UINT64)(list, sig, 1, args, 0, NULL, 0);
            hr = XSLOT(list, 9, HRESULT)(list);
            XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
            XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 2);
            for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 2; i++)
                Sleep(10);
            if (SUCCEEDED(ID3D12Resource_Map(buf, 0, NULL, (void **)&mapped)))
            {
                CHECK(mapped[0] == 0x5ca71e77, "list slot 61 with a command signature runs ExecuteIndirect (%#x)",
                      mapped[0]);
                mapped[0] = 0;
                ID3D12Resource_Unmap(buf, 0, NULL);
            }
        }
        /* The older layout (SDK 0x20002, a UE5 title's) has ExecuteIndirect in PC's place, 59
         * (XValkyrie's June 2020 kit tables), and RSSetShadingRate at 124. */
        if (SUCCEEDED(hr = ID3D12CommandAllocator_Reset(alloc)))
            hr = XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
        if (SUCCEEDED(hr))
        {
            XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
            XSLOT(list, 33, void, void *)(list, rs);
            XSLOT(list, 25, void, void *)(list, pso);
            XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
            XSLOT(list, 124, void, UINT, const void *)(list, 0, NULL);
            /* SetPipelineState1 at 120 and DispatchRays at 121 (PC 75 and 76): a state object that
             * isn't ours (Unreal 5 sets some) is skipped, and so are its rays. */
            {
                static void *const no_methods[16];
                static struct { void *const *vtbl; BYTE junk[256]; } stranger = { no_methods };
                D3D12_DISPATCH_RAYS_DESC rays = { .Width = 1, .Height = 1, .Depth = 1 };
                XSLOT(list, 120, void, void *)(list, &stranger);
                XSLOT(list, 121, void, const D3D12_DISPATCH_RAYS_DESC *)(list, &rays);
                /* An acceleration structure copy (119, PC 74) into nothing: skipped. */
                XSLOT(list, 119, void, UINT64, UINT64, UINT)(list, 0, 0, 4);
                XSLOT(list, 25, void, void *)(list, pso);
            }
            XSLOT(list, 59, void, void *, UINT, void *, UINT64, void *, UINT64)(list, sig, 1, args, 0, NULL, 0);
            hr = XSLOT(list, 9, HRESULT)(list);
            XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
            XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 3);
            for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 3; i++)
                Sleep(10);
            if (SUCCEEDED(ID3D12Resource_Map(buf, 0, NULL, (void **)&mapped)))
            {
                CHECK(mapped[0] == 0x5ca71e77, "list slot 59 runs ExecuteIndirect, after slots 124, 120 and 121 (%#x)", mapped[0]);
                mapped[0] = 0;
                ID3D12Resource_Unmap(buf, 0, NULL);
            }
        }
        /* vkd3d-proton patch 0017: an argument record nobody filled (all ones) or past D3D12's
         * 2^24 groups a dimension runs empty instead of hanging the GPU, a good record after one
         * still runs, and a count buffer's count is clamped to MaxCommandCount. Without the
         * clamp the first list never finishes. */
        if (SUCCEEDED(hr) && SUCCEEDED(hr = ID3D12Resource_Map(args, 0, NULL, (void **)&dispatch)))
        {
            dispatch[0] = (D3D12_DISPATCH_ARGUMENTS){ ~0u, ~0u, ~0u };
            dispatch[1] = (D3D12_DISPATCH_ARGUMENTS){ 1, 1, 1 };
            dispatch[2] = (D3D12_DISPATCH_ARGUMENTS){ 1u << 24, 1, 1 };
            ((UINT *)dispatch)[16] = ~0u; /* a count, at byte 64 */
            ID3D12Resource_Unmap(args, 0, NULL);
        }
        for (int pass = 0; pass < 2 && SUCCEEDED(hr); pass++)
        {
            UINT64 value = 4 + pass;
            if (SUCCEEDED(hr = ID3D12CommandAllocator_Reset(alloc)))
                hr = XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
            if (FAILED(hr))
                break;
            XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
            XSLOT(list, 33, void, void *)(list, rs);
            XSLOT(list, 25, void, void *)(list, pso);
            XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
            if (!pass)
            {
                XSLOT(list, 61, void, void *, UINT, void *, UINT64, void *, UINT64)(list, sig, 1, args, 0, NULL, 0);
                XSLOT(list, 61, void, void *, UINT, void *, UINT64, void *, UINT64)(list, sig, 1, args, 24, NULL, 0);
            }
            else
                XSLOT(list, 61, void, void *, UINT, void *, UINT64, void *, UINT64)(list, sig, 2, args, 0, args, 64);
            hr = XSLOT(list, 9, HRESULT)(list);
            XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
            XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, value);
            for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < value; i++)
                Sleep(10);
            CHECK(ID3D12Fence_GetCompletedValue(fence) >= value, "ExecuteIndirect of %s finishes",
                  pass ? "a count past MaxCommandCount" : "all-ones and 2^24-group records");
            if (ID3D12Fence_GetCompletedValue(fence) < value)
                break;
            if (SUCCEEDED(ID3D12Resource_Map(buf, 0, NULL, (void **)&mapped)))
            {
                if (!pass)
                    CHECK(mapped[0] == 0, "all-ones and 2^24-group ExecuteIndirect records run empty (%#x)", mapped[0]);
                else
                    CHECK(mapped[0] == 0x5ca71e77, "ExecuteIndirect runs the good record after an all-ones one, "
                          "its count clamped (%#x)", mapped[0]);
                mapped[0] = 0;
                ID3D12Resource_Unmap(buf, 0, NULL);
            }
        }
        if (args) ID3D12Resource_Release(args);
        if (sig) ID3D12CommandSignature_Release(sig);
    }

done:
    if (fence) ID3D12Fence_Release(fence);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (pso)
    {
        /* The title's last Release says 0, but the PSO lives on for a few presents (work in
         * flight may still use it, as on the console). */
        ULONG refs = XSLOT(pso, 2, ULONG)(pso);
        static const GUID probe = { 0x1234, 0x5678, 0x9abc, { 1, 2, 3, 4, 5, 6, 7, 8 } };
        UINT size = 0;
        HRESULT alive = refs ? E_FAIL : XSLOT(pso, 3, HRESULT, REFGUID, UINT *, void *)(pso, &probe, &size, NULL);
        CHECK(!refs && alive == DXGI_ERROR_NOT_FOUND,
              "a PSO's last Release returns 0 and it is kept for later presents (%lu, %#lx)", refs, alive);
    }
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    if (blob) ID3D10Blob_Release(blob);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

/* Console compute pipelines (vkd3d-proton patch 0002, docs/shaders.md): a
 * SPIR-V shader under an SCSV header gets its user SGPRs packed from the root
 * parameters at dispatch. It writes s0-s31 and a marker to the address in
 * s0:s1, root parameter 0 (a root UAV). Expected packing: s0:s1 the UAV's GPU
 * VA, s2-s5 the four root constants, s6:s7 the table's address as heap slot 10 reports it
 * (0 without descriptor buffers, as on lavapipe). */
static void test_console_shader(void *dev, void *queue)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(m, "D3D12SerializeRootSignature");
    static const UINT constants[4] = { 0x11111111, 0x22222222, 0x33333333, 0x44444444 };
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_DESCRIPTOR_RANGE range = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, 0 };
    D3D12_ROOT_PARAMETER params[3] = {
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV, .Descriptor = { 0, 0 } },
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, .Constants = { 0, 0, 4 } },
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, .DescriptorTable = { 1, &range } },
    };
    D3D12_ROOT_SIGNATURE_DESC rsd = { 3, params };
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    D3D12_COMPUTE_PIPELINE_STATE_DESC cd = { 0 };
    void *heap = NULL, *rs = NULL, *pso = NULL, *list = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Resource *buf = NULL;
    ID3D12Fence *fence = NULL;
    ID3D10Blob *blob = NULL;
    UINT64 gpu, va, table;
    UINT inc, *out = NULL;
    HRESULT hr;

    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr) && SUCCEEDED(hr = serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL)))
        hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
    if (SUCCEEDED(hr))
    {
        cd.pRootSignature = rs;
        cd.CS.pShaderBytecode = test_console_comp;
        cd.CS.BytecodeLength = sizeof(test_console_comp);
        hr = XSLOT(dev, 11, HRESULT, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &cd,
                &IID_ID3D12PipelineState, &pso);
        CHECK(SUCCEEDED(hr), "compute PSO from SCSV SPIR-V (patched vkd3d-proton) %#lx", hr);
    }
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                &IID_ID3D12Fence, (void **)&fence);
    if (FAILED(hr)) goto done;

    gpu = XSLOT(heap, 10, UINT64)(heap);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    va = ID3D12Resource_GetGPUVirtualAddress(buf);

    XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
    XSLOT(list, 33, void, void *)(list, rs);
    XSLOT(list, 25, void, void *)(list, pso);
    XSLOT(list, 45, void, UINT, UINT64)(list, 0, va);                                 /* PC 41 */
    XSLOT(list, 39, void, UINT, UINT, const void *, UINT)(list, 1, 4, constants, 0);  /* PC 35 */
    XSLOT(list, 35, void, UINT, UINT64)(list, 2, gpu + 2 * inc);                      /* PC 31 */
    XSLOT(list, 14, void, UINT, UINT, UINT)(list, 1, 1, 1);
    hr = XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
    for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
        Sleep(10);
    if (FAILED(hr = ID3D12Resource_Map(buf, 0, NULL, (void **)&out)))
    {
        CHECK(0, "map the console shader's output (%#lx)", hr);
        goto done;
    }
    table = out[6] | (UINT64)out[7] << 32;
    CHECK(out[32] == 0x5ca7c0de, "console shader ran (marker %#x)", out[32]);
    CHECK((out[0] | (UINT64)out[1] << 32) == va, "s0:s1 = root UAV address %#llx (got %#x:%#x)",
          (unsigned long long)va, out[1], out[0]);
    CHECK(!memcmp(out + 2, constants, sizeof(constants)), "s2-s5 = root constants (%#x %#x %#x %#x)",
          out[2], out[3], out[4], out[5]);
    CHECK(table == 0 || table == gpu + 2 * inc, "s6:s7 = descriptor table at the heap's slot-10 address %#llx (got %#llx)",
          (unsigned long long)(gpu + 2 * inc), (unsigned long long)table);
    printf("info console SGPR table address: %s\n", !table ? "0 (no title or real heap addresses)"
           : title_heap_va() ? "shadow-heap title address" : "real heap address");
    ID3D12Resource_Unmap(buf, 0, NULL);

done:
    if (fence) ID3D12Fence_Release(fence);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (pso) XSLOT(pso, 2, ULONG)(pso);
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    if (blob) ID3D10Blob_Release(blob);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

/* Console graphics pipelines (patch 0002): SCSV VS and PS carrying DXBC
 * I/O signatures. The VS draws a fullscreen triangle into an 8x8 target; the
 * PS writes its SGPRs to dwords 64-95 of the root UAV in s0:s1, and a marker
 * at 96. Same root parameters as the compute test, set through the graphics
 * slots, so the draw path's packing is checked. */
static void test_console_graphics(void *dev, void *queue)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(m, "D3D12SerializeRootSignature");
    static const UINT constants[4] = { 0x55555555, 0x66666666, 0x77777777, 0x88888888 };
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_DESCRIPTOR_HEAP_DESC rtvd = { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    D3D12_DESCRIPTOR_RANGE range = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, 0 };
    D3D12_ROOT_PARAMETER params[3] = {
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV, .Descriptor = { 0, 0 } },
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, .Constants = { 0, 0, 4 } },
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, .DescriptorTable = { 1, &range } },
    };
    D3D12_ROOT_SIGNATURE_DESC rsd = { 3, params };
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 512, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 8, 8, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET };
    D3D12_VIEWPORT vp = { 0, 0, 8, 8, 0, 1 };
    D3D12_RECT sc = { 0, 0, 8, 8 };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gd = { 0 };
    void *heap = NULL, *rtv_heap = NULL, *rs = NULL, *pso = NULL, *list = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Resource *buf = NULL, *rt = NULL;
    ID3D12Fence *fence = NULL;
    ID3D10Blob *blob = NULL;
    UINT64 gpu, va, table;
    SIZE_T rtv;
    UINT inc, *out = NULL;
    HRESULT hr;

    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &rtvd,
                &IID_ID3D12DescriptorHeap, &rtv_heap);
    if (SUCCEEDED(hr) && SUCCEEDED(hr = serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL)))
        hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
    if (SUCCEEDED(hr))
    {
        gd.pRootSignature = rs;
        gd.VS = (D3D12_SHADER_BYTECODE){ test_console_vert, sizeof(test_console_vert) };
        gd.PS = (D3D12_SHADER_BYTECODE){ test_console_frag, sizeof(test_console_frag) };
        gd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        gd.SampleMask = ~0u;
        gd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        gd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        gd.RasterizerState.DepthClipEnable = TRUE;
        gd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        gd.NumRenderTargets = 1;
        gd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        gd.SampleDesc.Count = 1;
        hr = XSLOT(dev, 10, HRESULT, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &gd,
                &IID_ID3D12PipelineState, &pso);
        CHECK(SUCCEEDED(hr), "graphics PSO from SCSV VS/PS with embedded signatures (patched vkd3d-proton) %#lx", hr);
    }
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &td, D3D12_RESOURCE_STATE_RENDER_TARGET, NULL, &IID_ID3D12Resource, (void **)&rt);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                &IID_ID3D12Fence, (void **)&fence);
    if (FAILED(hr)) goto done;

    gpu = XSLOT(heap, 10, UINT64)(heap);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    va = ID3D12Resource_GetGPUVirtualAddress(buf);
    rtv = XSLOT(rtv_heap, 9, SIZE_T)(rtv_heap);
    XSLOT(dev, 20, void, ID3D12Resource *, const void *, SIZE_T)(dev, rt, NULL, rtv);

    XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
    XSLOT(list, 34, void, void *)(list, rs);                                           /* PC 30 */
    XSLOT(list, 25, void, void *)(list, pso);
    XSLOT(list, 46, void, UINT, UINT64)(list, 0, va);                                 /* PC 42 */
    XSLOT(list, 40, void, UINT, UINT, const void *, UINT)(list, 1, 4, constants, 0);  /* PC 36 */
    XSLOT(list, 36, void, UINT, UINT64)(list, 2, gpu + 3 * inc);                      /* PC 32 */
    XSLOT(list, 20, void, D3D12_PRIMITIVE_TOPOLOGY)(list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    XSLOT(list, 21, void, UINT, const D3D12_VIEWPORT *)(list, 1, &vp);
    XSLOT(list, 22, void, UINT, const D3D12_RECT *)(list, 1, &sc);
    XSLOT(list, 50, void, UINT, const SIZE_T *, BOOL, const SIZE_T *)(list, 1, &rtv, FALSE, NULL); /* PC 46 */
    XSLOT(list, 12, void, UINT, UINT, UINT, UINT)(list, 3, 1, 0, 0);
    hr = XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
    for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
        Sleep(10);
    if (FAILED(hr = ID3D12Resource_Map(buf, 0, NULL, (void **)&out)))
    {
        CHECK(0, "map the console pixel shader's output (%#lx)", hr);
        goto done;
    }
    table = out[64 + 6] | (UINT64)out[64 + 7] << 32;
    CHECK(out[96] == 0x5ca7f4a9, "console VS/PS drew (pixel shader marker %#x)", out[96]);
    CHECK((out[64] | (UINT64)out[65] << 32) == va, "graphics s0:s1 = root UAV address %#llx (got %#x:%#x)",
          (unsigned long long)va, out[65], out[64]);
    CHECK(!memcmp(out + 66, constants, sizeof(constants)), "graphics s2-s5 = root constants (%#x %#x %#x %#x)",
          out[66], out[67], out[68], out[69]);
    CHECK(table == 0 || table == gpu + 3 * inc, "graphics s6:s7 = descriptor table at the heap's real address %#llx "
          "(got %#llx)", (unsigned long long)(gpu + 3 * inc), (unsigned long long)table);
    ID3D12Resource_Unmap(buf, 0, NULL);

done:
    if (fence) ID3D12Fence_Release(fence);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (rt) ID3D12Resource_Release(rt);
    if (pso) XSLOT(pso, 2, ULONG)(pso);
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    if (blob) ID3D10Blob_Release(blob);
    if (rtv_heap) XSLOT(rtv_heap, 2, ULONG)(rtv_heap);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

/* Descriptor lookup (patch 4): two dispatches of a console shader that reads
 * the raw descriptor at its table address and finds its heap index in the
 * bound heap's lookup table. Entry 2 is written straight into the
 * shader-visible heap (slot 19), entry 3 copied in from a CPU heap (slot
 * 24). Needs descriptor buffers (RADV); elsewhere the shader reports a skip. */
/* RDNA2 compute shader for the shadow-heap pass trip (llvm-mc -mcpu=gfx1030).
 * s[0:1] points at a CBV/SRV/UAV descriptor table, s[2:3] at a sampler table;
 * the s_load offsets (in dwords 1, 3 and 5) are patched to the heap increments.
 *   s_load_dwordx8 s[4:11], s[0:1], 2*inc         ; T# at index 2
 *   s_load_dwordx4 s[12:15], s[0:1], 1*inc        ; V# at index 1
 *   s_load_dwordx4 s[16:19], s[2:3], 1*sampler_inc ; S# at index 1
 *   s_waitcnt lgkmcnt(0)
 *   v_mov_b32 v1, 0 ; v_mov_b32 v2, 0 ; v_mov_b32 v3, 0
 *   image_sample_lz v[8:11], v[2:3], s[4:11], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
 *   s_waitcnt vmcnt(0)
 *   buffer_store_dword v8, v0, s[12:15], 0 offen
 *   s_endpgm */
static const UINT32 rdna_shadow_round_trip[] = {
    0xf40c0100, 0xfa000040, 0xf4080300, 0xfa000020, 0xf4080401, 0xfa000010, 0xbf8cc07f, 0x7e020280,
    0x7e040280, 0x7e060280, 0xf09c0f08, 0x00810802, 0xbf8c3f70, 0xe0701000, 0x80030800, 0xbf810000,
};

/* Compiles rdna_shadow_round_trip with AnyPS5 against the shadow heaps and
 * checks that its own descriptor decoders found the views written through the
 * API, and that each binding maps back to vkd3d-proton's descriptor. */
static void anyps5_round_trip(UINT64 res_va, const void *res_data, UINT res_size, UINT inc, SIZE_T res_cpu,
        UINT64 smp_va, const void *smp_data, UINT smp_size, UINT sinc, SIZE_T smp_cpu)
{
    HMODULE sa5 = LoadLibraryA("rebox_anyps5.dll");
    BOOL (WINAPI *find)(const UINT32 *, UINT, BOOL, SIZE_T *) =
            (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxShadowFind");
    int (*compile)(const struct sa5_compute_request *, struct sa5_result **, char *, size_t);
    UINT32 (*binding_count)(const struct sa5_result *);
    int (*binding)(const struct sa5_result *, UINT32, struct sa5_binding *);
    void (*result_free)(struct sa5_result *);
    struct sa5_region regions[2] = { { res_va, res_data, res_size }, { smp_va, smp_data, smp_size } };
    UINT32 code[ARRAYSIZE(rdna_shadow_round_trip)], user[4] = { (UINT32)res_va, 0, (UINT32)smp_va, 0 };
    struct sa5_compute_request req = { code, ARRAYSIZE(code), 0x100000, 32, user, 4, { 1, 1, 1 },
                                       0x00ac0000, 4 << 1 /* USER_SGPR */, regions, 2 };
    const UINT32 *t = (const UINT32 *)((const BYTE *)res_data + 2 * inc);
    const UINT32 *v = (const UINT32 *)((const BYTE *)res_data + inc);
    const UINT32 *s = (const UINT32 *)((const BYTE *)smp_data + sinc);
    BOOL found_t = FALSE, found_v = FALSE, found_s = FALSE;
    struct sa5_result *result = NULL;
    char error[512] = "";

    if (!sa5)
    {
        printf("skip AnyPS5 pass trip: no rebox_anyps5.dll\n");
        return;
    }
    compile = (void *)GetProcAddress(sa5, "sa5_compile_compute");
    binding_count = (void *)GetProcAddress(sa5, "sa5_result_binding_count");
    binding = (void *)GetProcAddress(sa5, "sa5_result_binding");
    result_free = (void *)GetProcAddress(sa5, "sa5_result_free");
    memcpy(code, rdna_shadow_round_trip, sizeof(code));
    /* OFFSET is bits 0-20 of the second dword; SOFFSET (null) stays. */
    code[1] = (code[1] & ~0x1fffffu) | 2 * inc;
    code[3] = (code[3] & ~0x1fffffu) | inc;
    code[5] = (code[5] & ~0x1fffffu) | sinc;
    if (compile(&req, &result, error, sizeof(error)))
    {
        CHECK(0, "AnyPS5 compiles the shadow-heap pass-trip shader: %s", error);
        return;
    }
    if (getenv("SMOKE_DUMP_SPIRV"))
    {
        const UINT32 *(*spirv)(const struct sa5_result *, UINT32 *) = (void *)GetProcAddress(sa5, "sa5_result_spirv");
        UINT32 words;
        const UINT32 *code_out = spirv(result, &words);
        FILE *f = fopen(getenv("SMOKE_DUMP_SPIRV"), "wb");
        if (f) { fwrite(code_out, 4, words, f); fclose(f); }
    }
    for (UINT32 i = 0; i < binding_count(result); i++)
    {
        struct sa5_binding b;
        SIZE_T cpu = 0;

        binding(result, i, &b);
        printf("     binding %u: kind %u set %u binding %u count %u words %u shape %d:", i, b.kind, b.set, b.binding,
               b.count, b.word_count, b.image_shape);
        for (UINT32 w = 0; w < b.word_count; w++)
            printf(" %08x", b.words[w]);
        printf("\n");
        if (b.kind == 4 /* SampledImage */ && b.word_count >= 8 && !memcmp(b.words, t, 32))
        {
            found_t = TRUE;
            CHECK(b.image_shape == 1, "AnyPS5 reads the shadow T# as a 2D image (shape %d)", b.image_shape);
            CHECK(find(b.words, 8, FALSE, &cpu) && cpu == res_cpu + 2 * inc,
                  "the T# binding maps back to vkd3d-proton's descriptor 2 (%#Ix)", cpu);
        }
        else if (b.kind == 6 /* Sampler */ && b.word_count >= 4 && !memcmp(b.words, s, 16))
        {
            found_s = TRUE;
            CHECK(find(b.words, 4, TRUE, &cpu) && cpu == smp_cpu + sinc,
                  "the S# binding maps back to vkd3d-proton's sampler 1 (%#Ix)", cpu);
        }
        else if (b.kind <= 3 && b.word_count >= 4 && !memcmp(b.words, v, 16))
        {
            found_v = TRUE;
            CHECK(find(b.words, 4, FALSE, &cpu) && cpu == res_cpu + inc,
                  "the V# binding maps back to vkd3d-proton's descriptor 1 (%#Ix)", cpu);
        }
    }
    CHECK(found_t, "AnyPS5 binds the T# written through CreateShaderResourceView");
    CHECK(found_v, "AnyPS5 binds the V# copied in through CopyDescriptorsSimple");
    CHECK(found_s, "AnyPS5 binds the S# written through CreateSampler");

    /* Fit the module to bindless arrays (made-up sets here) at the heap slots found above, with
     * the flattened SRT as a host buffer, and check the result validates. */
    {
        int (*finalize)(const struct sa5_result *, const struct sa5_remap *, UINT32, const struct sa5_host_buffer *,
                UINT32, UINT32 **, UINT32 *, char *, size_t) = (void *)GetProcAddress(sa5, "sa5_finalize");
        void (*free_words)(UINT32 *) = (void *)GetProcAddress(sa5, "sa5_free_words");
        struct sa5_remap remaps[8];
        struct sa5_host_buffer hosts[4];
        UINT32 indices[8][16], nremap = 0, nhost = 0, *words = NULL, word_count = 0;
        BOOL resolved = TRUE;

        for (UINT32 i = 0; i < binding_count(result) && nremap < 8 && nhost < 4; i++)
        {
            struct sa5_binding b;
            binding(result, i, &b);
            if (b.role <= 2 && b.count <= 16)
            {
                UINT32 per = b.count ? b.word_count / b.count : 0;
                for (UINT32 e = 0; e < b.count; e++)
                {
                    SIZE_T cpu = 0, start = b.kind == 6 ? smp_cpu : res_cpu;
                    if (!find(b.words + e * per, per, b.kind == 6, &cpu))
                        resolved = FALSE;
                    indices[nremap][e] = (UINT32)((cpu - start) / (b.kind == 6 ? sinc : inc));
                }
                remaps[nremap] = (struct sa5_remap){ b.set, b.binding, b.kind, 1 + b.kind, 0, b.count, indices[nremap] };
                nremap++;
            }
            else
                hosts[nhost++] = (struct sa5_host_buffer){ b.set, b.binding, nhost, b.word_count };
        }
        CHECK(resolved && nremap == 3 && nhost == 1, "every guest binding resolves to a heap slot (%u guest, %u host)",
              nremap, nhost);
        if (finalize(result, remaps, nremap, hosts, nhost, &words, &word_count, error, sizeof(error)))
            CHECK(0, "sa5_finalize rebinds the module onto bindless arrays: %s", error);
        else
        {
            CHECK(word_count > 5, "sa5_finalize rebinds the module onto bindless arrays (%u words, valid)", word_count);
            if (getenv("SMOKE_DUMP_SPIRV"))
            {
                char path[MAX_PATH];
                FILE *f;
                snprintf(path, sizeof(path), "%s.final", getenv("SMOKE_DUMP_SPIRV"));
                if ((f = fopen(path, "wb"))) { fwrite(words, 4, word_count, f); fclose(f); }
            }
            free_words(words);
        }

        /* With a draw block, a host buffer whose length is in it (SA5_LENGTH_IN_DRAW_BLOCK) makes the
         * same module whatever its size: a flattened SRT's size baked in made a module (and a
         * pipeline) of a world's large compute shaders for every size it came in. */
        {
            int (*finalize_draw)(const struct sa5_result *, const struct sa5_remap *, UINT32,
                    const struct sa5_host_buffer *, UINT32, UINT32, UINT32 **, UINT32 *, char *, size_t) =
                    (void *)GetProcAddress(sa5, "sa5_finalize_draw");
            int (*validate)(const UINT32 *, UINT32, char *, size_t) = (void *)GetProcAddress(sa5, "sa5_validate");
            UINT32 *a = NULL, *c = NULL, a_count = 0, c_count = 0;
            BOOL baked_differ = FALSE, valid = FALSE;
            for (UINT32 i = 0; i < nremap; i++)
                remaps[i].draw_index_base = SA5_DRAW_ADDRESSES + 4 * i;
            /* Baked, two sizes make two modules. */
            if (finalize_draw && !finalize_draw(result, remaps, nremap, hosts, nhost, 128, &a, &a_count, error,
                    sizeof(error)))
            {
                for (UINT32 i = 0; i < nhost; i++)
                    hosts[i].array_length += 4;
                if (!finalize_draw(result, remaps, nremap, hosts, nhost, 128, &c, &c_count, error, sizeof(error)))
                    baked_differ = a_count != c_count || memcmp(a, c, a_count * 4);
                free_words(a);
                if (c)
                    free_words(c);
                a = c = NULL;
            }
            CHECK(baked_differ, "sa5_finalize_draw with a baked host buffer length: a module for each size");
            for (UINT32 i = 0; i < nhost; i++)
                hosts[i].array_length = SA5_LENGTH_IN_DRAW_BLOCK;
            if (finalize_draw && !finalize_draw(result, remaps, nremap, hosts, nhost, 128, &a, &a_count, error,
                    sizeof(error)))
                valid = validate && !validate(a, a_count, error, sizeof(error));
            CHECK(a && valid, "sa5_finalize_draw reads a host buffer's length from the draw block "
                  "(%u words, %s) %s", a_count, valid ? "valid" : "invalid", valid ? "" : error);
            if (a)
                free_words(a);
            if (c)
                free_words(c);
        }
    }
    result_free(result);
}

/* Console-format shadow heaps (shadow.c): views written through the API come
 * out as RDNA2 descriptors, readable at the heap's title address, and map
 * back to vkd3d-proton's descriptors. */
static void test_shadow_heap(void *dev)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    HRESULT (WINAPI *shadow_heap)(SIZE_T, UINT64 *, void **, UINT *, UINT *) =
            (void *)GetProcAddress(m, "ReboxShadowHeap");
    BOOL (WINAPI *shadow_read)(UINT64, void *, SIZE_T) = (void *)GetProcAddress(m, "ReboxShadowRead");
    BOOL (WINAPI *shadow_find)(const UINT32 *, UINT, BOOL, SIZE_T *) = (void *)GetProcAddress(m, "ReboxShadowFind");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_DESCRIPTOR_HEAP_DESC cd_heap = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    D3D12_DESCRIPTOR_HEAP_DESC sd = { D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 32, 1, 3, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE };
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = { .Format = DXGI_FORMAT_R32_TYPELESS,
            .ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = { 0, 64, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW } };
    D3D12_SAMPLER_DESC smp = { D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                               D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR, 0, 1,
                               D3D12_COMPARISON_FUNC_NEVER, { 0 }, 0, 8 };
    void *heap = NULL, *cpu_heap = NULL, *smp_heap = NULL;
    ID3D12Resource *buf = NULL, *tex = NULL;
    UINT64 title_va = 0, smp_va = 0;
    UINT32 *res, *sam, readback[8];
    UINT inc, sinc, stride, count, scount;
    SIZE_T cpu, cpu_only, scpu, found = 0;
    HRESULT hr;

    CHECK(shadow_heap && shadow_read && shadow_find, "d3d12_xs exports the shadow heap functions");
    if (!shadow_heap || !shadow_read || !shadow_find) return;
    if (!title_heap_va())
    {
        printf("skip shadow heap test: REBOX_HEAP_VA turns title addresses off\n");
        return;
    }
    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &cd_heap,
                &IID_ID3D12DescriptorHeap, &cpu_heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &sd,
                &IID_ID3D12DescriptorHeap, &smp_heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &td, D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&tex);
    CHECK(SUCCEEDED(hr), "objects for the shadow heap test (%#lx)", hr);
    if (FAILED(hr)) goto done;

    cpu = XSLOT(heap, 9, SIZE_T)(heap);
    cpu_only = XSLOT(cpu_heap, 9, SIZE_T)(cpu_heap);
    scpu = XSLOT(smp_heap, 9, SIZE_T)(smp_heap);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    sinc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    if (FAILED(shadow_heap(cpu, &title_va, (void **)&res, &stride, &count)))
    {
        printf("skip shadow heaps: none for this device (increments %u/%u too small?)\n", inc, sinc);
        goto done;
    }
    CHECK(stride == inc && count == 4 && title_va >= 0x10000000 && title_va < 0x100000000ull && !(title_va & 0xffff),
          "shader-visible heap has a shadow: title address %#llx, stride %u, %u descriptors", (unsigned long long)title_va, stride, count);
    CHECK(SUCCEEDED(shadow_heap(scpu, &smp_va, (void **)&sam, NULL, &scount)) && smp_va && smp_va != title_va,
          "sampler heap has its own title address (%#llx)", (unsigned long long)smp_va);
    {
        UINT64 none = 1;
        CHECK(SUCCEEDED(shadow_heap(cpu_only, &none, NULL, NULL, NULL)) && !none,
              "a heap that isn't shader-visible has a shadow but no title address");
    }

    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, tex, NULL, cpu + 2 * inc);
    XSLOT(dev, 19, void, ID3D12Resource *, ID3D12Resource *, const D3D12_UNORDERED_ACCESS_VIEW_DESC *, SIZE_T)(dev,
            buf, NULL, &uav, cpu_only);
    XSLOT(dev, 24, void, UINT, SIZE_T, SIZE_T, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, 1, cpu + inc, cpu_only,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    XSLOT(dev, 22, void, const D3D12_SAMPLER_DESC *, SIZE_T)(dev, &smp, scpu + sinc);

    {
        const UINT32 *t = (const UINT32 *)((BYTE *)res + 2 * inc);
        UINT width = ((t[1] >> 30) | (t[2] & 0xfff) << 2) + 1, height = ((t[2] >> 14) & 0x3fff) + 1;
        CHECK(((t[1] >> 20) & 0x1ff) == 56 && width == 64 && height == 32 && (t[3] >> 28) == 9
              && ((t[3] >> 12) & 0xf) == 0 && ((t[3] >> 16) & 0xf) == 2 && (t[3] & 0xfff) == 0xfac,
              "SRV becomes a T#: 8_8_8_8 UNORM, %ux%u, 2D, mips 0-%u, swizzle %#x", width, height,
              (t[3] >> 16) & 0xf, t[3] & 0xfff);
    }
    {
        const UINT32 *v = (const UINT32 *)((BYTE *)res + inc);
        UINT64 base = v[0] | (UINT64)(v[1] & 0xffff) << 32;
        CHECK(base == ID3D12Resource_GetGPUVirtualAddress(buf) && ((v[1] >> 16) & 0x3fff) == 4 && v[2] == 64,
              "UAV copied from a CPU heap becomes a V#: base %#llx, stride %u, %u records", base,
              (v[1] >> 16) & 0x3fff, v[2]);
    }
    {
        const UINT32 *s = (const UINT32 *)((BYTE *)sam + sinc);
        CHECK((s[0] & 0x1ff) == (2 | 0 << 3 | 1 << 6) && ((s[2] >> 20) & 3) == 1 && ((s[2] >> 22) & 3) == 1
              && ((s[2] >> 26) & 3) == 2 && ((s[1] >> 12) & 0xfff) == 8 * 256,
              "sampler becomes an S#: clamp/wrap/mirror, linear filters, max LOD 8");
    }
    {
        /* Slot 61, CreateSamplerX: the same sampler in the console's own terms gives the same S#. */
        struct { UINT flags, mag, min, mip, z, mode; D3D12_TEXTURE_ADDRESS_MODE u, v, w;
                 FLOAT min_lod, max_lod, bias, bias2; UINT aniso; FLOAT aniso_bias; UINT threshold;
                 D3D12_COMPARISON_FUNC cmp; FLOAT border[4]; } x = {
            0, 1, 1, 2, 0, 0, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            D3D12_TEXTURE_ADDRESS_MODE_MIRROR, 0.0f, 8.0f, 0.0f, 0.0f, 1, 0.0f, 0, D3D12_COMPARISON_FUNC_NEVER };
        XSLOT(dev, 61, void, const void *, SIZE_T)(dev, &x, scpu);
        CHECK(!memcmp(sam, (BYTE *)sam + sinc, 16), "CreateSamplerX (slot 61) gives the same S# as CreateSampler");
        /* Leave slot 0 different, so the S# lookup below finds slot 1. */
        x.mag = x.min = 0;
        x.mip = 1;
        XSLOT(dev, 61, void, const void *, SIZE_T)(dev, &x, scpu);
        CHECK(memcmp(sam, (BYTE *)sam + sinc, 16), "CreateSamplerX with point filters gives a different S#");
    }
    CHECK(shadow_read(title_va + 2 * inc, readback, 32) && !memcmp(readback, (BYTE *)res + 2 * inc, 32),
          "ReboxShadowRead reads the T# at its title address");
    CHECK(!shadow_read(title_va + 3 * inc, readback, 2 * inc), "ReboxShadowRead refuses a read past the heap");
    CHECK(shadow_find((const UINT32 *)((BYTE *)res + inc), 4, FALSE, &found) && found == cpu + inc,
          "the V# maps back to the shader-visible copy, not the CPU heap (%#Ix)", found);

    anyps5_round_trip(title_va, res, 4 * inc, inc, cpu, smp_va, sam, scount * sinc, sinc, scpu);

done:
    if (buf) ID3D12Resource_Release(buf);
    if (tex) ID3D12Resource_Release(tex);
    if (smp_heap) XSLOT(smp_heap, 2, ULONG)(smp_heap);
    if (cpu_heap) XSLOT(cpu_heap, 2, ULONG)(cpu_heap);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

/* A console mesh stage's bindings at every draw (console_compile.c's compile_now after AnyPS5's memo
 * hits): a V# and four T#s from a table in GPU memory, the first T# a view in a shader-visible heap,
 * the others T#s the title wrote itself that name no texture (bound to the dummy). Timed per call
 * with few textures and GPU memory chunks, then with thousands, as a world has; with the binding
 * cache (ReboxConsoleBindingStats), a change to a descriptor in the table or in the heap must give
 * new bindings, and an unchanged one the same. */
static const UINT32 bindings_bench_code[] = {
    0xbfa00003,                         /* s_inst_prefetch 3 */
    0xf4080508, 0xfa000000,             /* s_load_dwordx4 s[20:23], s[16:17], 0x0: the V# */
    0xf40c0608, 0xfa000020,             /* s_load_dwordx8 s[24:31], s[16:17], 0x20: T# 0 */
    0xf40c0808, 0xfa000040,             /* s_load_dwordx8 s[32:39], s[16:17], 0x40: T# 1 */
    0xf40c0a08, 0xfa000060,             /* s_load_dwordx8 s[40:47], s[16:17], 0x60: T# 2 */
    0xf40c0c08, 0xfa000080,             /* s_load_dwordx8 s[48:55], s[16:17], 0x80: T# 3 */
    0xbf8cc07f,                         /* s_waitcnt lgkmcnt(0) */
    0xf4200e0a, 0xfa000000,             /* s_buffer_load_dword s56, s[20:23], 0x0 */
    0xbf8cc07f,
    0x7e080280, 0x7e0a0280,             /* v_mov_b32 v4, 0; v_mov_b32 v5, 0 */
    0xf0000f08, 0x00060804,             /* image_load v[8:11], v[4:5], s[24:31] dmask:0xf 2D */
    0xf0000f08, 0x00080c04,             /* image_load v[12:15], v[4:5], s[32:39] */
    0xf0000f08, 0x000a1004,             /* image_load v[16:19], v[4:5], s[40:47] */
    0xf0000f08, 0x000c1404,             /* image_load v[20:23], v[4:5], s[48:55] */
    0xbf8c3f70,                         /* s_waitcnt vmcnt(0) */
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xb07c1003, 0xbf900009, 0x7d880083, 0xbefe046a,
    0x7e020d00, 0x7e040280,
    0x7e060238,                         /* v_mov_b32 v3, s56 */
    0xf80008cf, 0x03020201,             /* exp pos0 v1, v2, v2, v3 done */
    0xf800020f, 0x14100c08,             /* exp param0 v8, v12, v16, v20 */
    0x7d840080, 0xbefe046a, 0x7e0802ff, 0x00200400,
    0xf8000941, 0x00000004, 0xbf810000,
};

/* The heap slots a mesh stage's draw block gives its bindings (after the 24 words of addresses and
 * lengths), through ReboxConsoleDrawBlock; FALSE without that export. */
static BOOL bench_slots(const struct rebox_compile_out *out, UINT32 *slots, UINT count)
{
    BOOL (WINAPI *block)(UINT64, void *, SIZE_T) = (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"),
            "ReboxConsoleDrawBlock");
    UINT32 words[24 + 8];
    UINT64 va = out->push[24] | (UINT64)out->push[25] << 32;
    if (!block || count > 8 || !block(va, words, (24 + count) * 4))
        return FALSE;
    memcpy(slots, words + 24, count * 4);
    return TRUE;
}

static double bench_compiles(HRESULT (WINAPI *compile)(const struct rebox_compile_in *, struct rebox_compile_out *),
                             const struct rebox_compile_in *in, UINT calls, UINT *failed)
{
    struct rebox_compile_out out;
    LARGE_INTEGER a, b, f;
    *failed = 0;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&a);
    for (UINT i = 0; i < calls; i++)
        *failed += FAILED(compile(in, &out));
    QueryPerformanceCounter(&b);
    return (double)(b.QuadPart - a.QuadPart) * 1e6 / f.QuadPart / calls;
}

/* The lookups a binding makes, timed: a T# by its texture's address (console_texture_of) and a guest
 * buffer at a title address (guest_buffer_mappable's GPU memory lookups), in ns a call. */
static void bench_lookups(const UINT32 *t, double *texture_ns, double *buffer_ns)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    ID3D12Resource *(WINAPI *texture_of)(const UINT32 *) = (void *)GetProcAddress(m, "ReboxConsoleTextureOf");
    UINT64 (WINAPI *buffer_length)(UINT64, UINT64) = (void *)GetProcAddress(m, "ReboxGuestBufferLength");
    LARGE_INTEGER a, b, f;
    UINT64 sink = 0;
    QueryPerformanceFrequency(&f);
    *texture_ns = *buffer_ns = 0;
    if (texture_of)
    {
        QueryPerformanceCounter(&a);
        for (UINT i = 0; i < 20000; i++)
            sink += (UINT_PTR)texture_of(t);
        QueryPerformanceCounter(&b);
        *texture_ns = (double)(b.QuadPart - a.QuadPart) * 1e9 / f.QuadPart / 20000;
    }
    if (buffer_length)
    {
        QueryPerformanceCounter(&a);
        for (UINT i = 0; i < 20000; i++)
            sink += buffer_length(0x3000000000ull + (i & 15) * 0x10000, 256);
        QueryPerformanceCounter(&b);
        *buffer_ns = (double)(b.QuadPart - a.QuadPart) * 1e9 / f.QuadPart / 20000;
    }
    if (sink == 1)
        printf("\n");
}

static void test_console_bindings(void *dev)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    void *(WINAPI *from_container)(const void *, SIZE_T, UINT32, SIZE_T *) = (void *)GetProcAddress(m,
            "ReboxConsoleGraphicsFromContainer");
    HRESULT (WINAPI *compile)(const struct rebox_compile_in *, struct rebox_compile_out *) = (void *)GetProcAddress(m,
            "ReboxConsoleCompile");
    void (WINAPI *release)(void *) = (void *)GetProcAddress(m, "ReboxConsoleFree");
    HRESULT (WINAPI *shadow_heap)(SIZE_T, UINT64 *, void **, UINT *, UINT *) = (void *)GetProcAddress(m,
            "ReboxShadowHeap");
    void (WINAPI *cache_stats)(UINT64 *) = (void *)GetProcAddress(m, "ReboxConsoleBindingStats");
    SIZE_T (WINAPI *inner_cpu)(SIZE_T) = (void *)GetProcAddress(m, "ReboxShadowInnerCpu");
    gpu_alloc_fn galloc = (gpu_alloc_fn)(void *)GetProcAddress(m, "ReboxGpuAlloc");
    gpu_free_fn gfree = (gpu_free_fn)(void *)GetProcAddress(m, "ReboxGpuFree");
    enum { TEXTURES = 2048, CHUNKS = 512, CALLS = 2000 };
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, TEXTURES,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 4, 4, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE };
    static ID3D12Resource *textures[TEXTURES];
    static void *chunks[CHUNKS];
    void *heap = NULL;
    UINT32 *table = NULL, sgprs[32] = { 0 }, slots[4], again[4], t0[8];
    UINT64 table_gpu = 0, title_va, stats[4] = { 0 }, before[4] = { 0 };
    const struct rebox_deferred_shader *h = NULL;
    struct rebox_compile_in in;
    struct rebox_compile_out out;
    UINT inc, stride, count, made = 0, chunk_count = 0, failed;
    SIZE_T cpu, size, blob_size = 0;
    BYTE *container = NULL, *data;
    double small, large, texture_ns[2], buffer_ns[2];
    HRESULT hr;

    if (!title_heap_va() || !from_container || !compile || !release || !shadow_heap || !galloc || !gfree)
    {
        printf("skip console bindings: no title heap addresses or exports\n");
        return;
    }
    if (!GetModuleHandleA("rebox_anyps5.dll") && !LoadLibraryA("rebox_anyps5.dll"))
    {
        printf("skip console bindings: no rebox_anyps5.dll\n");
        return;
    }
    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    for (UINT i = 0; SUCCEEDED(hr) && i < 2; i++, made++)
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &td, D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&textures[i]);
    if (SUCCEEDED(hr))
        hr = galloc(4096, 256, (void **)&table, &table_gpu);
    CHECK(SUCCEEDED(hr), "objects for the console bindings test (%#lx)", hr);
    if (FAILED(hr))
        goto done;
    cpu = XSLOT(heap, 9, SIZE_T)(heap);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, textures[0], NULL, cpu);
    if (FAILED(shadow_heap(cpu, &title_va, (void **)&data, &stride, &count)))
    {
        printf("skip console bindings: no shadow heap\n");
        goto done;
    }
    memcpy(t0, data, 32);

    /* The table: a V# of 16 bytes at +0x800, T# 0 the heap's view, T#s 1-3 at addresses no texture is at. */
    memset(table, 0, 4096);
    table[0] = (UINT32)(table_gpu + 0x800);
    table[1] = (UINT32)((table_gpu + 0x800) >> 32) & 0xffff;
    table[2] = 16;
    table[3] = (77u << 12) | 0xfac;
    for (UINT k = 0; k < 4; k++)
    {
        memcpy(table + 8 + 8 * k, t0, 32);
        if (k)
            table[8 + 8 * k] = 0x00123400 + k * 0x100, table[9 + 8 * k] = (t0[1] & ~0xffu) | 0x3f;
    }
    if (!(container = mesh_container(bindings_bench_code, sizeof(bindings_bench_code), &size))
            || !(h = from_container(container, size, 0x80, &blob_size)))
    {
        CHECK(0, "a deferred mesh stage for the console bindings test");
        goto done;
    }
    memset(&in, 0, sizeof(in));
    for (UINT k = 0; k < 7; k++)
        in.bindless[k][0] = 0, in.bindless[k][1] = k;
    in.shader = h;
    in.shader_size = (UINT32)blob_size;
    in.user_sgpr_count = 32;
    in.user_sgprs = sgprs;
    in.heap_cpu[0] = inner_cpu ? inner_cpu(cpu) : cpu;
    in.increment[0] = inc;
    sgprs[8] = (UINT32)table_gpu;
    sgprs[9] = (UINT32)(table_gpu >> 32);
    hr = compile(&in, &out);
    if (FAILED(hr))
    {
        char message[512];
        void (WINAPI *last)(char *, SIZE_T) = (void *)GetProcAddress(m, "ReboxConsoleLastResult");
        if (last)
            last(message, sizeof(message));
        CHECK(0, "the console bindings test's mesh stage compiles (%#lx: %s)", hr, last ? message : "?");
        goto done;
    }
    if (bench_slots(&out, slots, 4))
        CHECK(slots[0] == 0 && slots[1] >= TEXTURES && slots[1] == slots[2] && slots[2] == slots[3],
              "T# 0 binds its heap slot (%u), the T#s at no texture's address the dummy (%u %u %u)", slots[0],
              slots[1], slots[2], slots[3]);

    small = bench_compiles(compile, &in, CALLS, &failed);
    bench_lookups(table + 16, &texture_ns[0], &buffer_ns[0]);
    /* A world's registries: thousands of textures viewed, hundreds of GPU memory chunks. */
    for (; made < TEXTURES; made++)
    {
        if (FAILED(XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &td, D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&textures[made])))
            break;
        XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, textures[made],
                NULL, cpu + (SIZE_T)made * inc);
    }
    for (UINT64 gpu; chunk_count < CHUNKS && SUCCEEDED(galloc(256, 128 << 10, &chunks[chunk_count], &gpu)); chunk_count++)
        ;
    compile(&in, &out); /* the memory regions and page table made again */
    large = bench_compiles(compile, &in, CALLS, &failed);
    bench_lookups(table + 16, &texture_ns[1], &buffer_ns[1]);
    printf("     console bindings bench: %.1f us a draw with few textures and chunks, %.1f us with %u textures and "
           "%u more GPU memory chunks (%u calls each, %u failed); a T# by address %.0f then %.0f ns, a guest buffer "
           "at a title address %.0f then %.0f ns\n", small, large, made, chunk_count, CALLS, failed, texture_ns[0],
           texture_ns[1], buffer_ns[0], buffer_ns[1]);
    CHECK(!failed, "the console bindings test's draws compile (%u failed)", failed);

    if (!cache_stats)
        goto done;
    /* Unchanged: the same slots, from the cache. */
    cache_stats(before);
    hr = compile(&in, &out);
    cache_stats(stats);
    CHECK(SUCCEEDED(hr) && bench_slots(&out, again, 4) && !memcmp(again, slots, sizeof(slots)) && stats[0] > before[0],
          "an unchanged draw reuses its bindings (slots %u %u %u %u, %llu hits)", again[0], again[1], again[2], again[3],
          (unsigned long long)stats[0]);
    /* T# 1 made the view in slot 5: a new binding. */
    memcpy(table + 16, data + 5 * stride, 32);
    hr = compile(&in, &out);
    CHECK(SUCCEEDED(hr) && bench_slots(&out, again, 4) && again[0] == 0 && again[1] == 5,
          "a T# changed in the table gives its new slot (%u %u)", again[0], again[1]);
    /* Slot 0 viewing another texture: T# 0's words are nowhere in the heap now, so its texture is found
     * by the address in them and viewed in the reserved tail. */
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, textures[1], NULL, cpu);
    hr = compile(&in, &out);
    CHECK(SUCCEEDED(hr) && bench_slots(&out, again, 4) && again[0] >= TEXTURES && again[0] != slots[1] && again[1] == 5,
          "a heap slot written again gives the T# it held a view of its own (%u, the dummy is %u)", again[0], slots[1]);
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, textures[0], NULL, cpu);
    hr = compile(&in, &out);
    CHECK(SUCCEEDED(hr) && bench_slots(&out, again, 4) && again[0] == 0,
          "and written back, binds it again (%u)", again[0]);
    /* An empty T#: the dummy, and noted for a late fill at every draw, cached or not. */
    memset(table + 24, 0, 32);
    cache_stats(before);
    compile(&in, &out);
    hr = compile(&in, &out);
    cache_stats(stats);
    CHECK(SUCCEEDED(hr) && stats[2] >= before[2] + 2 && bench_slots(&out, again, 4) && again[2] == slots[1],
          "an empty T# binds the dummy and is noted for a late fill at each draw (%llu notes)",
          (unsigned long long)(stats[2] - before[2]));

done:
    if (h)
        release((void *)h);
    free(container);
    for (UINT i = 0; i < chunk_count; i++)
        gfree(chunks[i]);
    if (table)
        gfree(table);
    for (UINT i = 0; i < made; i++)
        if (textures[i])
            ID3D12Resource_Release(textures[i]);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

/* A single-level texture's SRV and UAV have the same T#: a binding finds the slot holding the view
 * it wants (a storage image bound to the SRV's slot, or a sampled one to the UAV's, was refused), and
 * a sampled binding of a UAV copied from a CPU heap still gets an SRV of its texture. */
static void test_shadow_views(void *dev)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    UINT (WINAPI *find_index)(SIZE_T, const UINT32 *, UINT, UINT) = (void *)GetProcAddress(m, "ReboxShadowFindIndex");
    UINT (WINAPI *srv_for_uav)(SIZE_T, UINT) = (void *)GetProcAddress(m, "ReboxShadowSrvForUav");
    HRESULT (WINAPI *shadow_heap)(SIZE_T, UINT64 *, void **, UINT *, UINT *) =
            (void *)GetProcAddress(m, "ReboxShadowHeap");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_DESCRIPTOR_HEAP_DESC cd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 32, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    void *heap = NULL, *cpu_heap = NULL;
    ID3D12Resource *tex = NULL;
    UINT32 *res, words[8];
    UINT64 title_va;
    UINT inc, stride, count, uav_slot, srv_slot;
    SIZE_T cpu, cpu_only;
    HRESULT hr;

    CHECK(find_index && srv_for_uav && shadow_heap, "d3d12_xs exports the shadow view lookups");
    if (!find_index || !srv_for_uav || !shadow_heap) return;
    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &cd,
                &IID_ID3D12DescriptorHeap, &cpu_heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &td, D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&tex);
    CHECK(SUCCEEDED(hr), "objects for the shadow view test (%#lx)", hr);
    if (FAILED(hr)) goto done;
    cpu = XSLOT(heap, 9, SIZE_T)(heap);
    cpu_only = XSLOT(cpu_heap, 9, SIZE_T)(cpu_heap);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (FAILED(shadow_heap(cpu, &title_va, (void **)&res, &stride, &count)))
    {
        printf("skip shadow views: no shadow for this device\n");
        goto done;
    }
    /* Slot 0 the UAV, slot 2 the SRV, slot 3 the UAV again, copied from a CPU heap. */
    XSLOT(dev, 19, void, ID3D12Resource *, ID3D12Resource *, const D3D12_UNORDERED_ACCESS_VIEW_DESC *, SIZE_T)(dev,
            tex, NULL, NULL, cpu);
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, tex, NULL, cpu + 2 * inc);
    XSLOT(dev, 19, void, ID3D12Resource *, ID3D12Resource *, const D3D12_UNORDERED_ACCESS_VIEW_DESC *, SIZE_T)(dev,
            tex, NULL, NULL, cpu_only);
    XSLOT(dev, 24, void, UINT, SIZE_T, SIZE_T, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, 1, cpu + 3 * inc, cpu_only,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    memcpy(words, res, sizeof(words));
    if (memcmp(words, (BYTE *)res + 2 * stride, sizeof(words)))
    {
        printf("skip shadow views: the texture's SRV and UAV have different T#s here\n");
        goto done;
    }
    uav_slot = find_index(cpu, words, 8, 1u << 4 /* SHADOW_VIEW_UAV_TEXTURE */);
    srv_slot = find_index(cpu, words, 8, 1u << 1 /* SHADOW_VIEW_SRV_TEXTURE */);
    CHECK(uav_slot == 0 || uav_slot == 3, "a storage image's T# finds a UAV slot (%u), not the SRV's", uav_slot);
    CHECK(srv_slot == 2, "a sampled image's T# finds the SRV slot (%u), not a UAV's", srv_slot);
    CHECK(srv_for_uav(cpu, 0) >= count && srv_for_uav(cpu, 0) != ~0u,
          "a texture UAV made in the heap gets an SRV in its reserved tail (%u)", srv_for_uav(cpu, 0));
    CHECK(srv_for_uav(cpu, 3) >= count && srv_for_uav(cpu, 3) != ~0u,
          "a texture UAV copied from a CPU heap gets an SRV in the reserved tail too (%u)", srv_for_uav(cpu, 3));

done:
    if (tex) ID3D12Resource_Release(tex);
    if (cpu_heap) XSLOT(cpu_heap, 2, ULONG)(cpu_heap);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

/* RDNA2 compute shader for the deferred dispatch test (llvm-mc -mcpu=gfx1030): stores
 * 0x1234abcd through the V# at the start of the descriptor table s[0:1] points at. It opens with
 * the FLAT_SCRATCH setup console shaders have, which d3d12_xs NOPs out.
 *   s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0x1000
 *   s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI, 0, 16), 2
 *   s_load_dwordx4 s[4:7], s[0:1], 0x0
 *   s_waitcnt lgkmcnt(0)
 *   v_mov_b32 v1, 0x1234abcd
 *   v_mov_b32 v2, 0
 *   buffer_store_dword v1, v2, s[4:7], 0 offen
 *   s_endpgm */
static const UINT32 rdna_uav_write[] = {
    0xba80f814, 0x00001000, 0xba807815, 0x00000002, 0xf4080100, 0xfa000000, 0xbf8cc07f, 0x7e0202ff, 0x1234abcd, 0x7e040280, 0xe0701000, 0x80010102, 0xbf810000,
};

/* A console DXBC container as the game ships them: a DXIL part holding an XBCH header and a zlib
 * stream (here one stored deflate block) of FSSM, then a NOBC stub. The FSSM header gives the code
 * offset (u32 at 0xC0, plus 0x28), float mode (0x104), RSRC2 (0x138) and thread-group size (0x13C).
 * Returns the container's size; free *out. */
/* The console driver's own container (FSPX in a DXBC part ACCC), as FidelityFX's console upscaler
 * creates its passes from: a head entry (thread-group size, RSRC1, RSRC2) and the code behind a
 * 0x24-byte header, padded after s_endpgm with s_code_end as the driver does. */
static SIZE_T make_fspx_container(const UINT32 *code, UINT dwords, UINT32 rsrc2, BYTE **out)
{
    enum { PAD = 4 };
    SIZE_T code_entry = 0x24 + (dwords + PAD) * 4, fspx_size = 0x30 + 0x1c + code_entry;
    SIZE_T total = 0x24 + 8 + ((fspx_size + 3) & ~(SIZE_T)3);
    BYTE *c = calloc(1, total), *f = c + 0x24 + 8;
    UINT32 *e = (UINT32 *)(f + 0x10);

    memcpy(c, "DXBC", 4);
    *(UINT32 *)(c + 0x14) = 1;
    *(UINT32 *)(c + 0x18) = (UINT32)total;
    *(UINT32 *)(c + 0x1c) = 1;
    *(UINT32 *)(c + 0x20) = 0x24;
    memcpy(c + 0x24, "ACCC", 4);
    *(UINT32 *)(c + 0x28) = (UINT32)fspx_size;
    memcpy(f, "FSPX", 4);
    *(UINT32 *)(f + 4) = 1;
    *(UINT32 *)(f + 8) = 2;
    *(UINT32 *)(f + 12) = (UINT32)fspx_size;
    e[0] = 0; e[1] = 0x30; e[2] = 0x1c;                     /* head */
    e[4] = 6; e[5] = 0x30 + 0x1c; e[6] = (UINT32)code_entry; /* code */
    *(UINT16 *)(f + 0x30 + 0x0c) = 1;
    *(UINT16 *)(f + 0x30 + 0x0e) = 1;
    *(UINT16 *)(f + 0x30 + 0x10) = 1;
    *(UINT32 *)(f + 0x30 + 0x14) = 0x40000000u | 1u << 21 | 0xc0u << 12;
    *(UINT32 *)(f + 0x30 + 0x18) = rsrc2;
    memcpy(f + 0x4c + 0x24, code, dwords * 4);
    for (int i = 0; i < PAD; i++)
        ((UINT32 *)(f + 0x4c + 0x24))[dwords + i] = 0xbf9f0000; /* s_code_end */
    *out = c;
    return total;
}

static SIZE_T make_console_container(const UINT32 *code, UINT dwords, UINT32 rsrc2, BYTE **out)
{
    SIZE_T fssm_size = 0x168 + dwords * 4, zsize = 2 + 5 + fssm_size + 4, part = 4 + 0x60 + zsize + 4 + 0x14;
    SIZE_T total = 0x24 + 8 + ((part + 3) & ~(SIZE_T)3);
    BYTE *c = calloc(1, total), *fssm = calloc(1, fssm_size), *p;
    UINT32 a = 1, b = 0;

    memcpy(fssm, "FSSM", 4);
    *(UINT32 *)(fssm + 0xc0) = 0x168 - 0x28;
    *(UINT32 *)(fssm + 0x104) = 0xc0;
    *(UINT32 *)(fssm + 0x138) = rsrc2;
    *(UINT32 *)(fssm + 0x13c) = 1;
    *(UINT32 *)(fssm + 0x140) = 1;
    *(UINT32 *)(fssm + 0x144) = 1;
    memcpy(fssm + 0x168, code, dwords * 4);
    for (SIZE_T i = 0; i < fssm_size; i++)
    {
        a = (a + fssm[i]) % 65521;
        b = (b + a) % 65521;
    }
    memcpy(c, "DXBC", 4);
    *(UINT32 *)(c + 0x14) = 1;
    *(UINT32 *)(c + 0x18) = (UINT32)total;
    *(UINT32 *)(c + 0x1c) = 1;
    *(UINT32 *)(c + 0x20) = 0x24;
    p = c + 0x24;
    memcpy(p, "DXIL", 4);
    *(UINT32 *)(p + 4) = (UINT32)(total - 0x24 - 8);
    p += 8;
    memcpy(p, "XBCH", 4);
    p += 4 + 0x60;
    *p++ = 0x78;
    *p++ = 0x01;
    *p++ = 0x01; /* final stored block */
    *p++ = (BYTE)fssm_size;
    *p++ = (BYTE)(fssm_size >> 8);
    *p++ = (BYTE)~fssm_size;
    *p++ = (BYTE)(~fssm_size >> 8);
    memcpy(p, fssm, fssm_size);
    p += fssm_size;
    *p++ = (BYTE)(b >> 8);
    *p++ = (BYTE)b;
    *p++ = (BYTE)(a >> 8);
    *p++ = (BYTE)a;
    memcpy(p, "NOBC", 4);
    free(fssm);
    *out = c;
    return total;
}

/* A deferred console compute PSO (patch 0005, console_compile.c): vkd3d-proton keeps the RDNA
 * code and, at dispatch, d3d12_xs compiles it with AnyPS5 against the shadow heaps and rebinds it
 * onto vkd3d-proton's bindless sets. The table starts at heap slot 1, so a wrong slot shows. */
static void test_console_dispatch(void *dev, void *queue)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(m, "D3D12SerializeRootSignature");
    struct
    {
        UINT32 magic, version, size, stage, wave_size, workgroup_size[3], rsrc1, rsrc2, code_offset, code_dwords;
        UINT64 code_address;
        UINT32 code[ARRAYSIZE(rdna_uav_write)];
    } blob = { 0x44524353, 1, sizeof(blob), 0x20 /* VK_SHADER_STAGE_COMPUTE_BIT */, 32, { 1, 1, 1 },
               0x00ac0000, 2 << 1 /* two user SGPRs */, offsetof(typeof(blob), code), ARRAYSIZE(rdna_uav_write),
               0x200000 };
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_DESCRIPTOR_RANGE range = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0 };
    D3D12_ROOT_PARAMETER param = { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                                   .DescriptorTable = { 1, &range } };
    D3D12_ROOT_SIGNATURE_DESC rsd = { 1, &param };
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = { .Format = DXGI_FORMAT_R32_TYPELESS,
            .ViewDimension = D3D12_UAV_DIMENSION_BUFFER, .Buffer = { 0, 64, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW } };
    D3D12_COMPUTE_PIPELINE_STATE_DESC cd = { 0 };
    void *heap = NULL, *rs = NULL, *pso = NULL, *list = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Resource *out_buf = NULL;
    ID3D12Fence *fence = NULL;
    ID3D10Blob *rs_blob = NULL;
    UINT inc, *out = NULL;
    SIZE_T cpu;
    UINT64 gpu;
    HRESULT hr;

    memcpy(blob.code, rdna_uav_write, sizeof(blob.code));
    if (!title_heap_va())
    {
        printf("skip deferred console dispatch: REBOX_HEAP_VA turns title addresses off\n");
        return;
    }
    if (!GetModuleHandleA("rebox_anyps5.dll") && !LoadLibraryA("rebox_anyps5.dll"))
    {
        printf("skip deferred console dispatch: no rebox_anyps5.dll\n");
        return;
    }
    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr) && SUCCEEDED(hr = serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rs_blob, NULL)))
        hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                ID3D10Blob_GetBufferPointer(rs_blob), ID3D10Blob_GetBufferSize(rs_blob), &IID_ID3D12RootSignature, &rs);
    if (SUCCEEDED(hr))
    {
        cd.pRootSignature = rs;
        cd.CS = (D3D12_SHADER_BYTECODE){ &blob, sizeof(blob) };
        hr = XSLOT(dev, 11, HRESULT, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &cd,
                &IID_ID3D12PipelineState, &pso);
        CHECK(SUCCEEDED(hr), "a deferred console compute PSO is created without compiling (%#lx)", hr);
    }
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &hp, D3D12_HEAP_FLAG_NONE,
                &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void **)&out_buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                &IID_ID3D12Fence, (void **)&fence);
    CHECK(SUCCEEDED(hr), "objects for the deferred dispatch test (%#lx)", hr);
    if (FAILED(hr)) goto done;

    cpu = XSLOT(heap, 9, SIZE_T)(heap);
    gpu = XSLOT(heap, 10, UINT64)(heap);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    XSLOT(dev, 19, void, ID3D12Resource *, ID3D12Resource *, const D3D12_UNORDERED_ACCESS_VIEW_DESC *, SIZE_T)(dev,
            out_buf, NULL, &uav, cpu + inc);

    XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
    XSLOT(list, 33, void, void *)(list, rs);
    XSLOT(list, 25, void, void *)(list, pso);
    XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
    XSLOT(list, 14, void, UINT, UINT, UINT)(list, 1, 1, 1);
    hr = XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
    for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
        Sleep(10);
    if (FAILED(hr = ID3D12Resource_Map(out_buf, 0, NULL, (void **)&out)))
    {
        CHECK(0, "map the deferred shader's output (%#lx)", hr);
        goto done;
    }
    CHECK(out[0] == 0x1234abcd, "the deferred console shader, compiled by AnyPS5 at dispatch, wrote %#x through "
          "its V# (expected 0x1234abcd)", out[0]);

    /* Same code, but s[0:1] comes from two root constants and points into GPU memory from
     * ReboxGpuAlloc holding a copy of the V#: the recompiler has to read that memory too. */
    {
        HRESULT (WINAPI *galloc)(SIZE_T, SIZE_T, void **, UINT64 *) = (void *)GetProcAddress(m, "ReboxGpuAlloc");
        HRESULT (WINAPI *shadow_heap)(SIZE_T, UINT64 *, void **, UINT *, UINT *) =
                (void *)GetProcAddress(m, "ReboxShadowHeap");
        D3D12_ROOT_PARAMETER cparam = { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                                        .Constants = { 0, 0, 2 } };
        D3D12_ROOT_SIGNATURE_DESC crsd = { 1, &cparam };
        void *crs = NULL, *cpso = NULL, *mem = NULL;
        ID3D10Blob *crs_blob = NULL;
        BYTE *shadow = NULL;
        UINT64 mem_gpu = 0;
        UINT32 ptr[2];

        out[0] = 0;
        hr = galloc(256, 256, &mem, &mem_gpu);
        if (SUCCEEDED(hr))
            hr = shadow_heap(cpu, NULL, (void **)&shadow, NULL, NULL);
        if (SUCCEEDED(hr) && SUCCEEDED(hr = serialize(&crsd, D3D_ROOT_SIGNATURE_VERSION_1, &crs_blob, NULL)))
            hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                    ID3D10Blob_GetBufferPointer(crs_blob), ID3D10Blob_GetBufferSize(crs_blob), &IID_ID3D12RootSignature,
                    &crs);
        if (SUCCEEDED(hr))
        {
            cd.pRootSignature = crs;
            hr = XSLOT(dev, 11, HRESULT, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &cd,
                    &IID_ID3D12PipelineState, &cpso);
        }
        if (SUCCEEDED(hr))
            hr = ID3D12CommandAllocator_Reset(alloc);
        if (SUCCEEDED(hr))
            hr = XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
        CHECK(SUCCEEDED(hr), "objects for the deferred dispatch through GPU memory (%#lx)", hr);
        if (SUCCEEDED(hr))
        {
            memcpy(mem, shadow + inc, 16);
            ptr[0] = (UINT32)mem_gpu;
            ptr[1] = (UINT32)(mem_gpu >> 32);
            XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
            XSLOT(list, 33, void, void *)(list, crs);
            XSLOT(list, 25, void, void *)(list, cpso);
            XSLOT(list, 39, void, UINT, UINT, const void *, UINT)(list, 0, 2, ptr, 0);
            XSLOT(list, 14, void, UINT, UINT, UINT)(list, 1, 1, 1);
            hr = XSLOT(list, 9, HRESULT)(list);
            XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
            XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 2);
            for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 2; i++)
                Sleep(10);
            CHECK(out[0] == 0x1234abcd, "a deferred console shader whose V# pointer leads into ReboxGpuAlloc "
                  "memory wrote %#x (expected 0x1234abcd)", out[0]);
        }

        /* The same code in a console container, created as an ordinary compute PSO: d3d12_xs
         * turns its FSSM into a deferred console shader (console_fssm.c). */
        {
            void *fpso = NULL;
            BYTE *container = NULL;
            SIZE_T csize = make_console_container(rdna_uav_write, ARRAYSIZE(rdna_uav_write), 2 << 1, &container);

            out[0] = 0;
            cd.pRootSignature = rs;
            cd.CS = (D3D12_SHADER_BYTECODE){ container, csize };
            hr = XSLOT(dev, 11, HRESULT, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &cd,
                    &IID_ID3D12PipelineState, &fpso);
            CHECK(SUCCEEDED(hr), "a compute PSO from a console FSSM container (%#lx)", hr);
            if (SUCCEEDED(hr) && SUCCEEDED(ID3D12CommandAllocator_Reset(alloc))
                    && SUCCEEDED(XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL)))
            {
                XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
                XSLOT(list, 33, void, void *)(list, rs);
                XSLOT(list, 25, void, void *)(list, fpso);
                XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
                XSLOT(list, 14, void, UINT, UINT, UINT)(list, 1, 1, 1);
                XSLOT(list, 9, HRESULT)(list);
                XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
                XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 3);
                for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 3; i++)
                    Sleep(10);
                CHECK(out[0] == 0x1234abcd, "the console container's FSSM code ran through AnyPS5 at dispatch "
                      "(wrote %#x)", out[0]);
            }
            if (fpso) XSLOT(fpso, 2, ULONG)(fpso);
            free(container);
        }

        /* The same code in an FSPX container (FidelityFX's console upscaler). */
        {
            void *fpso = NULL;
            BYTE *container = NULL;
            SIZE_T csize = make_fspx_container(rdna_uav_write, ARRAYSIZE(rdna_uav_write), 2 << 1, &container);

            out[0] = 0;
            cd.pRootSignature = rs;
            cd.CS = (D3D12_SHADER_BYTECODE){ container, csize };
            hr = XSLOT(dev, 11, HRESULT, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &cd,
                    &IID_ID3D12PipelineState, &fpso);
            CHECK(SUCCEEDED(hr), "a compute PSO from a console FSPX container (%#lx)", hr);
            if (SUCCEEDED(hr) && SUCCEEDED(ID3D12CommandAllocator_Reset(alloc))
                    && SUCCEEDED(XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL)))
            {
                XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
                XSLOT(list, 33, void, void *)(list, rs);
                XSLOT(list, 25, void, void *)(list, fpso);
                XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
                XSLOT(list, 14, void, UINT, UINT, UINT)(list, 1, 1, 1);
                XSLOT(list, 9, HRESULT)(list);
                XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
                XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 4);
                for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 4; i++)
                    Sleep(10);
                CHECK(out[0] == 0x1234abcd, "the FSPX container's code ran through AnyPS5 at dispatch (wrote %#x)",
                      out[0]);
            }
            if (fpso) XSLOT(fpso, 2, ULONG)(fpso);
            free(container);
        }
        if (cpso) XSLOT(cpso, 2, ULONG)(cpso);
        if (crs) XSLOT(crs, 2, ULONG)(crs);
        if (crs_blob) ID3D10Blob_Release(crs_blob);
    }

    /* A loop that would run four billion times (s8 counts up from 1 until it wraps to 0) goes
     * back 2^20 times and leaves at the exit test at its top (patches/anyps5/0012, on by
     * default): its s_add runs 2^20 + 1 times, so it writes 0x100002, and it reports a loop-limit
     * fault. */
    {
        static const UINT32 endless[] = {
            0xba80f814, 0x00001000, 0xba807815, 0x00000002, 0xf4080100, 0xfa000000,
            0xbe880381, /* s_mov_b32 s8, 1 */
            0xbf8cc07f,
            0x80088108, /* s_add_u32 s8, s8, 1 */
            0xbf078008, /* s_cmp_lg_u32 s8, 0 */
            0xbf85fffd, /* s_cbranch_scc1 (the s_add) */
            0x7e020208, /* v_mov_b32 v1, s8 */
            0x7e040280, 0xe0701000, 0x80010102, 0xbf810000,
        };
        struct
        {
            UINT32 magic, version, size, stage, wave_size, workgroup_size[3], rsrc1, rsrc2, code_offset, code_dwords;
            UINT64 code_address;
            UINT32 code[ARRAYSIZE(endless)];
        } lblob = { 0x44524353, 1, sizeof(lblob), 0x20, 32, { 1, 1, 1 }, 0x00ac0000, 2 << 1,
                    offsetof(typeof(lblob), code), ARRAYSIZE(endless), 0x210000 };
        UINT (WINAPI *console_faults)(void) = (void *)GetProcAddress(m, "ReboxConsoleFaults");
        const char *guard = getenv("APS5_LOOP_GUARD");
        UINT faults = console_faults ? console_faults() : 0;
        void *lpso = NULL;

        memcpy(lblob.code, endless, sizeof(endless));
        out[0] = 0;
        cd.pRootSignature = rs;
        cd.CS = (D3D12_SHADER_BYTECODE){ &lblob, sizeof(lblob) };
        hr = guard && strtoul(guard, NULL, 0) != 0x100000 ? E_ABORT
             : XSLOT(dev, 11, HRESULT, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **)(dev, &cd,
                    &IID_ID3D12PipelineState, &lpso);
        if (hr == E_ABORT)
            printf("skip the loop guard test: APS5_LOOP_GUARD=%s\n", guard);
        else if (SUCCEEDED(hr) && SUCCEEDED(ID3D12CommandAllocator_Reset(alloc))
                && SUCCEEDED(XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL)))
        {
            XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
            XSLOT(list, 33, void, void *)(list, rs);
            XSLOT(list, 25, void, void *)(list, lpso);
            XSLOT(list, 35, void, UINT, UINT64)(list, 0, gpu + inc);
            XSLOT(list, 14, void, UINT, UINT, UINT)(list, 1, 1, 1);
            XSLOT(list, 9, HRESULT)(list);
            XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
            XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 5);
            for (int i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 5; i++)
                Sleep(10);
            CHECK(ID3D12Fence_GetCompletedValue(fence) >= 5 && out[0] == 0x100002,
                  "an endless console loop stops after 2^20 iterations (wrote %#x, expected 0x100002)", out[0]);
            CHECK(console_faults && console_faults() > faults, "the stopped loop reports a loop-limit fault");
        }
        else
            CHECK(0, "objects for the loop guard test (%#lx)", hr);
        if (lpso) XSLOT(lpso, 2, ULONG)(lpso);
    }
    ID3D12Resource_Unmap(out_buf, 0, NULL);

done:
    if (fence) ID3D12Fence_Release(fence);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (out_buf) ID3D12Resource_Release(out_buf);
    if (pso) XSLOT(pso, 2, ULONG)(pso);
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    if (rs_blob) ID3D10Blob_Release(rs_blob);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}
/* XCRDAPI.dll (DirectStorage): a plain and a zlib read from a file queue, a copy through a
 * memory queue, each followed by a status entry. Slots as shims/xcrdapi/xcrdapi.c names them. */
static void test_xcrdapi(void)
{
    static const BYTE zlib_data[] = {
        0x78,0xda,0x2b,0x4a,0x4d,0xca,0xaf,0x50,0x70,0xc9,0x2c,0x4a,0x4d,0x2e,0x09,0x2e,0xc9,0x2f,0x4a,0x4c,0x4f,
        0x55,0xa8,0xca,0xc9,0x4c,0x52,0x28,0x1a,0xd9,0x12,0x00,0xcd,0x0b,0x69,0x4c };
    static const char unit[] = "rebox DirectStorage zlib ";
    struct { UINT64 source_type; UINT16 capacity; INT8 priority; const char *name; } fdesc = { 0, 128 }, mdesc = { 1, 128 };
    struct req { UINT64 options; void *dst; UINT64 page; UINT32 dst_size, reserved; void *src; UINT64 offset;
                 UINT32 src_size, intermediate; UINT64 tag; const char *name; };
    typedef HRESULT (WINAPI *get_factory_fn)(REFIID, void **);
    HMODULE m = LoadLibraryA("XCRDAPI.dll");
    get_factory_fn get_factory = m ? (get_factory_fn)(void *)GetProcAddress(m, "DStorageGetFactory") : NULL;
    void *factory = NULL, *fq = NULL, *mq = NULL, *file = NULL, *status = NULL;
    WCHAR path[MAX_PATH], dir[MAX_PATH];
    BYTE plain[64], inflated[280 + 16] = { 0 }, copy[16];
    HANDLE h;
    DWORD written;
    BOOL ok;

    CHECK(get_factory != NULL, "XCRDAPI.dll exports DStorageGetFactory");
    if (!get_factory) return;
    CHECK(SUCCEEDED(get_factory(&IID_IUnknown, &factory)) && factory, "DStorageGetFactory");
    GetTempPathW(MAX_PATH, dir);
    GetTempFileNameW(dir, L"dst", 0, path);
    h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
    for (int i = 0; i < 64; i++) { BYTE b = (BYTE)(i * 3); WriteFile(h, &b, 1, &written, NULL); }
    WriteFile(h, zlib_data, sizeof(zlib_data), &written, NULL);
    CloseHandle(h);

    ok = SUCCEEDED(XSLOT(factory, 3, HRESULT, const void *, REFIID, void **)(factory, &fdesc, &IID_IUnknown, &fq))
         && SUCCEEDED(XSLOT(factory, 3, HRESULT, const void *, REFIID, void **)(factory, &mdesc, &IID_IUnknown, &mq))
         && SUCCEEDED(XSLOT(factory, 4, HRESULT, const WCHAR *, REFIID, void **)(factory, path, &IID_IUnknown, &file))
         && SUCCEEDED(XSLOT(factory, 5, HRESULT, UINT32, const char *, REFIID, void **)(factory, 4, "test",
                                                                                         &IID_IUnknown, &status));
    CHECK(ok, "DirectStorage factory creates queues, a file and a status array");
    /* Slot 10 of the newer factory: CreateQueue with a 0x20-byte desc (a u64 after the 0x18). */
    {
        struct { UINT64 source_type; UINT16 capacity; INT8 priority; const char *name; UINT64 extra; } d1 = { 1, 128, 0, NULL, 1 };
        void *q1 = NULL;
        CHECK(SUCCEEDED(XSLOT(factory, 10, HRESULT, const void *, REFIID, void **)(factory, &d1, &IID_IUnknown, &q1)) && q1,
              "DirectStorage factory slot 10 creates a queue from the newer 0x20-byte desc");
        if (q1) XSLOT(q1, 2, ULONG)(q1);
    }
    if (ok)
    {
        struct req r1 = { 0, plain, 0, sizeof(plain), 0, file, 0, 64 };
        struct req r2 = { 1 /* zlib */, inflated, 0, 280, 0, file, 64, sizeof(zlib_data) };
        struct req r3 = { 1ull << 22 /* source memory */, copy, 0, sizeof(copy), 0, (void *)"0123456789abcdef", 0, 16 };
        BOOL plain_ok = TRUE, inflated_ok = TRUE;

        XSLOT(fq, 3, void, const struct req *)(fq, &r1);
        XSLOT(fq, 4, void, void *, UINT32)(fq, status, 0);
        XSLOT(fq, 3, void, const struct req *)(fq, &r2);
        XSLOT(fq, 4, void, void *, UINT32)(fq, status, 1);
        XSLOT(fq, 6, void)(fq);
        XSLOT(mq, 3, void, const struct req *)(mq, &r3);
        XSLOT(mq, 4, void, void *, UINT32)(mq, status, 2);
        XSLOT(mq, 6, void)(mq);
        for (int i = 0; i < 500 && !(XSLOT(status, 3, BOOL, UINT32)(status, 0) && XSLOT(status, 3, BOOL, UINT32)(status, 1)
                                     && XSLOT(status, 3, BOOL, UINT32)(status, 2)); i++)
            Sleep(10);
        for (int i = 0; i < 64; i++) plain_ok &= plain[i] == (BYTE)(i * 3);
        for (int i = 0; i < 280; i++) inflated_ok &= inflated[i] == (BYTE)unit[i % (sizeof(unit) - 1)];
        CHECK(XSLOT(status, 4, HRESULT, UINT32)(status, 0) == S_OK && plain_ok, "DirectStorage file read");
        CHECK(XSLOT(status, 4, HRESULT, UINT32)(status, 1) == S_OK && inflated_ok, "DirectStorage zlib read");
        CHECK(XSLOT(status, 4, HRESULT, UINT32)(status, 2) == S_OK && !memcmp(copy, "0123456789abcdef", 16),
              "DirectStorage memory copy");
        CHECK(XSLOT(status, 3, BOOL, UINT32)(status, 3), "an unused status entry reads complete");

        /* A queue never submitted submits on its own after a short wait (a UE5 title never calls
         * Submit): a request and an event through slot 5 (EnqueueSetEvent), with no Submit. */
        {
            BYTE again[16] = { 0 };
            struct req r4 = { 1ull << 22, again, 0, sizeof(again), 0, (void *)"fedcba9876543210", 0, 16 };
            HANDLE done = CreateEventA(NULL, TRUE, FALSE, NULL);
            void *uq = NULL;

            if (SUCCEEDED(XSLOT(factory, 3, HRESULT, const void *, REFIID, void **)(factory, &mdesc, &IID_IUnknown, &uq)))
            {
                XSLOT(uq, 3, void, const struct req *)(uq, &r4);
                XSLOT(uq, 5, void, HANDLE, UINT64)(uq, done, 0);
            }
            CHECK(uq && WaitForSingleObject(done, 5000) == WAIT_OBJECT_0 && !memcmp(again, "fedcba9876543210", 16),
                  "DirectStorage runs requests without Submit and sets an event from slot 5 (EnqueueSetEvent)");
            CloseHandle(done);
            if (uq) XSLOT(uq, 2, ULONG)(uq);
        }

        /* A queue the title submits keeps what it enqueues until Submit. */
        {
            BYTE later[16] = { 0 };
            struct req r6 = { 1ull << 22, later, 0, sizeof(later), 0, (void *)"waits for Submit", 0, 16 };
            HANDLE done = CreateEventA(NULL, TRUE, FALSE, NULL);
            BOOL waited;

            XSLOT(mq, 3, void, const struct req *)(mq, &r6);
            XSLOT(mq, 5, void, HANDLE, UINT64)(mq, done, 0);
            waited = WaitForSingleObject(done, 200) == WAIT_TIMEOUT && !later[0];
            XSLOT(mq, 6, void)(mq);
            CHECK(waited && WaitForSingleObject(done, 5000) == WAIT_OBJECT_0 && !memcmp(later, "waits for Submit", 16),
                  "DirectStorage runs a submitted queue's requests at Submit, not at enqueue");
            CloseHandle(done);
        }

        /* A zlib read into physical pages (options bit 21): the destination is a page array, the
         * offset into the first page at +0x10; the data straddles two mapped pages. */
        {
            HMODULE xmem = LoadLibraryA("xmem.dll");
            BOOL (WINAPI *alloc_pages)(ULONGLONG, ULONG_PTR *, ULONG_PTR *) =
                    xmem ? (void *)GetProcAddress(xmem, "XMemAllocatePhysicalPages") : NULL;
            BOOL (WINAPI *map_pages)(void *, ULONG_PTR, ULONG_PTR *) =
                    xmem ? (void *)GetProcAddress(xmem, "XMemMapPhysicalPages") : NULL;
            ULONG_PTR pages[2], count = 2;
            BYTE *view = VirtualAlloc(NULL, 2 << 16, MEM_RESERVE, PAGE_READWRITE);
            HANDLE done = CreateEventA(NULL, TRUE, FALSE, NULL);
            BOOL paged_ok = alloc_pages && map_pages && view && alloc_pages(0, &count, pages) && map_pages(view, 2, pages);

            if (paged_ok)
            {
                struct req r5 = { 1 | 1ull << 21, pages, 0x10000 - 100, 280, 0, file, 64, sizeof(zlib_data) };
                XSLOT(fq, 3, void, const struct req *)(fq, &r5);
                XSLOT(fq, 5, void, HANDLE, UINT64)(fq, done, 0);
                XSLOT(fq, 6, void)(fq);
                paged_ok = WaitForSingleObject(done, 5000) == WAIT_OBJECT_0;
                for (int i = 0; paged_ok && i < 280; i++)
                    paged_ok &= view[0x10000 - 100 + i] == (BYTE)unit[i % (sizeof(unit) - 1)];
            }
            CHECK(paged_ok, "DirectStorage zlib read into two physical pages");
            CloseHandle(done);

            /* A title's page table, filled for the batch between enqueue and Submit: the entry held
             * another page at enqueue (one that slot had before), which must stay untouched. */
            {
                ULONG_PTR two[2], n = 2, *table = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*table));
                BYTE *old_view = VirtualAlloc(NULL, 1 << 16, MEM_RESERVE, PAGE_READWRITE);
                BYTE *new_view = VirtualAlloc(NULL, 1 << 16, MEM_RESERVE, PAGE_READWRITE);
                BOOL table_ok = alloc_pages && map_pages && table && old_view && new_view && alloc_pages(0, &n, two)
                                && map_pages(old_view, 1, &two[0]) && map_pages(new_view, 1, &two[1]);

                done = CreateEventA(NULL, TRUE, FALSE, NULL);
                if (table_ok)
                {
                    struct req r7 = { 1 | 1ull << 21, table, 0, 280, 0, file, 64, sizeof(zlib_data) };
                    table[0] = two[0];
                    XSLOT(fq, 3, void, const struct req *)(fq, &r7);
                    XSLOT(fq, 5, void, HANDLE, UINT64)(fq, done, 0);
                    Sleep(50);
                    table[0] = two[1];
                    XSLOT(fq, 6, void)(fq);
                    table_ok = WaitForSingleObject(done, 5000) == WAIT_OBJECT_0;
                    for (int i = 0; table_ok && i < 280; i++)
                        table_ok &= new_view[i] == (BYTE)unit[i % (sizeof(unit) - 1)] && !old_view[i];
                }
                CHECK(table_ok, "DirectStorage reads a page table filled after enqueue as the request runs");
                CloseHandle(done);
                HeapFree(GetProcessHeap(), 0, table);
            }
        }
    }
    if (file) XSLOT(file, 2, ULONG)(file);
    if (status) XSLOT(status, 2, ULONG)(status);
    if (fq) XSLOT(fq, 2, ULONG)(fq);
    if (mq) XSLOT(mq, 2, ULONG)(mq);
    DeleteFileW(path);
}

/* BCPack through XCRDAPI.dll, with the synthetic tables of the format notes (uniform models, pair
 * tables giving c%8 and c/8%8; no Microsoft data), written here and named by REBOX_BCPACK_TABLES:
 * a BC4 request from a file offset, zlib then BC4 with an intermediate size, and an empty BC7
 * stream, which fails. The decoder itself is tested by tests/bcpack_test.c. */
static void test_xcrdapi_bcpack(void)
{
    static const char vector_hex[] =
        "0000000000000000050505050404040401010101101010100201000000000000"
        "000014052068b0f803030202242424240200000000000000000000000404040410"
        "101010000000002068b0f8242424240000000000000000";
    static const BYTE expected[16] = { 0x8a, 0x7d, 0xd0, 0x58, 0x3f, 0xd0, 0x58, 0x3f,
                                       0x8b, 0x7e, 0x1a, 0xeb, 0x07, 0x1a, 0xeb, 0x07 };
    struct { UINT64 source_type; UINT16 capacity; INT8 priority; const char *name; } fdesc = { 0, 128 }, mdesc = { 1, 128 };
    struct req { UINT64 options; void *dst; UINT64 page; UINT32 dst_size, reserved; void *src; UINT64 offset;
                 UINT32 src_size, intermediate; UINT64 tag; const char *name; };
    HMODULE m = LoadLibraryA("XCRDAPI.dll");
    HRESULT (WINAPI *get_factory)(REFIID, void **) = m ? (void *)GetProcAddress(m, "DStorageGetFactory") : NULL;
    void *factory = NULL, *fq = NULL, *mq = NULL, *file = NULL, *status = NULL;
    char tables_path[MAX_PATH], dir[MAX_PATH];
    WCHAR chunk_path[MAX_PATH], wdir[MAX_PATH];
    BYTE *tables = calloc(1, 77982), vector[88], zlib[2 + 5 + 88 + 4], out1[16], out2[16], out3[16];
    HANDLE h;
    DWORD written;

    if (!get_factory || !tables)
    {
        CHECK(0, "XCRDAPI.dll for BCPack");
        free(tables);
        return;
    }
    for (unsigned i = 0; i < 88; i++)
        sscanf(vector_hex + 2 * i, "%2hhx", &vector[i]);
    for (unsigned mdl = 0; mdl < 128; mdl++)
        for (unsigned b = 0; b <= mdl / 16 + 1; b++)
        {
            unsigned f = 1u << (15 - mdl / 16);
            tables[18 * mdl + 2 * b] = (BYTE)f, tables[18 * mdl + 2 * b + 1] = (BYTE)(f >> 8);
        }
    for (unsigned c = 0; c < 64; c++)
        tables[2560 + c] = tables[2880 + c] = tables[2944 + c] = (BYTE)((c & 7) | (c >> 3) << 4);
    tables[3008] = 0x46;
    GetTempPathA(MAX_PATH, dir);
    GetTempFileNameA(dir, "bct", 0, tables_path);
    h = CreateFileA(tables_path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
    WriteFile(h, tables, 77982, &written, NULL);
    CloseHandle(h);
    SetEnvironmentVariableA("REBOX_BCPACK_TABLES", tables_path);

    /* The vector at offset 100 of a file, and as a stored zlib stream after it. */
    zlib[0] = 0x78, zlib[1] = 0x01, zlib[2] = 0x01, zlib[3] = 88, zlib[4] = 0, zlib[5] = (BYTE)~88, zlib[6] = 0xff;
    memcpy(zlib + 7, vector, 88);
    memset(zlib + 95, 0, 4); /* Adler-32, not checked */
    GetTempPathW(MAX_PATH, wdir);
    GetTempFileNameW(wdir, L"bcp", 0, chunk_path);
    h = CreateFileW(chunk_path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
    for (int i = 0; i < 100; i++) { BYTE b = (BYTE)i; WriteFile(h, &b, 1, &written, NULL); }
    WriteFile(h, vector, 88, &written, NULL);
    WriteFile(h, zlib, sizeof(zlib), &written, NULL);
    CloseHandle(h);

    get_factory(&IID_IUnknown, &factory);
    if (factory && SUCCEEDED(XSLOT(factory, 3, HRESULT, const void *, REFIID, void **)(factory, &fdesc, &IID_IUnknown, &fq))
            && SUCCEEDED(XSLOT(factory, 3, HRESULT, const void *, REFIID, void **)(factory, &mdesc, &IID_IUnknown, &mq))
            && SUCCEEDED(XSLOT(factory, 4, HRESULT, const WCHAR *, REFIID, void **)(factory, chunk_path, &IID_IUnknown, &file))
            && SUCCEEDED(XSLOT(factory, 5, HRESULT, UINT32, const char *, REFIID, void **)(factory, 3, "bcpack",
                                                                                            &IID_IUnknown, &status)))
    {
        struct req r1 = { 4 << 1, out1, 0, 16, 0, file, 100, 88 };
        struct req r2 = { 1 | 4 << 1, out2, 0, 16, 0, file, 188, sizeof(zlib), 128 /* intermediate */ };
        struct req r3 = { 7 << 1 | 1ull << 22, out3, 0, 16, 0, vector, 0, 0 };
        memset(out1, 0xcc, 16), memset(out2, 0xcc, 16);
        XSLOT(fq, 3, void, const struct req *)(fq, &r1);
        XSLOT(fq, 4, void, void *, UINT32)(fq, status, 0);
        XSLOT(fq, 3, void, const struct req *)(fq, &r2);
        XSLOT(fq, 4, void, void *, UINT32)(fq, status, 1);
        XSLOT(fq, 6, void)(fq);
        XSLOT(mq, 3, void, const struct req *)(mq, &r3);
        XSLOT(mq, 4, void, void *, UINT32)(mq, status, 2);
        XSLOT(mq, 6, void)(mq);
        for (int i = 0; i < 500 && !(XSLOT(status, 3, BOOL, UINT32)(status, 0) && XSLOT(status, 3, BOOL, UINT32)(status, 1)
                                     && XSLOT(status, 3, BOOL, UINT32)(status, 2)); i++)
            Sleep(10);
        CHECK(XSLOT(status, 4, HRESULT, UINT32)(status, 0) == S_OK && !memcmp(out1, expected, 16),
              "DirectStorage BCPack BC4 from a file offset (%#lx)", XSLOT(status, 4, HRESULT, UINT32)(status, 0));
        CHECK(XSLOT(status, 4, HRESULT, UINT32)(status, 1) == S_OK && !memcmp(out2, expected, 16),
              "DirectStorage zlib, then BCPack BC4, through an intermediate size");
        CHECK(XSLOT(status, 4, HRESULT, UINT32)(status, 2) == HRESULT_FROM_WIN32(ERROR_INVALID_DATA),
              "DirectStorage: an empty BC7 stream fails its request (%#lx)", XSLOT(status, 4, HRESULT, UINT32)(status, 2));
    }
    else
        CHECK(0, "DirectStorage queues, a file and a status array for BCPack");
    if (file) XSLOT(file, 2, ULONG)(file);
    if (status) XSLOT(status, 2, ULONG)(status);
    if (fq) XSLOT(fq, 2, ULONG)(fq);
    if (mq) XSLOT(mq, 2, ULONG)(mq);
    DeleteFileW(chunk_path);
    DeleteFileA(tables_path);
    free(tables);
}

/* An audio state monitor looked up by name, as GameChat2.dll does: Wine's aborts the process; ours
 * (through the GetProcAddress replacement) says full volume. */
static void test_audio_state_monitor(void)
{
    HMODULE m = LoadLibraryA("windows.media.mediacontrol.dll");
    HRESULT (WINAPI *create)(UINT, void **) = m ? (void *)GetProcAddress(m, "CreateRenderAudioStateMonitorForCategory") : NULL;
    void *monitor = NULL;
    INT64 handle = 0;

    if (!m)
    {
        printf("skip audio state monitor: no windows.media.mediacontrol.dll in this Wine\n");
        return;
    }
    CHECK(create && SUCCEEDED(create(3 /* GameChat */, &monitor)) && monitor,
          "CreateRenderAudioStateMonitorForCategory by GetProcAddress gives a monitor (not Wine's abort)");
    if (!monitor) return;
    CHECK(XSLOT(monitor, 3, HRESULT, void *, void *, INT64 *)(monitor, NULL, NULL, &handle) == S_OK && handle
              && XSLOT(monitor, 5, UINT)(monitor) == 2,
          "the monitor registers a callback and reports full volume");
    XSLOT(monitor, 4, void, INT64)(monitor, handle);
    XSLOT(monitor, 2, ULONG)(monitor);
}

/* gameinput.dll (v0): the gamepad is enumerated as connected, readings carry its state, and
 * other kinds have none. */
static void *gi_seen_device;
static void WINAPI gi_device_callback(UINT64 token, void *context, void *device, UINT64 timestamp, UINT32 now,
                                      UINT32 before)
{
    if (now & 1)
        gi_seen_device = device;
}

static void test_gameinput(void)
{
    typedef HRESULT (WINAPI *create_fn)(void **);
    HMODULE m = LoadLibraryA("gameinput.dll");
    create_fn create = m ? (create_fn)(void *)GetProcAddress(m, "GameInputCreate") : NULL;
    void *gi = NULL, *reading = NULL, *none = (void *)1;
    UINT64 token = 0;
    struct { UINT32 buttons; float lt, rt, lx, ly, rx, ry; } state;
    const BYTE *info;

    CHECK(create != NULL, "gameinput.dll exports GameInputCreate");
    if (!create || FAILED(create(&gi)) || !gi) return;
    CHECK(SUCCEEDED(XSLOT(gi, 9, HRESULT, void *, UINT32, UINT32, UINT32, void *, void *, UINT64 *)(gi, NULL,
              0x00040000, 1, 2 /* blocking */, NULL, (void *)gi_device_callback, &token)) && gi_seen_device,
          "GameInput enumerates a connected gamepad (%p)", gi_seen_device);
    if (!gi_seen_device) return;
    info = XSLOT(gi_seen_device, 3, const BYTE *)(gi_seen_device);
    CHECK(info && *(const UINT32 *)info == 328 && (*(const UINT32 *)(info + 104) & 0x00040000)
              && XSLOT(gi_seen_device, 4, UINT32)(gi_seen_device) == 0x47,
          "the gamepad's device info and status (connected, input and output enabled, synchronized)");
    {
        void *found = NULL;
        BYTE other[32] = { 1 };
        CHECK(SUCCEEDED(XSLOT(gi, 16, HRESULT, const void *, void **)(gi, info + 32, &found)) && found == gi_seen_device,
              "GameInput FindDeviceFromId finds the gamepad by its id");
        CHECK(XSLOT(gi, 16, HRESULT, const void *, void **)(gi, other, &found) == (HRESULT)0x838a0002 && !found,
              "GameInput FindDeviceFromId finds no other device");
    }
    CHECK(SUCCEEDED(XSLOT(gi, 4, HRESULT, UINT32, void *, void **)(gi, 0x00040000, gi_seen_device, &reading)) && reading,
          "GameInput GetCurrentReading(gamepad)");
    if (reading)
    {
        memset(&state, 0xcc, sizeof(state));
        CHECK(XSLOT(reading, 22, BOOL, void *)(reading, &state) && state.lx >= -1.0f && state.lx <= 1.0f,
              "the reading has a gamepad state (buttons %#x)", state.buttons);
        /* run-smoke.sh sets REBOX_PAD_SCRIPT=0:x: X is held for the first 200 ms after the DLL loads. */
        if (getenv("REBOX_PAD_SCRIPT"))
            CHECK(state.buttons & 0x10, "REBOX_PAD_SCRIPT=0:x holds X just after load (buttons %#x)", state.buttons);
        {
            void *next = (void *)1, *again = NULL, *nav = NULL;
            UINT32 buttons = 0xffffffff;
            /* Nothing changed: no newer reading; the next current reading is a new one all the same. */
            CHECK(XSLOT(gi, 5, HRESULT, void *, UINT32, void *, void **)(gi, reading, 0x00040000, NULL, &next)
                      == (HRESULT)0x838a0003 && !next, "GameInput GetNextReading: nothing newer while the state holds");
            CHECK(SUCCEEDED(XSLOT(gi, 4, HRESULT, UINT32, void *, void **)(gi, 0x00040000, NULL, &again))
                      && XSLOT(again, 4, UINT64, UINT32)(again, 0x00040000) > XSLOT(reading, 4, UINT64, UINT32)(reading, 0x00040000)
                      && (XSLOT(again, 3, UINT32)(again) & 0x00040000),
                  "each GetCurrentReading is a new gamepad reading (a GDK title polls it)");
            if (again) XSLOT(again, 2, ULONG)(again);
            CHECK(SUCCEEDED(XSLOT(gi, 4, HRESULT, UINT32, void *, void **)(gi, 0x01000000 /* UI navigation */, NULL, &nav))
                      && nav && (XSLOT(nav, 3, UINT32)(nav) & 0x01000000) && XSLOT(nav, 24, BOOL, UINT32 *)(nav, &buttons)
                      && !(buttons & ~0xfffffu),
                  "the gamepad gives UI navigation readings (buttons %#x)", buttons);
            if (nav) XSLOT(nav, 2, ULONG)(nav);
            CHECK(XSLOT(gi, 5, HRESULT, void *, UINT32, void *, void **)(gi, NULL, 0x00040000, NULL, &next) == E_INVALIDARG,
                  "GameInput GetNextReading needs a reference reading");
        }
        XSLOT(reading, 2, ULONG)(reading);
    }
    CHECK(XSLOT(gi, 4, HRESULT, UINT32, void *, void **)(gi, 0x00000010 /* keyboard */, NULL, &none) == (HRESULT)0x838a0003
              && !none, "GameInput has no keyboard readings");
}

/* acphal.dll: the coprocessor channel acknowledges each command with its id and ticks audio
 * frames, so a title polling PopMessage for its command (a GDK title at startup) goes on. */
static void test_acphal(void)
{
    struct { UINT32 type, reserved; UINT64 value, id; } msg;
    HMODULE m = LoadLibraryA("acphal.dll");
    HRESULT (WINAPI *create)(void **) = m ? (void *)GetProcAddress(m, "AcpHalCreate") : NULL;
    void *hal = NULL;
    BOOL got_frame = FALSE;
    DWORD start;

    CHECK(create != NULL, "acphal.dll exports AcpHalCreate");
    if (!create || FAILED(create(&hal)) || !hal) return;
    CHECK(XSLOT(hal, 2, HRESULT, UINT32, UINT64, UINT32, const void *, UINT32)(hal, 2, 1, ~0u, NULL, 0) == E_UNEXPECTED,
          "IAcpHal SubmitCommand fails before Connect");
    CHECK(XSLOT(hal, 0, HRESULT, UINT32, UINT32)(hal, 0x804, 0x804) == S_OK, "IAcpHal Connect");
    CHECK(XSLOT(hal, 2, HRESULT, UINT32, UINT64, UINT32, const void *, UINT32)(hal, 2, 1, ~0u, NULL, 0) == S_OK
              && XSLOT(hal, 2, HRESULT, UINT32, UINT64, UINT32, const void *, UINT32)(hal, 3, 0, ~0u, NULL, 0) == S_OK,
          "IAcpHal SubmitCommand");
    CHECK(XSLOT(hal, 4, UINT32)(hal) >= 1, "IAcpHal GetNumMessages counts the done command");
    memset(&msg, 0xcc, sizeof(msg));
    CHECK(XSLOT(hal, 3, BOOL, void *)(hal, &msg) && msg.type == 0x10 && msg.id == 1,
          "IAcpHal PopMessage gives command 1 done (type %#x, id %#llx)", msg.type, (unsigned long long)msg.id);
    /* Command id 0 has no message; then an audio frame comes within a few ms. */
    for (start = GetTickCount(); !got_frame && GetTickCount() - start < 1000;)
    {
        if (XSLOT(hal, 3, BOOL, void *)(hal, &msg))
        {
            CHECK(msg.type == 2, "only audio frames follow (type %#x)", msg.type);
            got_frame = msg.type == 2;
        }
        else
            Sleep(1);
    }
    CHECK(got_frame, "IAcpHal PopMessage ticks audio frames");
    CHECK(XSLOT(hal, 1, HRESULT)(hal) == S_OK, "IAcpHal Disconnect");
    while (XSLOT(hal, 3, BOOL, void *)(hal, &msg) && msg.type == 2)
        ;
    CHECK(msg.type == 0x80 && !XSLOT(hal, 3, BOOL, void *)(hal, &msg), "IAcpHal says it disconnected, then nothing");
    XSLOT(hal, 5, void)(hal);
}

/* Sends one coprocessor command with a two-word payload. */
static void xma_send(void *hal, UINT32 type, UINT32 voice, UINT32 value)
{
    UINT32 data[2] = { voice, value };
    XSLOT(hal, 2, HRESULT, UINT32, UINT64, UINT32, const void *, UINT32)(hal, type, type == 18 ? 0xf1f1f1f1 : 0, ~0u, data, 0);
}

/* Takes what a voice's ring of `size` blocks gives, as the title's mixer does: whatever lies
 * between its read block and the write block, or all of it when the ring is full, then command
 * 18. Stops after `limit` blocks, or when the ring stays empty. */
static UINT32 xma_take(void *hal, volatile UINT32 *ctx, UINT32 voice, UINT32 size, UINT32 limit)
{
    UINT32 taken = 0;
    DWORD start = GetTickCount();

    while (GetTickCount() - start < 5000 && taken < limit)
    {
        UINT32 blocks;
        DWORD wait = GetTickCount();
        /* The title doesn't look at the end bit: a ring with nothing new for 300 ms is the end. */
        while (GetTickCount() - wait < 300 && (ctx[1] >> 31) && (ctx[0] >> 27) == (ctx[9] & 31))
            Sleep(1);
        blocks = (ctx[1] >> 31) ? ((ctx[0] >> 27) - (ctx[9] & 31) + size) % size : size;
        if (!blocks)
            break;
        if (blocks > limit - taken)
            blocks = limit - taken;
        taken += blocks;
        xma_send(hal, 18, voice, blocks);
    }
    return taken;
}

/* A packet of `frames` frames of `bits` bits each, the first right after the header. */
static void xma_packet(BYTE *packet, UINT frames, UINT bits)
{
    memset(packet, 0, 2048);
    packet[0] = (BYTE)(frames << 2);
    packet[2] = 1; /* metadata 1, as the title's packets have */
    for (UINT i = 0, at = 32; i < frames; i++, at += bits)
        for (UINT b = 0; b < 15; b++)
            if (bits >> (14 - b) & 1)
                packet[(at + b) / 8] |= 0x80 >> ((at + b) % 8);
}

/* acphal.dll's XMA decoder, driven as the title's mixer drives it (its words checked in its
 * code): an enabled context's frames fill its output ring at the stream's pace (decoded, or
 * silence for frames that don't decode); the ring waits when full until command 18 moves the
 * read block on; the title counts the samples it takes against the input's length in frames, so
 * the voice must give every frame from where it starts, never fewer. */
static void test_acphal_xma(void)
{
    HMODULE m = LoadLibraryA("acphal.dll");
    HRESULT (WINAPI *create)(void **) = m ? (void *)GetProcAddress(m, "AcpHalCreate") : NULL;
    HRESULT (WINAPI *alloc)(void **, UINT32 *, UINT32, UINT32, UINT32) = m ? (void *)GetProcAddress(m, "ApuAlloc") : NULL;
    HRESULT (WINAPI *shape)(BYTE *) = m ? (void *)GetProcAddress(m, "AcpHalAllocateShapeContexts") : NULL;
    UINT32 (WINAPI *map)(const void *) = m ? (void *)GetProcAddress(m, "ApuMapVirtualAddress") : NULL;
    BYTE request[0x48] = { 0 };
    volatile UINT32 *ctx = NULL;
    UINT32 in_apu = 0, in1_apu = 0, out_apu = 0, taken, first;
    BYTE *input = NULL, *input1 = NULL, *output = NULL;
    void *hal = NULL;

    if (!create || !alloc || !shape || !map || FAILED(create(&hal)) || !hal)
    {
        CHECK(0, "acphal.dll exports what the XMA test needs");
        return;
    }
    ((UINT32 *)request)[4] = 4; /* four XMA contexts */
    CHECK(shape(request) == S_OK && (ctx = *(UINT32 **)(request + 0x18 + 8 * 4)) != NULL, "AcpHalAllocateShapeContexts(4 XMA)");
    if (!ctx) return;
    ctx += 16; /* context 1 */
    CHECK(alloc((void **)&input, &in_apu, 3 * 2048, 0x800, 0) == S_OK && alloc((void **)&input1, &in1_apu, 2 * 2048, 0x800, 0) == S_OK
              && alloc((void **)&output, &out_apu, 8 * 256, 0x100, 0) == S_OK
              && in_apu && map(input) == in_apu && map(output + 0x10) == out_apu + 0x10,
          "ApuAlloc's APU addresses are not 0 and ApuMapVirtualAddress agrees");
    CHECK(XSLOT(hal, 0, HRESULT, UINT32, UINT32)(hal, 64, 64) == S_OK, "IAcpHal Connect");

    /* 1. Two packets of three frames (lengths unreadable: zeros), mono, a ring of 4 blocks:
     * 6 frames are 24 blocks of 128 samples, and the end adds a frame of silence (4 blocks). */
    memset(input, 0, 2 * 2048);
    input[0] = input[2048] = 3 << 2;
    memset(output, 0x55, 4 * 256);
    ctx[0] = 2 | 1u << 20 | 4u << 22;
    ctx[1] = 3u << 27 | 1u << 31;
    ctx[2] = ctx[3] = ctx[4] = ctx[9] = 0;
    ctx[5] = ctx[6] = in_apu;
    ctx[7] = out_apu;
    xma_send(hal, 5, 1, 0);
    first = xma_take(hal, ctx, 1, 4, 4);
    CHECK(first == 4 && output[0] != 0x55 && output[3 * 256 + 255] != 0x55,
          "the ring fills: 4 blocks written, then it waits for the title (%u)", first);
    taken = first + xma_take(hal, ctx, 1, 4, 1000);
    CHECK(taken == 28, "command 18 frees the ring: two packets of 3 frames made %u blocks (28)", taken);
    CHECK((ctx[2] >> 31) && ((ctx[2] >> 26) & 31) == 4 && !(ctx[0] & 3u << 20),
          "at the end of the input the context says so (%08x %08x)", ctx[0], ctx[2]);
    xma_send(hal, 7, 1, 0);

    /* 2. A stream across both inputs (3 packets, then 2, of four 4000-bit frames each), started
     * as the title starts a voice part-way in: at frame 2 (word 2: its bit position) with one
     * subframe skipped. 18 frames less 128 samples, plus the end's frame: 75 blocks. */
    for (int i = 0; i < 3; i++)
        xma_packet(input + i * 2048, 4, 4000);
    for (int i = 0; i < 2; i++)
        xma_packet(input1 + i * 2048, 4, 4000);
    ctx[0] = 3 | 3u << 20 | 8u << 22;
    ctx[1] = 2 | 1u << 24 | 1u << 27 | 1u << 31;
    ctx[2] = 32 + 2 * 4000;
    ctx[3] = ctx[4] = ctx[9] = 0;
    ctx[5] = in_apu;
    ctx[6] = in1_apu;
    ctx[7] = out_apu;
    xma_send(hal, 5, 1, 0);
    taken = xma_take(hal, ctx, 1, 8, 1000);
    CHECK(taken == 75, "a stream across both inputs, started at its third frame, made %u blocks (75)", taken);
    CHECK(!(ctx[0] & 3u << 20) && (ctx[4] >> 31) == 1 && ((ctx[2] >> 26) & 31) == 4,
          "both inputs were used, the second last (%08x %08x %08x)", ctx[0], ctx[2], ctx[4]);

    /* 3. The title stops a voice and sets it up again at once (it reuses contexts between
     * chunks): the new setup plays whole, untouched by the old one. */
    ctx[0] = 3 | 3u << 20 | 8u << 22;
    ctx[1] = 2 | 1u << 27 | 1u << 31;
    ctx[2] = 32;
    ctx[4] = ctx[9] = 0;
    xma_send(hal, 5, 1, 0);
    xma_take(hal, ctx, 1, 8, 8);
    xma_send(hal, 7, 1, 0);
    ctx[0] = 2 | 1u << 20 | 8u << 22;
    ctx[1] = 1u << 27 | 1u << 31;
    ctx[2] = 32;
    ctx[4] = ctx[9] = 0;
    xma_send(hal, 5, 1, 0);
    taken = xma_take(hal, ctx, 1, 8, 1000);
    CHECK(taken == 36, "a voice set up again at once plays its new input whole: %u blocks (36)", taken);

    /* 4. A loop played once more: from frame 1 of packet 0 to inside frame 2 of packet 1 (256
     * samples of it kept). 4 + 2 frames and 256 samples, then 3 + 4 frames, then the end's
     * frame: 58 blocks; the loop count goes to 0. */
    ctx[0] = 2 | 1u << 12 | 1u << 20 | 8u << 22;
    ctx[1] = 1u << 12 | 1u << 27 | 1u << 31;
    ctx[2] = 32;
    ctx[3] = 32 + 4000;
    ctx[4] = 16384 + 32 + 2 * 4000 + 10;
    ctx[9] = 0;
    xma_send(hal, 5, 1, 0);
    taken = xma_take(hal, ctx, 1, 8, 1000);
    CHECK(taken == 58 && !(ctx[0] >> 12 & 0xff), "a loop played once more made %u blocks (58), count %u",
          taken, ctx[0] >> 12 & 0xff);
    xma_send(hal, 7, 1, 0);

    XSLOT(hal, 1, HRESULT)(hal);
    XSLOT(hal, 5, void)(hal);
}

/* xapu.dll: the decoder queue answers each command at once, a Process with the Opus packet
 * decoded, or silence of the length asked for without one (FMOD in a GDK title drives it);
 * acphal's other FMOD imports answer too. */
/* chad.dll, the console's audio DSP: connecting says it isn't there and leaves no connection, so a
 * title's audio engine goes on without it; the rest fail or do nothing. */
static void test_chad(void)
{
    /* The title's debug text reaches the log (run-smoke.sh looks for it). */
    OutputDebugStringA("rebox smoke boot line\nsecond line\n");
    /* REBOX_WRITE_WATCH (set by run-smoke.sh for this run): a write into the watched range is let
     * through and logged with the writing instruction. */
    {
        volatile UINT64 *watched = VirtualAlloc((void *)0x7e0000000ull, 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (watched)
        {
            Sleep(400); /* the watch arms within 100 ms of the range being committed */
            watched[1] = 0x1234;
            watched[0x200] = 0x5678; /* the next page: not watched */
            CHECK(watched[1] == 0x1234 && watched[0x200] == 0x5678, "writes into a write-watched range land");
        }
    }
    /* The title's network is off by default: names other than localhost don't resolve. */
    {
        HMODULE ws = LoadLibraryA("ws2_32.dll");
        int (WINAPI *gai)(const char *, const char *, const void *, void **) =
            ws ? (void *)GetProcAddress(ws, "getaddrinfo") : NULL;
        void (WINAPI *fai)(void *) = ws ? (void *)GetProcAddress(ws, "freeaddrinfo") : NULL;
        int (WINAPI *startup)(WORD, void *) = ws ? (void *)GetProcAddress(ws, "WSAStartup") : NULL;
        BYTE wsa[512];
        void *res = NULL;
        int remote, local;
        if (startup)
            startup(0x202, wsa);
        remote = gai ? gai("example.com", "80", NULL, &res) : -1;
        if (!remote && res && fai) fai(res);
        res = NULL;
        local = gai ? gai("localhost", "80", NULL, &res) : -1;
        if (!local && res && fai) fai(res);
        CHECK(remote == 11001 && local == 0, "the title's network is off: example.com doesn't resolve (%d), localhost does (%d)",
              remote, local);
    }
    HMODULE m = LoadLibraryA("chad.dll");
    HRESULT (WINAPI *connect)(void *, UINT32, UINT32, UINT32, UINT32, HANDLE *) =
            m ? (void *)GetProcAddress(m, "XDspConnectWithMaximumStreamLimit") : NULL;
    HRESULT (WINAPI *activate)(HANDLE, UINT64, UINT64, UINT64) = m ? (void *)GetProcAddress(m, "XDspActivate") : NULL;
    HRESULT (WINAPI *submit)(HANDLE, const void *) = m ? (void *)GetProcAddress(m, "XDspSubmitCommand") : NULL;
    HRESULT (WINAPI *disconnect)(HANDLE) = m ? (void *)GetProcAddress(m, "XDspDisconnect") : NULL;
    HANDLE h = (HANDLE)1;
    BYTE command[64] = { 0 };

    CHECK(connect && activate && submit && disconnect && GetProcAddress(m, "XDspDeactivate"),
          "chad.dll exports the five XDsp functions a title imports");
    if (!connect || !activate || !submit || !disconnect)
        return;
    /* A title stops at a breakpoint when it can't connect: the DSP connects and takes commands. */
    h = NULL;
    CHECK(connect(command, 0, 0, 0x80, 1, &h) == S_OK && h && activate(h, 0, 0, 0) == S_OK
          && submit(h, command) == S_OK && disconnect(h) == S_OK,
          "the DSP connects, activates, takes a command and disconnects");
    CHECK(FAILED(submit(NULL, command)) && FAILED(activate(NULL, 0, 0, 0)),
          "XDsp calls without a connection fail");
}

static void test_xapu(void)
{
#pragma pack(push, 1)
    struct command { UINT32 type, stream, sequence, options; void *context; };
    struct process { struct command id; float ramp, pitch; UINT32 first_frame, frames, input_bytes; void *input;
                     UINT32 output_bytes; void *output; } p;
    struct result { struct command id; UINT32 state; UINT64 state_data, time; HRESULT error; UINT32 frames,
                    output_bytes; void *output; float pitch; UINT32 input_bytes; void *input; } r;
    struct { UINT32 process, streams, commands_per_stream, data_bytes, frames_per_output, options; } in = { 1, 2, 4, 0, 256, 0 };
    struct { void *data; HANDLE ready; } out;
#pragma pack(pop)
    HMODULE m = LoadLibraryA("xapu.dll"), acp = LoadLibraryA("acphal.dll");
    HRESULT (WINAPI *connect)(void *, void *, HANDLE *) = m ? (void *)GetProcAddress(m, "XApuConnect") : NULL;
    HRESULT (WINAPI *enqueue)(HANDLE, const void *, UINT32 *) = m ? (void *)GetProcAddress(m, "XApuEnqueueCommand") : NULL;
    HRESULT (WINAPI *dequeue)(HANDLE, void *) = m ? (void *)GetProcAddress(m, "XApuDequeueResult") : NULL;
    HRESULT (WINAPI *disconnect)(HANDLE) = m ? (void *)GetProcAddress(m, "XApuDisconnect") : NULL;
    UINT32 (WINAPI *map)(const void *) = acp ? (void *)GetProcAddress(acp, "ApuMapVirtualAddress") : NULL;
    BOOL (WINAPI *valid)(const void *, UINT32) = acp ? (void *)GetProcAddress(acp, "ApuIsVirtualAddressValid") : NULL;
    HRESULT (WINAPI *release)(void) = acp ? (void *)GetProcAddress(acp, "AcpHalReleaseShapeContexts") : NULL;
    BYTE buf[4096];
    HANDLE h = NULL;
    UINT32 n = 0, a1 = 0, a2 = 0;
    BOOL silent = TRUE;

    CHECK(map && valid && release, "acphal.dll exports FMOD's ApuMapVirtualAddress, ApuIsVirtualAddressValid and AcpHalReleaseShapeContexts");
    /* Titles pass the pointer alone and keep the result: memory outside the heap gets an address
     * too, and every byte of it maps back. */
    {
        BYTE *own = VirtualAlloc(NULL, 0x4000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        void *(WINAPI *back)(UINT32) = acp ? (void *)GetProcAddress(acp, "ApuMapApuAddress") : NULL;
        if (map && valid && release && own)
        {
            a1 = map(own);
            a2 = map(own + 0x2000);
            CHECK(a1 && a2 == a1 + 0x2000 && valid(own, 16) && release() == S_OK,
                  "ApuMapVirtualAddress(va) gives an address for the title's memory (%#x, %#x)", a1, a2);
        }
        CHECK(back && own && back(a1 + 0x10) == own + 0x10 && back(a2) == own + 0x2000 && !back(0),
              "ApuMapApuAddress maps APU addresses back to memory");
    }
    CHECK(connect && enqueue && dequeue && disconnect, "xapu.dll exports FMOD's four imports");
    if (!connect || !enqueue || !dequeue || !disconnect) return;
    CHECK(connect(&in, &out, &h) == S_OK && h, "XApuConnect");
    if (!h) return;
    memset(&p, 0, sizeof(p));
    memset(buf, 0xaa, sizeof(buf));
    p.id.type = 3;
    p.id.stream = 1;
    p.pitch = 1.0f;
    p.output_bytes = sizeof(buf);
    p.output = buf;
    CHECK(enqueue(h, &p, &n) == S_OK && n == 1, "XApuEnqueueCommand(Process)");
    memset(&r, 0, sizeof(r));
    CHECK(dequeue(h, &r) == S_OK && r.frames == 256 && r.output == buf && r.state == 3,
          "XApuDequeueResult gives the frames asked for (%u)", r.frames);
    for (int i = 0; i < 256 * 8; i++)
        silent &= buf[i] == 0;
    CHECK(silent && buf[256 * 8] == 0xaa, "the output is silence of that length");
    CHECK(dequeue(h, &r) == (HRESULT)0x8ac81001, "XApuDequeueResult says when there is nothing");
    /* An Opus packet (TOC 0xfc: CELT, 20 ms, stereo, one frame, here a quiet one) decodes to 960
     * frames at 48 kHz; from first_frame 100 on, 860 are kept. Silence would give the 1024 asked. */
    {
        static const BYTE packet[] = { 0xfc, 0xff, 0xfe };
        static float pcm[2048 * 2];
        struct { struct command id; UINT32 channels; } act;
        memset(&act, 0, sizeof(act));
        act.id.type = 1;
        act.id.stream = 0;
        act.channels = 2;
        CHECK(enqueue(h, &act, &n) == S_OK && dequeue(h, &r) == S_OK, "XApuEnqueueCommand(Activate, 2 channels)");
        memset(&p, 0, sizeof(p));
        p.id.type = 3;
        p.pitch = 1.0f;
        p.first_frame = 100;
        p.frames = 1024;
        p.input = (void *)packet;
        p.input_bytes = sizeof(packet);
        p.output_bytes = sizeof(pcm);
        p.output = pcm;
        memset(&r, 0, sizeof(r));
        CHECK(enqueue(h, &p, &n) == S_OK && dequeue(h, &r) == S_OK && r.frames == 860 && r.output_bytes == 860 * 8,
              "an Opus packet decodes: 960 frames, 860 from frame 100 (%u, %u bytes)", r.frames, r.output_bytes);
    }
    CHECK(disconnect(h) == S_OK, "XApuDisconnect");
}

/* A texture placed at a title address (slot 44) is filled from the tiled memory there: at first
 * use, and again after a reported write (DirectStorage reports its writes the same way). */
/* A texture placed on memory the title has only reserved: it is filled once the memory is
 * committed and written, after a few submissions (rechecked with backoff). */
/* Views of memory (slot 70) of sampled textures get memory of their own, kept per address and
 * description. A world streaming in reuses its texture pools: a view of another shape over the
 * same memory retires the old texture, which is released once the GPU is past it, so the memory
 * stays bounded; a view of the same shape in another format (sRGB) is kept beside it. */
static void test_view_texture_retire(void *dev, void *queue)
{
    void (WINAPI *view_memory)(UINT64 *, UINT64 *, UINT64 *, UINT64 *) =
            (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxViewMemory");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    struct { D3D12_SHADER_RESOURCE_VIEW_DESC v; UINT64 address; BYTE pad[8]; } view = { { DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING } };
    BYTE console_desc[0x40] = { 0 };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    UINT64 views0, retired0, textures0, bytes0, views, retired, textures, peak, bytes;
    ID3D12CommandAllocator *alloc = NULL;
    void *heap = NULL, *list = NULL, *memory;
    SIZE_T start;
    const int count = 64;

    if (!view_memory || !(memory = VirtualAlloc(NULL, 4 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)))
        return;
    if (FAILED(XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap)))
        return;
    start = XSLOT(heap, 9, SIZE_T)(heap);
    view.v.Texture2D.MipLevels = 1;
    view.address = (UINT64)(ULONG_PTR)memory;
    view_memory(&views0, &retired0, &textures0, &bytes0);
    /* A pool slot refilled with textures of a new size each time. */
    for (int i = 0; i < count; i++)
    {
        td.Width = 64 + 4 * i;
        memcpy(console_desc, &td, sizeof(td));
        XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start);
    }
    view_memory(&views, &retired, &peak, &bytes);
    CHECK(peak - textures0 == count && views - views0 == 1,
          "views of memory: %d textures of other shapes over the same memory leave one view kept (%llu kept, "
          "%llu of their textures alive)", count, (unsigned long long)(views - views0),
          (unsigned long long)(peak - textures0));
    /* A submission, so the GPU's progress is watched; then the retired ones are released. */
    if (SUCCEEDED(XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
            &IID_ID3D12CommandAllocator, (void **)&alloc))
            && SUCCEEDED(XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *,
                               ID3D12PipelineState *, REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc,
                               NULL, &IID_ID3D12GraphicsCommandList, &list)))
    {
        XSLOT(list, 9, HRESULT)(list);
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    }
    for (int i = 0; i < 100; i++)
    {
        view_memory(&views, &retired, &textures, &bytes);
        if (textures - textures0 <= 1)
            break;
        Sleep(100);
    }
    CHECK(textures - textures0 == 1 && retired == retired0 && bytes - bytes0 < 1 << 20,
          "views of memory: the %d retired textures are released once the GPU is past them (%llu alive, %llu waiting, "
          "%llu bytes)", count - 1, (unsigned long long)(textures - textures0), (unsigned long long)retired,
          (unsigned long long)(bytes - bytes0));
    /* The last shape again, in sRGB: kept beside it; and the first again: the same one. */
    view.v.Format = td.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    memcpy(console_desc, &td, sizeof(td));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start);
    view.v.Format = td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    memcpy(console_desc, &td, sizeof(td));
    console_desc[4] = 0x5a; /* the PC's padding: not part of the key */
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start);
    view_memory(&views, &retired, &textures, &bytes);
    CHECK(textures - textures0 == 2 && views - views0 == 2 && retired == retired0,
          "views of memory: an sRGB view of the same texture is kept beside it, and the same one again is found "
          "(%llu kept, %llu alive, %llu retired)", (unsigned long long)(views - views0),
          (unsigned long long)(textures - textures0), (unsigned long long)(retired - retired0));
    /* A small texture inside it, past its start (a view of a mip, maybe): both stay. */
    td.Width = td.Height = 16;
    memcpy(console_desc, &td, sizeof(td));
    view.address = (UINT64)(ULONG_PTR)memory + 0x4000;
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start);
    view_memory(&views, &retired, &textures, &bytes);
    CHECK(textures - textures0 == 3 && views - views0 == 3 && retired == retired0,
          "views of memory: a texture inside another, past its start, leaves it (%llu kept, %llu retired)",
          (unsigned long long)(views - views0), (unsigned long long)(retired - retired0));
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    XSLOT(heap, 2, ULONG)(heap);
    /* The memory stays: the views kept still name it. */
}

/* A retired view texture (test_view_texture_retire) whose view is still in a descriptor slot is
 * kept while any slot holds it: a copy of the view in another heap counts too. Once both slots are
 * rewritten it is released, and no slot names it. */
static void test_view_texture_held(void *dev, void *queue)
{
    void (WINAPI *view_memory)(UINT64 *, UINT64 *, UINT64 *, UINT64 *) =
            (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxViewMemory");
    ID3D12Resource *(WINAPI *slot_resource)(SIZE_T) =
            (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxShadowSlotResource");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_DESCRIPTOR_HEAP_DESC cd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    struct { D3D12_SHADER_RESOURCE_VIEW_DESC v; UINT64 address; BYTE pad[8]; } view = { { DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING } };
    D3D12_SHADER_RESOURCE_VIEW_DESC null = { DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_SRV_DIMENSION_TEXTURE2D,
                                             D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
    BYTE console_a[0x40] = { 0 }, console_b[0x40] = { 0 };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    UINT64 views0, retired0, textures0, bytes0, views, retired, textures, bytes;
    ID3D12Resource *a = NULL, *b = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    void *heap = NULL, *cpu_heap = NULL, *list = NULL, *memory;
    SIZE_T start, cpu_start;
    UINT inc;

    if (!view_memory || !slot_resource
            || !(memory = VirtualAlloc(NULL, 4 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)))
        return;
    if (FAILED(XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap))
            || FAILED(XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &cd,
            &IID_ID3D12DescriptorHeap, &cpu_heap)))
        goto done;
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    start = XSLOT(heap, 9, SIZE_T)(heap);
    cpu_start = XSLOT(cpu_heap, 9, SIZE_T)(cpu_heap);
    view.v.Texture2D.MipLevels = null.Texture2D.MipLevels = 1;
    view.address = (UINT64)(ULONG_PTR)memory;
    memcpy(console_a, &td, sizeof(td));
    td.Width = 128;
    memcpy(console_b, &td, sizeof(td));
    view_memory(&views0, &retired0, &textures0, &bytes0);

    /* A in slot 0, copied into the CPU heap; B over the same memory in slot 1 retires A. */
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_a, &view, start);
    a = slot_resource(start);
    XSLOT(dev, 24, void, UINT, SIZE_T, SIZE_T, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, 1, cpu_start, start,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_b, &view, start + inc);
    b = slot_resource(start + inc);
    if (SUCCEEDED(XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
            &IID_ID3D12CommandAllocator, (void **)&alloc))
            && SUCCEEDED(XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *,
                               ID3D12PipelineState *, REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc,
                               NULL, &IID_ID3D12GraphicsCommandList, &list)))
    {
        XSLOT(list, 9, HRESULT)(list);
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    }
    Sleep(3500);
    view_memory(&views, &retired, &textures, &bytes);
    CHECK(a && b && a != b && slot_resource(cpu_start) == a && textures - textures0 == 2 && retired - retired0 == 1,
          "views of memory: a retired texture whose view is in two slots is kept past the GPU (%llu alive, %llu "
          "waiting)", (unsigned long long)(textures - textures0), (unsigned long long)(retired - retired0));

    /* Slot 0 gets B's view: the copy still holds A. */
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_b, &view, start);
    Sleep(1000);
    view_memory(&views, &retired, &textures, &bytes);
    CHECK(slot_resource(start) == b && textures - textures0 == 2 && retired - retired0 == 1,
          "views of memory: rewriting one slot leaves the retired texture held by the copy (%llu alive)",
          (unsigned long long)(textures - textures0));

    /* The copy is rewritten with a null view: A goes, and no slot names it. */
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, NULL, &null, cpu_start);
    for (int i = 0; i < 40; i++)
    {
        view_memory(&views, &retired, &textures, &bytes);
        if (textures - textures0 <= 1)
            break;
        Sleep(100);
    }
    CHECK(textures - textures0 == 1 && retired == retired0 && slot_resource(cpu_start) != a
          && slot_resource(start) == b && slot_resource(start + inc) == b,
          "views of memory: once no slot holds it, the retired texture is released (%llu alive, %llu waiting)",
          (unsigned long long)(textures - textures0), (unsigned long long)(retired - retired0));
done:
    if (list) XSLOT(list, 2, ULONG)(list);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (cpu_heap) XSLOT(cpu_heap, 2, ULONG)(cpu_heap);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

/* View textures nothing can read are evicted (proxy.c, memory_views_evict): many made, each in a
 * descriptor slot that is then rewritten; once not asked for a while they go, past the GPU, all but
 * the one a slot still holds. A view of one asked for again makes it anew, filled from title memory,
 * and so does a T# the title wrote itself with another's address. Over a budget, ones no slot holds
 * go even if lately asked for. (run-smoke.sh turns the periodic passes off; these are called.) */
static void test_view_texture_evict(void *dev, void *queue)
{
    enum { COUNT = 32 };
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    void (WINAPI *view_memory)(UINT64 *, UINT64 *, UINT64 *, UINT64 *) = (void *)GetProcAddress(m, "ReboxViewMemory");
    UINT (WINAPI *evict)(UINT, UINT64) = (void *)GetProcAddress(m, "ReboxViewEvict");
    ID3D12Resource *(WINAPI *slot_resource)(SIZE_T) = (void *)GetProcAddress(m, "ReboxShadowSlotResource");
    void *(WINAPI *texture_of)(const UINT32 *) = (void *)GetProcAddress(m, "ReboxConsoleTextureOf");
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    tile_fn tile = (tile_fn)(void *)GetProcAddress(m, "ReboxTile");
    struct { UINT mode; DXGI_FORMAT format; UINT width, height, layers, mips; BOOL volume, mip_major; } surface =
            { 1 /* SW_4KB_S */, DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, FALSE, FALSE };
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    struct { D3D12_SHADER_RESOURCE_VIEW_DESC v; UINT64 address; BYTE pad[8]; } view = { { DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING } };
    D3D12_SHADER_RESOURCE_VIEW_DESC null = { DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_SRV_DIMENSION_TEXTURE2D,
                                             D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256 * 64, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    static UINT32 linear[64 * 64];
    ID3D12Resource *made[COUNT], *again = NULL, *revived = NULL, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    BYTE console_desc[0x40] = { 0 };
    UINT64 views, retired, textures0, peak, textures, bytes;
    void *heap = NULL, *list = NULL;
    UINT32 *out = NULL;
    BYTE *memory;
    SIZE_T start;
    UINT inc, gone = 0, value = 0;
    HRESULT hr;

    if (!view_memory || !evict || !slot_resource || !texture_of || !tile
            || !(memory = VirtualAlloc(NULL, (COUNT + 8) << 16, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)))
        return;
    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (FAILED(hr))
        goto done;
    XSLOT(list, 9, HRESULT)(list);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    start = XSLOT(heap, 9, SIZE_T)(heap);
    view.v.Texture2D.MipLevels = null.Texture2D.MipLevels = 1;
    memcpy(console_desc, &td, sizeof(td));
    for (int i = 0; i < 64 * 64; i++)
        linear[i] = i * 2246822519u + 5;
    tile(&surface, memory + (5 << 16), 0, 0, linear, 256, 256 * 64);

    /* Texture 0 stays in slot 0; the others pass through slot 1, which is then emptied. */
    view_memory(&views, &retired, &textures0, &bytes);
    for (int i = 0; i < COUNT; i++)
    {
        view.address = (UINT64)(ULONG_PTR)(memory + ((SIZE_T)i << 16));
        XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + (i ? inc : 0));
        made[i] = slot_resource(start + (i ? inc : 0));
    }
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, NULL, &null, start + inc);
    view_memory(&views, &retired, &peak, &bytes);
    {
        BOOL distinct = TRUE;
        for (int i = 0; i < COUNT; i++)
            distinct &= made[i] && (!i || made[i] != made[i - 1]);
        CHECK(distinct && peak - textures0 <= COUNT && peak - textures0 >= COUNT - 2,
              "view textures: %d made at %d addresses (%llu more alive)", COUNT, COUNT,
              (unsigned long long)(peak - textures0));
    }
    /* Lately asked for: none of them goes (older tests' may). */
    evict(10000, 0);
    view.address = (UINT64)(ULONG_PTR)(memory + (1 << 16));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 2 * inc);
    CHECK(slot_resource(start + 2 * inc) == made[1], "view textures: none is evicted while lately asked for");
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, NULL, &null,
            start + 2 * inc);
    view_memory(&views, &retired, &peak, &bytes);
    Sleep(1200);
    gone = evict(1000, 0);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    for (int i = 0; i < 100; i++)
    {
        view_memory(&views, &retired, &textures, &bytes);
        if (peak - textures >= COUNT - 1)
            break;
        Sleep(100);
    }
    CHECK(gone >= COUNT - 1 && peak - textures >= COUNT - 1 && slot_resource(start) == made[0],
          "view textures: the %d no slot holds are evicted and released past the GPU, the one in a slot stays "
          "(%u evicted, %llu released)", COUNT - 1, gone, (unsigned long long)(peak - textures));
    /* Texture 0 is still the one its view finds. */
    view.address = (UINT64)(ULONG_PTR)memory;
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 2 * inc);
    CHECK(slot_resource(start + 2 * inc) == made[0], "view textures: the one a slot holds is found again, not made anew");

    /* Texture 5's view again: made anew, filled from its memory. */
    view.address = (UINT64)(ULONG_PTR)(memory + (5 << 16));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + inc);
    again = slot_resource(start + inc);
    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    if (again)
    {
        D3D12_TEXTURE_COPY_LOCATION src = { again, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = 0 } };
        D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                            { .PlacedFootprint = { 0, { DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 256 } } } };
        XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
              const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    }
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++value);
    for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < value; i++)
        Sleep(10);
    ID3D12Resource_Map(buf, 0, NULL, (void **)&out);
    view_memory(&views, &retired, &textures, &bytes);
    CHECK(again && again != made[5] && out && !memcmp(out, linear, sizeof(linear)),
          "view textures: an evicted one asked for again is made anew and filled from title memory (%p, was %p)",
          (void *)again, (void *)made[5]);
    if (out)
        ID3D12Resource_Unmap(buf, 0, NULL);

    /* A T# the title wrote itself with texture 6's address: made anew too, and its view then finds that one. */
    {
        UINT64 at = (UINT64)(ULONG_PTR)(memory + (6 << 16));
        UINT32 t[2] = { (UINT32)(at >> 8), (UINT32)(at >> 40) & 0xff };
        revived = texture_of(t);
        view.address = at;
        XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 3 * inc);
        CHECK(revived && revived != made[6] && slot_resource(start + 3 * inc) == revived,
              "view textures: a T# of an evicted one's address makes it anew (%p, was %p)", (void *)revived,
              (void *)made[6]);
    }

    /* Over a budget: the ones no slot holds go, even lately asked for; those in slots stay. */
    view.address = (UINT64)(ULONG_PTR)(memory + ((SIZE_T)(COUNT + 1) << 16));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + inc);
    view.address = (UINT64)(ULONG_PTR)(memory + ((SIZE_T)(COUNT + 2) << 16));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + inc);
    Sleep(1200);
    gone = evict(0, 1);
    CHECK(gone >= 1 && slot_resource(start) == made[0] && slot_resource(start + inc)
          && slot_resource(start + 3 * inc) == revived,
          "view textures: over the budget the ones no slot holds are evicted, those in slots stay (%u went)", gone);
done:
    CHECK(SUCCEEDED(hr), "view texture eviction test set up (%#lx)", hr);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
    /* The memory stays: the views kept still name it. */
}

/* View buffers with memory of their own (a buffer view of title memory that isn't GPU memory):
 * one no slot holds and not asked for a while is retired (proxy.c, view_buffer_retire_enabled),
 * one a slot holds stays, and one asked for again is made anew. A change in GPU memory elsewhere
 * leaves them be: only one whose own memory became GPU memory is made again there. */
static void test_view_buffer_retire(void *dev, void *queue)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    void (WINAPI *view_buffers)(UINT64 *, UINT64 *, UINT64 *, UINT64 *) = (void *)GetProcAddress(m, "ReboxViewBuffers");
    UINT (WINAPI *evict)(UINT, UINT64) = (void *)GetProcAddress(m, "ReboxViewEvict");
    ID3D12Resource *(WINAPI *slot_resource)(SIZE_T) = (void *)GetProcAddress(m, "ReboxShadowSlotResource");
    HRESULT (WINAPI *gpu_alloc)(SIZE_T, SIZE_T, void **, UINT64 *) =
            (void *)GetProcAddress(m, "ReboxGpuAlloc");
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    struct { D3D12_SHADER_RESOURCE_VIEW_DESC v; UINT64 address; BYTE pad[8]; } view = { { DXGI_FORMAT_R32_UINT,
            D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING } };
    D3D12_SHADER_RESOURCE_VIEW_DESC null = { DXGI_FORMAT_R32_UINT, D3D12_SRV_DIMENSION_BUFFER,
                                             D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
    /* With unordered access: the GPU may write them, so each has memory of its own. */
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 4096, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
    BYTE console_desc[0x40] = { 0 };
    UINT64 alive0, bytes0, evicted0, superseded0, alive, bytes, evicted, superseded;
    ID3D12Resource *a, *b, *a2, *b2;
    void *heap = NULL, *gpu_cpu = NULL;
    UINT64 gpu_va;
    BYTE *memory;
    SIZE_T start;
    UINT inc;

    if (!view_buffers || !evict || !slot_resource
            || !(memory = VirtualAlloc(NULL, 4 << 16, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)))
        return;
    if (FAILED(XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap)))
        return;
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    start = XSLOT(heap, 9, SIZE_T)(heap);
    view.v.Buffer.NumElements = null.Buffer.NumElements = 1024;
    memcpy(console_desc, &bd, sizeof(bd));
    view_buffers(&alive0, &bytes0, &evicted0, &superseded0);
    /* A in slot 0, B in slot 1, which is then emptied. */
    view.address = (UINT64)(ULONG_PTR)memory;
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start);
    a = slot_resource(start);
    view.address = (UINT64)(ULONG_PTR)(memory + (2 << 16));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + inc);
    b = slot_resource(start + inc);
    view_buffers(&alive, &bytes, &evicted, &superseded);
    if (alive - alive0 != 2)
    {
        /* This memory became GPU memory in place (the GPU VA identity layer is there): no own buffers. */
        printf("info view buffers: plain memory is GPU memory here (%llu own made), retiring not tested\n",
               (unsigned long long)(alive - alive0));
        XSLOT(heap, 2, ULONG)(heap);
        return;
    }
    CHECK(a && b && a != b && bytes - bytes0 >= 8192, "view buffers: two with memory of their own (%llu bytes)",
          (unsigned long long)(bytes - bytes0));
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, NULL, &null, start + inc);
    /* GPU memory made elsewhere: neither is made again. */
    if (gpu_alloc)
        gpu_alloc(32 << 20, 1 << 16, &gpu_cpu, &gpu_va);
    view.address = (UINT64)(ULONG_PTR)memory;
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 2 * inc);
    CHECK(slot_resource(start + 2 * inc) == a, "view buffers: one is found again after GPU memory is made elsewhere");
    XSLOT(dev, 18, void, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, NULL, &null,
            start + 2 * inc);
    evict(10000, 0);
    view_buffers(&alive, &bytes, &evicted, &superseded);
    CHECK(alive - alive0 == 2 && evicted == evicted0, "view buffers: none is retired while lately asked for");
    Sleep(1200);
    evict(1000, 0);
    view_buffers(&alive, &bytes, &evicted, &superseded);
    CHECK(alive - alive0 == 1 && evicted - evicted0 == 1 && slot_resource(start) == a && superseded == superseded0,
          "view buffers: the one no slot holds is retired once not asked for, the one in a slot stays (%llu alive, "
          "%llu retired)", (unsigned long long)(alive - alive0), (unsigned long long)(evicted - evicted0));
    /* Both asked for again: A is the same, B made anew. */
    view.address = (UINT64)(ULONG_PTR)memory;
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 2 * inc);
    a2 = slot_resource(start + 2 * inc);
    view.address = (UINT64)(ULONG_PTR)(memory + (2 << 16));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 3 * inc);
    b2 = slot_resource(start + 3 * inc);
    view_buffers(&alive, &bytes, &evicted, &superseded);
    CHECK(a2 == a && b2 && alive - alive0 == 2, "view buffers: one retired is made anew when asked for again (%llu alive)",
          (unsigned long long)(alive - alive0));
    /* Without unordered access: nothing writes them, so they share one zeroed heap, no memory of their own,
     * and read zeros. */
    {
        D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
        D3D12_RESOURCE_DESC rd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 4096, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
        ID3D12Resource *z1, *z2, *rb = NULL;
        bd.Flags = D3D12_RESOURCE_FLAG_NONE;
        memcpy(console_desc, &bd, sizeof(bd));
        view_buffers(&alive0, &bytes0, &evicted0, &superseded0);
        view.address = (UINT64)(ULONG_PTR)(memory + (1 << 16));
        XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 2 * inc);
        z1 = slot_resource(start + 2 * inc);
        view.address = (UINT64)(ULONG_PTR)(memory + (3 << 16));
        XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 3 * inc);
        z2 = slot_resource(start + 3 * inc);
        view_buffers(&alive, &bytes, &evicted, &superseded);
        CHECK(z1 && z2 && z1 != z2 && alive - alive0 == 2 && bytes == bytes0,
              "view buffers without unordered access share the zeroed heap (%llu more, %llu bytes more)",
              (unsigned long long)(alive - alive0), (unsigned long long)(bytes - bytes0));
        if (z1 && SUCCEEDED(XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
                const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev,
                &readback, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource,
                (void **)&rb)))
        {
            ID3D12CommandAllocator *alloc = NULL;
            ID3D12Fence *fence = NULL;
            void *list = NULL;
            UINT32 *out = NULL;
            BOOL zero = FALSE;
            if (SUCCEEDED(XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &IID_ID3D12CommandAllocator, (void **)&alloc))
                    && SUCCEEDED(XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0,
                    D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&fence))
                    && SUCCEEDED(XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *,
                    ID3D12PipelineState *, REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL,
                    &IID_ID3D12GraphicsCommandList, &list)))
            {
                XSLOT(list, 15, void, ID3D12Resource *, UINT64, ID3D12Resource *, UINT64, UINT64)(list, rb, 0, z1, 0, 4096);
                XSLOT(list, 9, HRESULT)(list);
                XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
                XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
                for (int k = 0; k < 500 && ID3D12Fence_GetCompletedValue(fence) < 1; k++)
                    Sleep(10);
                if (SUCCEEDED(ID3D12Resource_Map(rb, 0, NULL, (void **)&out)))
                {
                    zero = TRUE;
                    for (int i = 0; i < 1024; i++)
                        zero &= !out[i];
                    ID3D12Resource_Unmap(rb, 0, NULL);
                }
            }
            CHECK(zero, "a view buffer in the zeroed heap reads zeros");
            {
                HMODULE a = GetModuleHandleA("rebox_anyps5.dll");
                void (*plan_stats)(UINT64 *, UINT64 *, UINT64 *, UINT64 *, UINT64 *) =
                        a ? (void *)GetProcAddress(a, "sa5_plan_stats") : NULL;
                UINT64 held = 0, plans = 0, ev = 0, re = 0, budget = 0;
                if (plan_stats)
                {
                    plan_stats(&held, &plans, &ev, &re, &budget);
                    CHECK(budget == 1024ull << 20, "AnyPS5's resource plans are bounded by default (%llu MiB, %llu held)",
                          (unsigned long long)budget >> 20, (unsigned long long)held);
                }
            }
            if (list) XSLOT(list, 2, ULONG)(list);
            if (fence) ID3D12Fence_Release(fence);
            if (alloc) ID3D12CommandAllocator_Release(alloc);
            ID3D12Resource_Release(rb);
        }
        /* An unordered access view over the first: made again with memory of its own (the GPU may
         * write it), and a shader resource view of it then finds that one. */
        {
            struct { D3D12_UNORDERED_ACCESS_VIEW_DESC v; UINT64 address, counter; } uav = { { DXGI_FORMAT_R32_UINT,
                    D3D12_UAV_DIMENSION_BUFFER } };
            ID3D12Resource *u, *again;
            uav.v.Buffer.NumElements = 1024;
            uav.address = (UINT64)(ULONG_PTR)(memory + (1 << 16));
            view_buffers(&alive0, &bytes0, &evicted0, &superseded0);
            XSLOT(dev, 71, void, const void *, const void *, SIZE_T)(dev, console_desc, &uav, start + 2 * inc);
            u = slot_resource(start + 2 * inc);
            view.address = uav.address;
            XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + 3 * inc);
            again = slot_resource(start + 3 * inc);
            view_buffers(&alive, &bytes, &evicted, &superseded);
            CHECK(u && u != z1 && again == u && bytes - bytes0 >= 4096 && superseded - superseded0 == 1,
                  "an unordered access view over a buffer in the zeroed heap makes it again with memory of its own "
                  "(%llu bytes more)", (unsigned long long)(bytes - bytes0));
        }
    }
    if (gpu_cpu)
    {
        void (WINAPI *gpu_free)(void *) = (void *)GetProcAddress(m, "ReboxGpuFree");
        if (gpu_free)
            gpu_free(gpu_cpu);
    }
    XSLOT(heap, 2, ULONG)(heap);
}

/* View textures given memory late (REBOX_VIEW_TEXTURE_DEFER, turned on here): one over memory not
 * committed yet is made without memory; once its memory is committed and written, its upload gives
 * it memory and fills it. Retired while a slot still holds it, it gives its memory back once the GPU
 * is past it, and the slot still names it. */
static void test_view_texture_defer(void *dev, void *queue)
{
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    void (WINAPI *defer)(BOOL) = (void *)GetProcAddress(m, "ReboxViewTextureDefer");
    int (WINAPI *memory_of)(ID3D12Resource *) = (void *)GetProcAddress(m, "ReboxViewTextureMemory");
    ID3D12Resource *(WINAPI *slot_resource)(SIZE_T) = (void *)GetProcAddress(m, "ReboxShadowSlotResource");
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    tile_fn tile = (tile_fn)(void *)GetProcAddress(m, "ReboxTile");
    struct { UINT mode; DXGI_FORMAT format; UINT width, height, layers, mips; BOOL volume, mip_major; } surface =
            { 1 /* SW_4KB_S */, DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, FALSE, FALSE };
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    struct { D3D12_SHADER_RESOURCE_VIEW_DESC v; UINT64 address; BYTE pad[8]; } view = { { DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING } };
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256 * 64, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    static UINT32 linear[64 * 64];
    BYTE console_desc[0x40] = { 0 };
    ID3D12Resource *res = NULL, *other = NULL, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *heap = NULL, *list = NULL;
    UINT32 *out = NULL;
    UINT64 value = 0;
    BYTE *memory;
    SIZE_T start;
    UINT inc;
    int state = -1;
    HRESULT hr;

    if (!defer || !memory_of || !slot_resource || !tile
            || !(memory = VirtualAlloc(NULL, 4 << 16, MEM_RESERVE, PAGE_READWRITE)))
        return;
    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (FAILED(hr))
        goto done;
    XSLOT(list, 9, HRESULT)(list);
    inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    start = XSLOT(heap, 9, SIZE_T)(heap);
    view.v.Texture2D.MipLevels = 1;
    view.address = (UINT64)(ULONG_PTR)memory;
    memcpy(console_desc, &td, sizeof(td));

    defer(TRUE);
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start);
    res = slot_resource(start);
    state = res ? memory_of(res) : -2;
    if (state == -1)
    {
        printf("info view textures given memory late: not made reserved here (no sparse images?), not tested\n");
        goto done;
    }
    CHECK(state == 0, "view textures: one over memory not committed yet is made without memory (%d)", state);
    /* Its memory committed and written: submissions find it readable, and its upload gives it memory. */
    for (int i = 0; i < 64 * 64; i++)
        linear[i] = i * 2654435761u + 7;
    VirtualAlloc(memory, 4 << 16, MEM_COMMIT, PAGE_READWRITE);
    tile(&surface, memory, 0, 0, linear, 256, 256 * 64);
    for (int i = 0; i < 12 && memory_of(res) == 0; i++)
    {
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
        XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++value);
        for (int k = 0; k < 500 && ID3D12Fence_GetCompletedValue(fence) < value; k++)
            Sleep(10);
    }
    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    {
        D3D12_TEXTURE_COPY_LOCATION src = { res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = 0 } };
        D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                            { .PlacedFootprint = { 0, { DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 256 } } } };
        XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
              const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    }
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++value);
    for (int k = 0; k < 500 && ID3D12Fence_GetCompletedValue(fence) < value; k++)
        Sleep(10);
    ID3D12Resource_Map(buf, 0, NULL, (void **)&out);
    CHECK(memory_of(res) == 1 && out && !memcmp(out, linear, sizeof(linear)),
          "view textures: one made without memory gets it at its first upload and is filled (%d)", memory_of(res));
    if (out)
        ID3D12Resource_Unmap(buf, 0, NULL);

    /* Another shape over its memory retires it; slot 0 still holds it. Past the GPU, its memory goes. */
    td.Width = 128;
    memcpy(console_desc, &td, sizeof(td));
    XSLOT(dev, 70, void, const void *, const void *, SIZE_T)(dev, console_desc, &view, start + inc);
    other = slot_resource(start + inc);
    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    for (int i = 0; i < 60 && memory_of(res) != 2; i++)
        Sleep(100);
    CHECK(other && other != res && memory_of(res) == 2 && slot_resource(start) == res,
          "view textures: a retired one a slot still holds gives its memory back past the GPU (%d)", memory_of(res));
done:
    defer(FALSE);
    CHECK(SUCCEEDED(hr), "view texture deferral test set up (%#lx)", hr);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (heap) XSLOT(heap, 2, ULONG)(heap);
}

static void test_placed_texture_late(void *dev, void *queue)
{
    struct { UINT mode; DXGI_FORMAT format; UINT width, height, layers, mips; BOOL volume, mip_major; } surface =
            { 1 /* SW_4KB_S */, DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, FALSE, FALSE };
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    tile_fn tile = (tile_fn)(void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxTile");
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256 * 64, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    static UINT32 linear[64 * 64];
    ID3D12Resource *tex = NULL, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *list = NULL, *reserved;
    UINT32 *out = NULL;
    UINT64 value = 0;
    HRESULT hr;

    if (!tile || !(reserved = VirtualAlloc(NULL, 65536, MEM_RESERVE, PAGE_NOACCESS)))
        return;
    hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *,
               REFIID, void **)(dev, (UINT64)(ULONG_PTR)reserved, &td, D3D12_RESOURCE_STATE_COPY_SOURCE, NULL,
               &IID_ID3D12Resource, (void **)&tex);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (FAILED(hr))
        goto done;
    /* Two submissions with the memory still reserved, then it is committed and written. */
    XSLOT(list, 9, HRESULT)(list);
    for (int i = 0; i < 2; i++)
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    VirtualAlloc(reserved, 65536, MEM_COMMIT, PAGE_READWRITE);
    for (int i = 0; i < 64 * 64; i++)
        linear[i] = i * 40503u + 17;
    tile(&surface, reserved, 0, 0, linear, 256, 256 * 64);
    for (int i = 0; i < 8; i++)
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    {
        D3D12_TEXTURE_COPY_LOCATION src = { tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = 0 } };
        D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                            { .PlacedFootprint = { 0, { DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 256 } } } };
        XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
              const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    }
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++value);
    for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < value; i++)
        Sleep(10);
    ID3D12Resource_Map(buf, 0, NULL, (void **)&out);
done:
    CHECK(out && !memcmp(out, linear, sizeof(linear)),
          "a texture placed on reserved memory is filled once the memory is committed and written (%#lx)", hr);
    if (out) ID3D12Resource_Unmap(buf, 0, NULL);
    /* The title takes the memory away again (a UE5 title remaps pages): submissions go on without
     * reading it. */
    if (out && list)
    {
        VirtualFree(reserved, 65536, MEM_DECOMMIT);
        for (int i = 0; i < 4; i++)
            XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
        XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++value);
        for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < value; i++)
            Sleep(10);
        CHECK(TRUE, "submissions go on after a placed texture's memory is decommitted");
    }
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (tex) ID3D12Resource_Release(tex);
    VirtualFree(reserved, 0, MEM_RELEASE);
}

/* A texture placed in XMemVirtualAlloc memory, which is write-watched: the CPU rewrites one texel
 * between submissions, away from the points a large texture's fingerprint reads (a video decoder
 * writing its planes), and the next submission uploads it. */
static void test_placed_texture_written(void *dev, void *queue)
{
    enum { W = 2048, H = 1024 };
    struct { UINT mode; DXGI_FORMAT format; UINT width, height, layers, mips; BOOL volume, mip_major; } surface =
            { 1 /* SW_4KB_S */, DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, 1, FALSE, FALSE };
    typedef void *(WINAPI *valloc_fn)(void *, SIZE_T, DWORD, ULONGLONG, DWORD);
    typedef UINT64 (WINAPI *size_fn)(const void *);
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    valloc_fn valloc = (valloc_fn)(void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemVirtualAlloc");
    size_fn surface_size = (size_fn)(void *)GetProcAddress(m, "ReboxDetileSize");
    tile_fn tile = (tile_fn)(void *)GetProcAddress(m, "ReboxTile");
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, W, H, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, W * 4 * H, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    ID3D12Resource *tex = NULL, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *list = NULL;
    UINT32 *linear = malloc(W * 4 * H), *out = NULL;
    BYTE *memory = NULL, *scratch = NULL;
    UINT64 size = 0, value = 0, step, at = 0;
    UINT changed = 0;
    HRESULT hr = E_FAIL;

    if (!valloc || !surface_size || !tile || !linear || !(size = surface_size(&surface)))
    {
        printf("skip write-watched placed texture: no addrlib\n");
        free(linear);
        return;
    }
    memory = valloc(NULL, (SIZE_T)size, MEM_RESERVE | MEM_COMMIT, 0, PAGE_READWRITE);
    scratch = malloc((size_t)size);
    if (memory && scratch)
    {
        for (int i = 0; i < W * H; i++)
            linear[i] = i * 2654435761u;
        tile(&surface, memory, 0, 0, linear, W * 4, (UINT64)W * 4 * H);
        hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
                   const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, (UINT64)(ULONG_PTR)memory, &td,
                   D3D12_RESOURCE_STATE_COPY_SOURCE, NULL, &IID_ID3D12Resource, (void **)&tex);
    }
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (FAILED(hr))
        goto done;
    /* Uploaded once, then sampled at each submission: a few submissions to settle. */
    XSLOT(list, 9, HRESULT)(list);
    for (int i = 0; i < 4; i++)
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    /* One texel changed, written to title memory as the tiling places it, at least 8 bytes from
     * every fingerprint point (32 spread over the texture, 8 bytes each). */
    step = (size - 8) / 31;
    for (int texel = W * H / 2 + W / 2 + 3; !at && texel < W * H; texel += 97)
    {
        UINT32 old = linear[texel];
        linear[texel] = 0xff0000ff;
        tile(&surface, scratch, 0, 0, linear, W * 4, (UINT64)W * 4 * H);
        for (UINT64 i = 0; i < size; i += 4)
            if (memcmp(scratch + i, memory + i, 4))
            {
                at = i % step >= 8 && i % step + 4 <= step ? i : 0;
                break;
            }
        if (!at)
            linear[texel] = old;
    }
    if (!at)
        goto done;
    memcpy(memory + at, scratch + at, 4);
    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    {
        D3D12_TEXTURE_COPY_LOCATION src = { tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = 0 } };
        D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                            { .PlacedFootprint = { 0, { DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, W * 4 } } } };
        XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
              const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    }
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++value);
    for (int i = 0; i < 1000 && ID3D12Fence_GetCompletedValue(fence) < value; i++)
        Sleep(10);
    if (SUCCEEDED(ID3D12Resource_Map(buf, 0, NULL, (void **)&out)))
        for (int i = 0; i < W * H; i++)
            changed += out[i] != linear[i];
done:
    CHECK(out && !changed, "a CPU write to a texture in write-watched memory, missed by its fingerprint, is "
          "uploaded at the next submission (offset %#llx, %u texels differ, %#lx)", (unsigned long long)at,
          out ? changed : W * H, hr);
    if (out) ID3D12Resource_Unmap(buf, 0, NULL);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (tex) ID3D12Resource_Release(tex);
    if (memory)
    {
        BOOL (WINAPI *vfree)(void *, SIZE_T, DWORD) = (void *)GetProcAddress(GetModuleHandleA("xmem.dll"), "XMemVirtualFree");
        vfree(memory, 0, MEM_RELEASE);
    }
    free(scratch);
    free(linear);
}

/* The texture's GPU copy read back after a submission: how many texels differ from expected (all
 * of them if it can't be read). */
static UINT tracked_readback(void *queue, void *list, ID3D12CommandAllocator *alloc, ID3D12Fence *fence,
                             UINT64 *value, ID3D12Resource *tex, ID3D12Resource *buf, const UINT32 *expected,
                             UINT w, UINT h)
{
    D3D12_TEXTURE_COPY_LOCATION src = { tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = 0 } };
    D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                        { .PlacedFootprint = { 0, { DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, w * 4 } } } };
    UINT32 *out = NULL;
    UINT changed = 0;

    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
          const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++*value);
    for (int i = 0; i < 1000 && ID3D12Fence_GetCompletedValue(fence) < *value; i++)
        Sleep(10);
    if (ID3D12Fence_GetCompletedValue(fence) < *value || FAILED(ID3D12Resource_Map(buf, 0, NULL, (void **)&out)))
        return w * h;
    for (UINT i = 0; i < w * h; i++)
        changed += out[i] != expected[i];
    ID3D12Resource_Unmap(buf, 0, NULL);
    return changed;
}

/* A texture placed in physical pages mapped into a range reserved for them (views of the pages'
 * sections, which write watching can't see): xmem write-tracks it. After its first upload its
 * pages are read-only; a CPU write into it, away from its fingerprint's points, faults, is let
 * through, and is uploaded at the next submission; so is a write through another mapping of the
 * same page, a ReadFile into it, a DirectStorage-style write into its pages, and a page mapped
 * anew under it (a remap). */
static void test_placed_texture_tracked(void *dev, void *queue)
{
    enum { W = 2048, H = 1024, UNIT = 64 << 10 };
    struct { UINT mode; DXGI_FORMAT format; UINT width, height, layers, mips; BOOL volume, mip_major; } surface =
            { 1 /* SW_4KB_S */, DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, 1, FALSE, FALSE };
    typedef void *(WINAPI *valloc_fn)(void *, SIZE_T, DWORD, ULONGLONG, DWORD);
    typedef BOOL (WINAPI *vfree_fn)(void *, SIZE_T, DWORD);
    typedef BOOL (WINAPI *palloc_fn)(ULONGLONG, ULONG_PTR *, ULONG_PTR *);
    typedef BOOL (WINAPI *pmap_fn)(void *, ULONG_PTR, ULONG_PTR *);
    typedef BOOL (WINAPI *pwrite_fn)(const ULONG_PTR *, ULONG_PTR, SIZE_T, const void *, SIZE_T);
    typedef UINT64 (WINAPI *size_fn)(const void *);
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    HMODULE m = GetModuleHandleA("d3d12_xs.dll"), x = GetModuleHandleA("xmem.dll");
    valloc_fn valloc = (valloc_fn)(void *)GetProcAddress(x, "XMemVirtualAlloc");
    vfree_fn vfree = (vfree_fn)(void *)GetProcAddress(x, "XMemVirtualFree");
    palloc_fn palloc = (palloc_fn)(void *)GetProcAddress(x, "XMemAllocatePhysicalPages");
    pmap_fn pmap = (pmap_fn)(void *)GetProcAddress(x, "XMemMapPhysicalPages");
    pwrite_fn pwrite = (pwrite_fn)(void *)GetProcAddress(x, "ReboxWritePhysicalPages");
    vquery_fn xquery = (vquery_fn)(void *)GetProcAddress(x, "XMemVirtualQuery");
    size_fn surface_size = (size_fn)(void *)GetProcAddress(m, "ReboxDetileSize");
    tile_fn tile = (tile_fn)(void *)GetProcAddress(m, "ReboxTile");
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, W, H, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, W * 4 * H, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    ID3D12Resource *tex = NULL, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *list = NULL;
    UINT32 *expected = malloc(W * 4 * H), *where = NULL;
    BYTE *memory = NULL, *alias = NULL, *spare = NULL;
    ULONG_PTR pages[160], count = 0, extra[1], one = 1;
    UINT64 size = 0, value = 0, step;
    UINT units = 0, changed;
    MEMORY_BASIC_INFORMATION mbi;
    HRESULT hr = E_FAIL;
    char path[MAX_PATH];
    HANDLE file;
    DWORD done;

    if (!valloc || !vfree || !palloc || !pmap || !pwrite || !xquery || !surface_size || !tile || !expected
            || !(size = surface_size(&surface)) || (units = (UINT)((size + UNIT - 1) / UNIT)) > ARRAYSIZE(pages))
    {
        printf("skip write-tracked placed texture: no addrlib or xmem\n");
        free(expected);
        return;
    }
    count = units;
    /* The texture's pages, one range reserved for physical pages (placeholders: views), a range
     * to map one of them at again, and a spare page for the remap. */
    if (!palloc(0, &count, pages) || count != units || !palloc(0, &one, extra)
            || !(memory = valloc(NULL, (SIZE_T)units * UNIT, MEM_RESERVE | 0x400000 /* MEM_PHYSICAL */, 0x3, PAGE_READWRITE))
            || !(alias = valloc(NULL, 2 * UNIT, MEM_RESERVE | 0x400000, 0x3, PAGE_READWRITE))
            || !pmap(memory, units, pages) || !(where = malloc((size_t)size)))
        goto done;
    spare = alias + UNIT;
    /* Where each texel lies in title memory: its index tiled. */
    for (int i = 0; i < W * H; i++)
        expected[i] = i;
    tile(&surface, where, 0, 0, expected, W * 4, (UINT64)W * 4 * H);
    for (int i = 0; i < W * H; i++)
        expected[i] = i * 2654435761u;
    tile(&surface, memory, 0, 0, expected, W * 4, (UINT64)W * 4 * H);
    hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
               const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, (UINT64)(ULONG_PTR)memory, &td,
               D3D12_RESOURCE_STATE_COPY_SOURCE, NULL, &IID_ID3D12Resource, (void **)&tex);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (FAILED(hr))
        goto done;
    XSLOT(list, 9, HRESULT)(list);
    changed = tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H);
    CHECK(!changed, "a texture in physical-page views is uploaded when placed (%u texels differ)", changed);
    {
        xmem_info info;
        CHECK(VirtualQuery(memory + 3 * UNIT, &mbi, sizeof(mbi)) && mbi.Protect == PAGE_READONLY
              && xquery(GetCurrentProcess(), memory + 3 * UNIT, &info) && info.mbi.Protect == PAGE_READWRITE
              && (info.gpu_access & 2),
              "its pages are armed for write tracking: read-only, though XMemVirtualQuery says writable (%#lx)",
              mbi.Protect);
    }
    /* Offsets in a unit at least 8 bytes from every fingerprint point (32 over the texture). */
    step = (size - 8) / 31;
#define AWAY(unit, from) ({ UINT64 o_ = (UINT64)(unit) * UNIT + (from); \
        while (o_ % step < 8 || o_ % step + 4 > step) o_ += 4; o_; })
    {
        /* A CPU write. */
        UINT64 at = AWAY(10, 0x1234);
        expected[where[at / 4]] = 0xff0000ff;
        *(volatile UINT32 *)(memory + at) = 0xff0000ff;
        changed = tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H);
        CHECK(!changed, "a CPU write into a write-tracked texture in physical-page views is uploaded at the next "
              "submission (%u texels differ)", changed);
    }
    if (pmap(alias, 1, &pages[20]))
    {
        /* A write through another mapping of one of its pages. */
        UINT64 at = AWAY(20, 0x2468);
        expected[where[at / 4]] = 0xff00ff00;
        *(volatile UINT32 *)(alias + at - 20 * UNIT) = 0xff00ff00;
        changed = tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H);
        CHECK(!changed && memory[at] == alias[at - 20 * UNIT],
              "a write through another mapping of a write-tracked page is uploaded (%u texels differ)", changed);
        pmap(alias, 1, NULL);
    }
    else
        CHECK(FALSE, "a page of the texture maps at a second address");
    if (pmap(spare, 1, extra))
    {
        /* A page mapped anew under the texture: what it holds is uploaded, and it is armed then. */
        UINT64 at = AWAY(30, 0x3000);
        memcpy(spare, memory + 30 * UNIT, UNIT);
        *(UINT32 *)(spare + at - 30 * UNIT) = 0xffff0000;
        expected[where[at / 4]] = 0xffff0000;
        CHECK(pmap(spare, 1, NULL) && pmap(memory + 30 * UNIT, 1, extra), "a spare page maps under the texture");
        changed = tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H);
        CHECK(!changed, "a page mapped anew under a write-tracked texture makes it upload (%u texels differ)", changed);
        CHECK(VirtualQuery(memory + 30 * UNIT, &mbi, sizeof(mbi)) && mbi.Protect == PAGE_READONLY,
              "the page mapped anew is armed after the upload (protect %#lx)", mbi.Protect);
    }
    else
        CHECK(FALSE, "a spare physical page maps");
    {
        /* DirectStorage into the pages (xmem writes through their home). */
        UINT64 at = AWAY(40, 0x800);
        UINT32 v = 0xff123456;
        expected[where[at / 4]] = v;
        changed = pwrite(&pages[40], 1, (SIZE_T)(at - 40 * UNIT), &v, 4)
                ? tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H) : W * H;
        CHECK(!changed, "a write into a write-tracked texture's physical pages (ReboxWritePhysicalPages) is uploaded "
              "(%u texels differ)", changed);
    }
    GetTempPathA(MAX_PATH, path);
    strcat(path, "rebox-tracked.bin");
    file = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_FLAG_DELETE_ON_CLOSE, NULL);
    if (file != INVALID_HANDLE_VALUE)
    {
        /* A file read into it: the system writes it, which doesn't fault; the read must not fail. */
        UINT64 at = AWAY(50, 0x400);
        UINT32 v = 0xff654321;
        BOOL ok = WriteFile(file, &v, 4, &done, NULL) && SetFilePointer(file, 0, NULL, FILE_BEGIN) == 0
                && ReadFile(file, memory + at, 4, &done, NULL) && done == 4;
        expected[where[at / 4]] = v;
        changed = ok ? tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H) : W * H;
        CHECK(ok && !changed, "a ReadFile into a write-tracked texture succeeds and is uploaded (error %lu, %u texels "
              "differ)", ok ? 0 : GetLastError(), changed);
        CloseHandle(file);
    }
    {
        /* A system call's write held over a submission (a DirectStorage request reading, then
         * decoding into its destination): the hold made the texture dirty, and its upload must not
         * arm the pages again before the write is done, or the request finds its destination
         * read-only and fails. What it wrote is uploaded once it's done. */
        void *(WINAPI *prepare)(ULONGLONG, ULONGLONG) = (void *)GetProcAddress(x, "ReboxPrepareWrite");
        void (WINAPI *finish)(void *) = (void *)GetProcAddress(x, "ReboxFinishWrite");
        UINT64 at = AWAY(60, 0x200);
        void *hold = prepare && finish ? prepare((ULONG_PTR)memory + 60 * UNIT, UNIT) : NULL;
        changed = tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H);
        CHECK(hold && VirtualQuery(memory + at, &mbi, sizeof(mbi)) && mbi.Protect == PAGE_READWRITE,
              "a held write's pages stay writable over a submission that uploads the texture (protect %#lx)",
              mbi.Protect);
        if (hold && mbi.Protect == PAGE_READWRITE)
            *(volatile UINT32 *)(memory + at) = 0xff0a0b0c;
        expected[where[at / 4]] = 0xff0a0b0c;
        if (finish)
            finish(hold);
        changed = tracked_readback(queue, list, alloc, fence, &value, tex, buf, expected, W, H);
        CHECK(!changed && VirtualQuery(memory + at, &mbi, sizeof(mbi)) && mbi.Protect == PAGE_READONLY,
              "what a held write wrote is uploaded once it's done, and the pages armed again (%u texels differ, "
              "protect %#lx)", changed, mbi.Protect);
    }
#undef AWAY
    {
        /* Armed pages made GPU memory in place (d3d12_xs adopts the run around a buffer, read-only
         * texture pages included): the driver's mapping replaces them, but Wine keeps their
         * read-only protection, and a copy is never disarmed: DirectStorage refused the destination
         * for good. Adoption makes them writable. Its own page and reservation, kept: an adopted
         * range stays one. */
        BOOL (WINAPI *track)(ULONGLONG, ULONGLONG) = (void *)GetProcAddress(x, "ReboxTrackPages");
        void (WINAPI *adopted)(ULONGLONG, ULONGLONG) = (void *)GetProcAddress(x, "ReboxPhysicalAdopted");
        BYTE *lone = valloc(NULL, UNIT, MEM_RESERVE | 0x400000, 0x3, PAGE_READWRITE);
        ULONG_PTR page[1];
        DWORD before = 0;

        one = 1;
        if (track && adopted && lone && palloc(0, &one, page) && pmap(lone, 1, page) && track((ULONG_PTR)lone, UNIT)
                && VirtualQuery(lone, &mbi, sizeof(mbi)))
        {
            before = mbi.Protect;
            adopted((ULONG_PTR)lone, UNIT);
        }
        CHECK(before == PAGE_READONLY && VirtualQuery(lone, &mbi, sizeof(mbi)) && mbi.Protect == PAGE_READWRITE,
              "armed physical pages made GPU memory in place are writable again (protect %#lx, then %#lx)", before,
              mbi.Protect);
    }
done:
    CHECK(SUCCEEDED(hr) && list, "a texture placed in physical-page views (%#lx)", hr);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (tex) ID3D12Resource_Release(tex);
    if (memory)
    {
        pmap(memory, units, NULL);
        vfree(memory, 0, MEM_RELEASE);
    }
    if (alias)
        vfree(alias, 0, MEM_RELEASE);
    free(where);
    free(expected);
}

/* An A8 glyph atlas placed early in memory nothing watches (a GPU memory chunk, or here the
 * title's own VirtualAlloc), uploaded once while empty, then many newer single-mip uncompressed
 * textures, all presumed hot, more than the hot budget: the atlas gains a letter, and the next
 * submission uploads it. The hot check went from the newest within its budget and never reached
 * the atlas (a credits sequence without its x-height letters). */
static void test_placed_atlas_behind_hot(void *dev, void *queue)
{
    enum { W = 1024, H = 1024, FILLERS = 48, FW = 512 };
    D3D12_RESOURCE_DESC ad = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, W, H, 1, 1, DXGI_FORMAT_A8_UNORM, { 1, 0 },
                               (D3D12_TEXTURE_LAYOUT)0x100 /* the console's linear */ };
    D3D12_RESOURCE_DESC fd = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, FW, FW, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 },
                               (D3D12_TEXTURE_LAYOUT)0x100 };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, W * H, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    ID3D12Resource *atlas = NULL, *fillers[FILLERS] = { 0 }, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *list = NULL;
    BYTE *memory = VirtualAlloc(NULL, W * H, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    BYTE *filler = VirtualAlloc(NULL, FW * FW * 4, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE), *out = NULL;
    UINT64 value = 0;
    UINT differ = W * H;
    HRESULT hr = memory && filler ? S_OK : E_OUTOFMEMORY;

    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
                   const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, (UINT64)(ULONG_PTR)memory, &ad,
                   D3D12_RESOURCE_STATE_COPY_SOURCE, NULL, &IID_ID3D12Resource, (void **)&atlas);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (FAILED(hr))
        goto done;
    /* The atlas uploads empty; then the newer textures, each uploaded once (64 a submission). */
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    for (int i = 0; i < FILLERS; i++)
    {
        memset(filler, i, FW * FW * 4);
        XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *,
              REFIID, void **)(dev, (UINT64)(ULONG_PTR)filler, &fd, D3D12_RESOURCE_STATE_COPY_SOURCE, NULL,
              &IID_ID3D12Resource, (void **)&fillers[i]);
    }
    for (int i = 0; i < 3; i++)
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    /* A letter. */
    for (int y = 500; y < 516; y++)
        memset(memory + y * W + 700, 0xff, 12);
    ID3D12CommandAllocator_Reset(alloc);
    XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
    {
        D3D12_TEXTURE_COPY_LOCATION src = { atlas, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = 0 } };
        D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                            { .PlacedFootprint = { 0, { DXGI_FORMAT_A8_UNORM, W, H, 1, W } } } };
        XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
              const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
    }
    XSLOT(list, 9, HRESULT)(list);
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, ++value);
    for (int i = 0; i < 1000 && ID3D12Fence_GetCompletedValue(fence) < value; i++)
        Sleep(10);
    if (SUCCEEDED(ID3D12Resource_Map(buf, 0, NULL, (void **)&out)))
    {
        differ = 0;
        for (int i = 0; i < W * H; i++)
            differ += out[i] != memory[i];
        ID3D12Resource_Unmap(buf, 0, NULL);
    }
done:
    CHECK(SUCCEEDED(hr) && !differ, "a glyph atlas placed before more presumed-hot textures than the hot budget holds "
          "gets its new letter at the next submission (%u texels differ, %#lx)", differ, hr);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    for (int i = 0; i < FILLERS; i++)
        if (fillers[i]) ID3D12Resource_Release(fillers[i]);
    if (atlas) ID3D12Resource_Release(atlas);
    if (memory) VirtualFree(memory, 0, MEM_RELEASE);
    if (filler) VirtualFree(filler, 0, MEM_RELEASE);
}

/* The older SDK's list has PC's query methods at 28-31 (XValkyrie; a UE5 title calls them): a
 * timestamp through EndQuery (29) and ResolveQueryData (30) reaches a readback buffer. */
static void test_list_queries(void *dev, void *queue)
{
    D3D12_QUERY_HEAP_DESC qd = { D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0 };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Resource *buf = NULL;
    ID3D12Fence *fence = NULL;
    void *qheap = NULL, *list = NULL;
    UINT64 *out = NULL;
    HRESULT hr;

    hr = XSLOT(dev, 39, HRESULT, const D3D12_QUERY_HEAP_DESC *, REFIID, void **)(dev, &qd, &IID_ID3D12QueryHeap, &qheap);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (SUCCEEDED(hr))
    {
        /* The console's list 5 IID (PC's first fields, Xbox ending; a UE5 title asks for it). */
        static const GUID list5 = { 0x55050859, 0x4024, 0x474c, { 0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74 } };
        void *l5 = NULL;
        HRESULT qhr = XSLOT(list, 0, HRESULT, REFIID, void **)(list, &list5, &l5);
        CHECK(SUCCEEDED(qhr) && l5 == list, "command list QueryInterface(console list 5) %#lx", qhr);
        if (l5)
            XSLOT(l5, 2, ULONG)(l5);
    }
    if (SUCCEEDED(hr))
    {
        XSLOT(list, 29, void, void *, D3D12_QUERY_TYPE, UINT)(list, qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        XSLOT(list, 30, void, void *, D3D12_QUERY_TYPE, UINT, UINT, ID3D12Resource *, UINT64)(list, qheap,
              D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, buf, 0);
        XSLOT(list, 9, HRESULT)(list);
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
        XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
        for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
            Sleep(10);
        ID3D12Resource_Map(buf, 0, NULL, (void **)&out);
    }
    CHECK(out && out[0], "a timestamp through list slots 29/30 (EndQuery, ResolveQueryData) -> %llu (%#lx)",
          out ? (unsigned long long)out[0] : 0ull, hr);
    if (SUCCEEDED(hr) && gpu_identity)
    {
        /* Console memory writes (list slots 69/70 bottom of pipe, 88 top, 71 a wait): to GPU memory
         * they run with the list; anywhere else they are made when recorded. */
        gpu_alloc_fn galloc = (gpu_alloc_fn)(void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxGpuAlloc");
        static UINT32 cpu_only;
        UINT32 *mem = NULL;
        UINT64 va = 0;
        if (galloc && SUCCEEDED(galloc(64 << 10, 64 << 10, (void **)&mem, &va)))
        {
            memset(mem, 0, 16);
            ID3D12CommandAllocator_Reset(alloc);
            XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
            XSLOT(list, 69, void, UINT64, UINT32)(list, va, 0x1234);
            XSLOT(list, 70, void, UINT64, UINT64)(list, va + 8, 0x1122334455667788ull);
            XSLOT(list, 88, void, UINT64, UINT32)(list, va + 4, 0xabcd);
            XSLOT(list, 71, void, UINT64, UINT32, UINT64, UINT64, UINT)(list, va, 3, 0x1234, ~0ull, 0);
            /* Wait64BitValueX (72) for a counter at 2 >= 1, as a retail title records it: a 32-bit 1
             * after the value with garbage above it, which isn't a mask (it held only at a 2 s
             * timeout when it was taken for one). */
            {
                static volatile UINT64 counter = 2;
                XSLOT(list, 72, void, UINT64, UINT32, UINT64, UINT64, UINT)(list, (UINT64)(ULONG_PTR)&counter,
                        D3D12_COMPARISON_FUNC_GREATER_EQUAL, 1, 0x23100000001ull, 0);
            }
            XSLOT(list, 69, void, UINT64, UINT32)(list, (UINT64)(ULONG_PTR)&cpu_only, 77);
            CHECK(cpu_only == 0, "Write32BitValueBottomOfPipeX outside GPU memory isn't made when recorded");
            XSLOT(list, 9, HRESULT)(list);
            /* Submitted through queue slot 35, the console's ExecuteCommandLists. */
            {
                ULONGLONG start = GetTickCount64();
                XSLOT(queue, 35, HRESULT, UINT, void *const *, UINT64, UINT)(queue, 1, &list, 1, 3);
                CHECK(GetTickCount64() - start < 1500, "Wait64BitValueX compares the whole value, not under the "
                      "garbage after it (submitted in %llu ms)", (unsigned long long)(GetTickCount64() - start));
            }
            XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 2);
            for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < 2; i++)
                Sleep(10);
            for (int i = 0; i < 200 && cpu_only != 77; i++)
                Sleep(10);
            CHECK(cpu_only == 77, "Write32BitValueBottomOfPipeX outside GPU memory is made by the CPU after the GPU's work");
            CHECK(mem[0] == 0x1234 && mem[1] == 0xabcd && *(UINT64 *)(mem + 2) == 0x1122334455667788ull,
                  "console memory writes (list slots 69/70/88, run by queue slot 35) -> %#x %#x %#llx", mem[0], mem[1],
                  (unsigned long long)*(UINT64 *)(mem + 2));
        }
    }
    if (out) ID3D12Resource_Unmap(buf, 0, NULL);
    if (fence)
    {
        /* SignalX (queue slot 23) is recorded for the hang report. */
        void (WINAPI *report)(void) = (void *)GetProcAddress(GetModuleHandleA("d3d12_xs.dll"), "ReboxHangReport");
        hr = XSLOT(queue, 23, HRESULT, ID3D12Fence *, UINT64, UINT)(queue, fence, 7, 0);
        for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < 7; i++)
            Sleep(10);
        CHECK(SUCCEEDED(hr) && ID3D12Fence_GetCompletedValue(fence) == 7 && report,
              "SignalX (queue slot 23) signals, and ReboxHangReport is exported %#lx", hr);
        if (report)
            report();
    }
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (qheap) XSLOT(qheap, 2, ULONG)(qheap);
}

/* d3d12_xs's struct detile_surface (shims/d3d12_xs/detile.h). */
struct test_surface
{
    UINT mode; /* 0 linear, 1 SW_4KB_S, 2 SW_64KB_D */
    DXGI_FORMAT format;
    UINT width, height, layers, mips;
    BOOL volume, mip_major;
};

/* Mip-major arrays (the default; the measured pool layout): each subresource is
 * the single-slice layout of that mip, at its offset times the slice count plus the slice times
 * the mip's size in one slice. */
static void test_mip_major_layout(void)
{
    typedef UINT64 (WINAPI *size_fn)(const void *);
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    size_fn surface_size = (size_fn)(void *)GetProcAddress(m, "ReboxDetileSize");
    tile_fn tile = (tile_fn)(void *)GetProcAddress(m, "ReboxTile");
    struct test_surface one = { 2, DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, 3, FALSE, FALSE };
    struct test_surface arr = { 2, DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 3, 3, FALSE, TRUE };
    static UINT32 linear[256 * 256];
    UINT64 slice, size;
    BYTE *single, *array;
    BOOL ok = TRUE;

    if (!surface_size || !tile || !(slice = surface_size(&one)))
        return;
    size = surface_size(&arr);
    single = calloc(1, slice);
    array = calloc(1, size);
    for (UINT layer = 0; layer < 3; layer++)
        for (UINT mip = 0; mip < 3; mip++)
        {
            UINT w = 256 >> mip;
            for (UINT i = 0; i < w * w; i++)
                linear[i] = i * 2654435761u + layer * 7919 + mip * 104729;
            tile(&arr, array, mip, layer, linear, w * 4, w * w * 4);
        }
    /* Mip 0 is 256 KiB, mip 1 64 KiB and mip 2 sits in the 64 KiB tail, smallest first: so in one
     * slice tail 0-64K, mip 1 64K-128K, mip 0 128K-384K; times 3 slices, mip-major. */
    for (UINT layer = 0; layer < 3; layer++)
        for (UINT mip = 0; mip < 3; mip++)
        {
            static const UINT64 start[3] = { 0x20000, 0x10000, 0 }, region[3] = { 0x40000, 0x10000, 0x10000 };
            UINT w = 256 >> mip;
            for (UINT i = 0; i < w * w; i++)
                linear[i] = i * 2654435761u + layer * 7919 + mip * 104729;
            memset(single, 0, slice);
            tile(&one, single, mip, 0, linear, w * 4, w * w * 4);
            ok &= slice == 0x60000 && size == 3 * slice
                  && !memcmp(single + start[mip], array + 3 * start[mip] + layer * region[mip], region[mip]);
        }
    CHECK(ok, "mip-major arrays: mip m of slice s at 3 * offset(m) + s * size(m) (slice %llu, array %llu bytes)",
          (unsigned long long)slice, (unsigned long long)size);
    free(single);
    free(array);
}

/* A small 2D array placed at a title address is read as SW_64KB_D, as every array in the measured
 * pools is, though its mip 0 is far below the 1 MiB the single-texture rule uses. */
static void test_placed_texture_array(void *dev, void *queue)
{
    struct test_surface surface =
            { 2 /* SW_64KB_D */, DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 2, 1, FALSE };
    typedef HRESULT (WINAPI *gpu_alloc_fn)(SIZE_T, SIZE_T, void **, UINT64 *);
    typedef UINT64 (WINAPI *size_fn)(const void *);
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    gpu_alloc_fn gpu_alloc = (gpu_alloc_fn)(void *)GetProcAddress(m, "ReboxGpuAlloc");
    size_fn surface_size = (size_fn)(void *)GetProcAddress(m, "ReboxDetileSize");
    tile_fn tile = (tile_fn)(void *)GetProcAddress(m, "ReboxTile");
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 2, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 2 * 256 * 64, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    static UINT32 linear[2][64 * 64];
    ID3D12Resource *tex = NULL, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *list = NULL, *cpu = NULL;
    UINT32 *out = NULL;
    UINT64 gpu = 0;
    HRESULT hr;

    if (!gpu_alloc || !surface_size || !tile || !surface_size(&surface) || FAILED(gpu_alloc(1 << 20, 65536, &cpu, &gpu)))
    {
        printf("skip placed texture array: no addrlib or GPU memory\n");
        return;
    }
    for (UINT l = 0; l < 2; l++)
    {
        for (int i = 0; i < 64 * 64; i++)
            linear[l][i] = (i + 1) * 2246822519u ^ (l + 1) * 0x9e3779b9u;
        tile(&surface, cpu, 0, l, linear[l], 256, 256 * 64);
    }
    hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *,
               REFIID, void **)(dev, gpu, &td, D3D12_RESOURCE_STATE_COPY_SOURCE, NULL, &IID_ID3D12Resource, (void **)&tex);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                   REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    if (SUCCEEDED(hr))
    {
        for (UINT l = 0; l < 2; l++)
        {
            D3D12_TEXTURE_COPY_LOCATION src = { tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = l } };
            D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    { .PlacedFootprint = { l * 256 * 64, { DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 256 } } } };
            XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
                  const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
        }
        XSLOT(list, 9, HRESULT)(list);
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
        XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
        for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
            Sleep(10);
        ID3D12Resource_Map(buf, 0, NULL, (void **)&out);
    }
    CHECK(out && !memcmp(out, linear, sizeof(linear)),
          "a small 2D array placed at a title address is read as SW_64KB_D, both slices (%#lx)", hr);
    if (out) ID3D12Resource_Unmap(buf, 0, NULL);
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (tex) ID3D12Resource_Release(tex);
}

/* flags: 0, or ALLOW_RENDER_TARGET: a small colour target the CPU fills (a title's glyph cache) is
 * kept in step with its memory too. */
static void test_placed_texture(void *dev, void *queue, D3D12_RESOURCE_FLAGS flags)
{
    struct test_surface surface =
            { 1 /* SW_4KB_S */, DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, FALSE };
    typedef HRESULT (WINAPI *gpu_alloc_fn)(SIZE_T, SIZE_T, void **, UINT64 *);
    typedef UINT64 (WINAPI *size_fn)(const void *);
    typedef BOOL (WINAPI *tile_fn)(const void *, void *, UINT, UINT, const void *, UINT64, UINT64);
    typedef void (WINAPI *written_fn)(UINT64, UINT64);
    HMODULE m = GetModuleHandleA("d3d12_xs.dll");
    gpu_alloc_fn gpu_alloc = (gpu_alloc_fn)(void *)GetProcAddress(m, "ReboxGpuAlloc");
    size_fn surface_size = (size_fn)(void *)GetProcAddress(m, "ReboxDetileSize");
    tile_fn tile = (tile_fn)(void *)GetProcAddress(m, "ReboxTile");
    written_fn written = (written_fn)(void *)GetProcAddress(m, "ReboxTitleMemoryWritten");
    D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                               { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN };
    D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256 * 64, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
    ID3D12Resource *tex = NULL, *buf = NULL;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    void *list = NULL, *cpu = NULL;
    UINT32 linear[64 * 64];
    UINT64 gpu = 0, size;
    const char *what = flags ? " (a 64x64 render target the CPU fills)" : "";
    HRESULT hr;

    td.Flags = flags;

    if (!gpu_alloc || !surface_size || !tile || !written)
    {
        CHECK(FALSE, "d3d12_xs exports the placed-texture functions");
        return;
    }
    size = surface_size(&surface);
    if (!size)
    {
        printf("skip placed textures: d3d12_xs built without addrlib\n");
        return;
    }
    CHECK(size == 16384, "a 64x64 RGBA8 texture in SW_4KB_S takes 16 KiB (%llu)", (unsigned long long)size);
    if (FAILED(gpu_alloc(1 << 20, 65536, &cpu, &gpu)))
    {
        printf("skip placed textures: no GPU memory\n");
        return;
    }
    for (int i = 0; i < 64 * 64; i++)
        linear[i] = i * 2654435761u;
    tile(&surface, cpu, 0, 0, linear, 256, 256 * 64);
    hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *,
               REFIID, void **)(dev, gpu, &td, D3D12_RESOURCE_STATE_COPY_SOURCE, NULL, &IID_ID3D12Resource,
               (void **)&tex);
    /* A T# the title writes itself holds the texture's address (base in 256-byte units): it finds
     * the texture placed there. */
    {
        void *(WINAPI *texture_of)(const UINT32 *) = (void *)GetProcAddress(m, "ReboxConsoleTextureOf");
        UINT32 t[2] = { (UINT32)(gpu >> 8), (UINT32)(gpu >> 40) & 0xff };
        CHECK(texture_of && tex && texture_of(t) == tex, "a T# with a placed texture's address finds it (%p, %p)",
              texture_of ? texture_of(t) : NULL, (void *)tex);
    }
    CHECK(tex && ID3D12Resource_GetGPUVirtualAddress(tex) == gpu,
          "a placed texture's GPU VA is the address the title placed it at (%#llx, %#llx)",
          tex ? (unsigned long long)ID3D12Resource_GetGPUVirtualAddress(tex) : 0, (unsigned long long)gpu);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &readback,
                   D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&buf);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   &IID_ID3D12CommandAllocator, (void **)&alloc);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                   &IID_ID3D12Fence, (void **)&fence);
    for (int pass = 0; pass < 3 && SUCCEEDED(hr); pass++)
    {
        D3D12_TEXTURE_COPY_LOCATION src = { tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { .SubresourceIndex = 0 } };
        D3D12_TEXTURE_COPY_LOCATION dst = { buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                            { .PlacedFootprint = { 0, { DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 256 } } } };
        UINT32 *out = NULL;
        BOOL same;

        if (pass == 1)
        {
            /* Rewrite the texture's memory and report it, as a DirectStorage request does. */
            for (int i = 0; i < 64 * 64; i++)
                linear[i] = ~linear[i] + i;
            tile(&surface, cpu, 0, 0, linear, 256, 256 * 64);
            written(gpu, size);
            ID3D12CommandAllocator_Reset(alloc);
        }
        if (pass == 2)
        {
            /* Rewrite it again without reporting it, as the CPU or a GPU copy would. */
            for (int i = 0; i < 64 * 64; i++)
                linear[i] = linear[i] * 3 + 7;
            tile(&surface, cpu, 0, 0, linear, 256, 256 * 64);
            ID3D12CommandAllocator_Reset(alloc);
        }
        if (!list)
            hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                       REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL,
                       &IID_ID3D12GraphicsCommandList, &list);
        else
            hr = XSLOT(list, 10, HRESULT, ID3D12CommandAllocator *, void *)(list, alloc, NULL);
        if (FAILED(hr))
            break;
        XSLOT(list, 16, void, const D3D12_TEXTURE_COPY_LOCATION *, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION *,
              const D3D12_BOX *)(list, &dst, 0, 0, 0, &src, NULL);
        XSLOT(list, 9, HRESULT)(list);
        XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
        XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, pass + 1);
        for (int i = 0; i < 500 && ID3D12Fence_GetCompletedValue(fence) < (UINT64)pass + 1; i++)
            Sleep(10);
        ID3D12Resource_Map(buf, 0, NULL, (void **)&out);
        same = out && !memcmp(out, linear, sizeof(linear));
        if (out) ID3D12Resource_Unmap(buf, 0, NULL);
        if (pass == 2)
            CHECK(same, "an unreported rewrite of the title memory is noticed and uploaded at the next submission%s",
                  what);
        else if (pass)
            CHECK(same, "a reported write to the title memory is uploaded at the next submission%s", what);
        else
            CHECK(same, "a texture placed at a title address is filled from its tiled memory (SW_4KB_S)%s", what);
    }
    if (list) XSLOT(list, 2, ULONG)(list);
    if (fence) ID3D12Fence_Release(fence);
    if (alloc) ID3D12CommandAllocator_Release(alloc);
    if (buf) ID3D12Resource_Release(buf);
    if (tex) ID3D12Resource_Release(tex);
}

static void test_xbox(void)
{
    HMODULE m = LoadLibraryA("d3d12_xs.dll");
    xbox_create_fn create = m ? (xbox_create_fn)(void *)GetProcAddress(m, "D3D12XboxCreateDevice") : NULL;
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize = m ? (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)
            GetProcAddress(m, "D3D12SerializeRootSignature") : NULL;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_ROOT_PARAMETER param = { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                                   .Constants = { 0, 0, 1 } };
    D3D12_ROOT_SIGNATURE_DESC rsd = { 1, &param };
    void *dev = NULL, *dev2 = NULL, *dev5 = NULL, *queue = NULL, *list = NULL, *heap = NULL, *rs = NULL, *same;
    ID3D12CommandAllocator *alloc = NULL;
    ID3D12Fence *fence = NULL;
    ID3D10Blob *blob = NULL;
    UINT64 *fields;
    SIZE_T cpu_start;
    UINT64 gpu_start;
    BYTE *embedded;
    HRESULT hr;
    int i;

    CHECK(create && serialize, "D3D12XboxCreateDevice present");
    if (!create || !serialize) return;
    test_gpu_memory(m);

    hr = create(NULL, NULL, &IID_ID3D12Device, &dev);
    if (FAILED(hr))
    {
        printf("skip D3D12XboxCreateDevice failed %#lx (no usable Vulkan device?)\n", hr);
        return;
    }
    hr = create(NULL, NULL, &IID_ID3D12Device2, &dev2);
    CHECK(SUCCEEDED(hr) && dev2 == dev, "D3D12XboxCreateDevice returns one device (%p, %p)", dev, dev2);
    /* Titles test this byte to decide whether to call CreateConstantBufferView and
     * CopyDescriptorsSimple or to write descriptors themselves (proxy.c). */
    CHECK(((const BYTE *)dev)[0x18] != 0, "device flags byte at +0x18 asks titles to call the descriptor methods");
    hr = XSLOT(dev, 0, HRESULT, REFIID, void **)(dev, &IID_ID3D12Device5, &dev5);
    CHECK(SUCCEEDED(hr) && dev5 == dev, "QueryInterface for another device version returns the proxy");

    /* Slot 44, CreatePlacedResourceX: a buffer at an address in GPU memory shares that memory; a
     * texture gets its own. */
    {
        typedef HRESULT (WINAPI *gpu_alloc_fn)(SIZE_T, SIZE_T, void **, UINT64 *);
        gpu_alloc_fn gpu_alloc = (gpu_alloc_fn)(void *)GetProcAddress(m, "ReboxGpuAlloc");
        D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
        D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   { 1, 0 } };
        ID3D12Resource *placed = NULL, *tex = NULL;
        void *cpu = NULL, *mapped = NULL;
        UINT64 gpu = 0;

        if (gpu_alloc && SUCCEEDED(gpu_alloc(1 << 20, 65536, &cpu, &gpu)))
        {
            UINT64 at = gpu + 65536;
            hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
                       const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, at, &bd, D3D12_RESOURCE_STATE_COMMON, NULL,
                       &IID_ID3D12Resource, (void **)&placed);
            if (SUCCEEDED(hr))
                ID3D12Resource_Map(placed, 0, NULL, &mapped);
            CHECK(SUCCEEDED(hr) && ID3D12Resource_GetGPUVirtualAddress(placed) == at
                      && mapped == (BYTE *)cpu + 65536,
                  "CreatePlacedResourceX (slot 44) places a buffer at its GPU address (%#lx)", hr);
            hr = XSLOT(dev, 44, HRESULT, UINT64, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
                       const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, at, &td, D3D12_RESOURCE_STATE_COMMON, NULL,
                       &IID_ID3D12Resource, (void **)&tex);
            CHECK(SUCCEEDED(hr) && tex, "CreatePlacedResourceX gives a texture its own memory (%#lx)", hr);
            if (placed) ID3D12Resource_Release(placed);
            if (tex) ID3D12Resource_Release(tex);
        }
        else
            printf("skip CreatePlacedResourceX: no GPU memory\n");
    }

    /* Two queue layouts share SDK 0x20004 (xcl): the older has the PC's GetTimestampFrequency and
     * GetClockCalibration at 16 and 17 (a GDK title), the newer Signal and Wait there; a fence at 16
     * or 17 tells them apart. */
    {
        UINT32 params[6] = { 0x20004 }, old[6] = { 0 };
        void *dev4 = NULL, *queue4 = NULL;
        ID3D12Fence *fence4 = NULL;

        hr = create(NULL, params, &IID_ID3D12Device, &dev4);
        CHECK(SUCCEEDED(hr) && dev4 == dev, "D3D12XboxCreateDevice with SDK version 0x20004 %#lx", hr);
        if (SUCCEEDED(hr))
            hr = XSLOT(dev, 8, HRESULT, const D3D12_COMMAND_QUEUE_DESC *, REFIID, void **)(dev, &qd,
                    &IID_ID3D12CommandQueue, &queue4);
        if (SUCCEEDED(hr))
            hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
                    &IID_ID3D12Fence, (void **)&fence4);
        if (SUCCEEDED(hr))
        {
            UINT64 frequency = 0, gpu = 0, cpu = 0;
            HRESULT f = XSLOT(queue4, 16, HRESULT, UINT64 *)(queue4, &frequency);
            HRESULT c = XSLOT(queue4, 17, HRESULT, UINT64 *, UINT64 *)(queue4, &gpu, &cpu);
            CHECK(SUCCEEDED(f) && frequency && SUCCEEDED(c),
                  "SDK 0x20004 queue, older layout: GetTimestampFrequency at 16 (%#lx, %llu Hz), GetClockCalibration "
                  "at 17 (%#lx)", f, (unsigned long long)frequency, c);
        }
        if (SUCCEEDED(hr))
            hr = XSLOT(queue4, 16, HRESULT, ID3D12Fence *, UINT64)(queue4, fence4, 7);
        for (i = 0; SUCCEEDED(hr) && i < 200 && ID3D12Fence_GetCompletedValue(fence4) != 7; i++)
            Sleep(10);
        CHECK(SUCCEEDED(hr) && fence4 && ID3D12Fence_GetCompletedValue(fence4) == 7,
              "SDK 0x20004 queue: Signal at slot 16 (%#lx)", hr);
        /* ... and the later command-list layout: markers at 57/58, PC 62 (OMSetDepthBounds)
         * at 109, then Close and execute on that queue. */
        {
            ID3D12CommandAllocator *alloc4 = NULL;
            void *list4 = NULL;
            hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &IID_ID3D12CommandAllocator, (void **)&alloc4);
            if (SUCCEEDED(hr))
                hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
                        REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc4, NULL, &IID_ID3D12GraphicsCommandList,
                        &list4);
            if (SUCCEEDED(hr))
            {
                XSLOT(list4, 57, void, UINT, const void *, UINT)(list4, 0, NULL, 0);
                XSLOT(list4, 109, void, FLOAT, FLOAT)(list4, 0.0f, 1.0f);
                XSLOT(list4, 58, void)(list4);
                /* A ray tracing pipeline from device slot 101 (CreateStateObject): a placeholder,
                 * set (list 114) and dispatched (115) without reaching vkd3d-proton. */
                {
                    D3D12_STATE_SUBOBJECT sub = { D3D12_STATE_SUBOBJECT_TYPE_EXISTING_COLLECTION, NULL };
                    D3D12_STATE_OBJECT_DESC sod = { D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, 1, &sub };
                    D3D12_DISPATCH_RAYS_DESC rays = { .Width = 1, .Height = 1, .Depth = 1 };
                    void *pipeline = NULL;
                    HRESULT so_hr = XSLOT(dev, 101, HRESULT, const D3D12_STATE_OBJECT_DESC *, REFIID, void **)(dev, &sod,
                            &IID_ID3D12StateObject, &pipeline);
                    CHECK(SUCCEEDED(so_hr) && pipeline, "CreateStateObject (device slot 101) gives a placeholder pipeline");
                    if (pipeline)
                    {
                        XSLOT(list4, 122, void, void *)(list4, pipeline);
                        XSLOT(list4, 123, void, const D3D12_DISPATCH_RAYS_DESC *)(list4, &rays);
                        /* A state object that isn't ours (Unreal 5 passed some): skipped too, not
                         * handed to vkd3d-proton, which would take it for one of its own. */
                        {
                            static void *const no_methods[16];
                            static struct { void *const *vtbl; BYTE junk[256]; } stranger = { no_methods };
                            XSLOT(list4, 122, void, void *)(list4, &stranger);
                            XSLOT(list4, 123, void, const D3D12_DISPATCH_RAYS_DESC *)(list4, &rays);
                            XSLOT(list4, 121, void, UINT64, UINT64, UINT)(list4, 0, 0, 4); /* copy, skipped */
                        }
                        /* ... and indirectly (list slot 61 with a rays signature, as Unreal 5 does). */
                        {
                            D3D12_INDIRECT_ARGUMENT_DESC ray_arg = { D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_RAYS };
                            D3D12_COMMAND_SIGNATURE_DESC ray_sig_desc = { sizeof(D3D12_DISPATCH_RAYS_DESC), 1, &ray_arg, 0 };
                            void *ray_sig = NULL;
                            ID3D12Resource *args = NULL;
                            D3D12_HEAP_PROPERTIES up = { D3D12_HEAP_TYPE_UPLOAD };
                            D3D12_RESOURCE_DESC ad = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                                                       { 1, 0 }, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
                            XSLOT(dev, 41, HRESULT, const D3D12_COMMAND_SIGNATURE_DESC *, void *, REFIID, void **)(dev,
                                    &ray_sig_desc, NULL, &IID_ID3D12CommandSignature, &ray_sig);
                            XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                                  D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &up,
                                  D3D12_HEAP_FLAG_NONE, &ad, D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource,
                                  (void **)&args);
                            if (ray_sig && args)
                            {
                                XSLOT(list4, 61, void, void *, UINT, void *, UINT64, void *, UINT64)(list4, ray_sig, 1, args,
                                        0, NULL, 0);
                                /* Unreal 5 calls it at the desktop's 59 as well. */
                                XSLOT(list4, 59, void, void *, UINT, void *, UINT64, void *, UINT64)(list4, ray_sig, 1, args,
                                        0, NULL, 0);
                            }
                            else
                                printf("skip indirect rays (signature %p, arguments %p)\n", ray_sig, (void *)args);
                            if (args) ID3D12Resource_Release(args);
                            if (ray_sig) IUnknown_Release((IUnknown *)ray_sig);
                        }
                        XSLOT(pipeline, 2, ULONG)(pipeline);
                    }
                }
                /* Barriers the desktop can't take (Unreal 5 passed a count with no array): left out. */
                {
                    D3D12_RESOURCE_BARRIER no_resource = { D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
                    no_resource.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                    XSLOT(list4, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list4, 1, NULL);
                    XSLOT(list4, 26, void, UINT, const D3D12_RESOURCE_BARRIER *)(list4, 1, &no_resource);
                }
                hr = XSLOT(list4, 9, HRESULT)(list4);
            }
            if (SUCCEEDED(hr))
            {
                XSLOT(queue4, 10, void, UINT, void *const *)(queue4, 1, &list4);
                hr = XSLOT(queue4, 16, HRESULT, ID3D12Fence *, UINT64)(queue4, fence4, 8);
            }
            for (i = 0; SUCCEEDED(hr) && i < 200 && ID3D12Fence_GetCompletedValue(fence4) != 8; i++)
                Sleep(10);
            CHECK(SUCCEEDED(hr) && ID3D12Fence_GetCompletedValue(fence4) == 8,
                  "SDK 0x20004 command list: markers at 57/58, PC 62 at 109, a placeholder's and a stranger state object's rays, barriers with nothing to "
                  "transition, executed (%#lx)", hr);
            if (SUCCEEDED(hr))
            {
                /* Slot 139: Reset with a third argument, on a closed list and on one still recording. */
                HRESULT closed, open;
                ID3D12CommandAllocator_Reset(alloc4);
                closed = XSLOT(list4, 139, HRESULT, void *, void *, UINT64)(list4, alloc4, NULL, 0);
                open = XSLOT(list4, 139, HRESULT, void *, void *, UINT64)(list4, alloc4, NULL, 0);
                CHECK(SUCCEEDED(closed) && SUCCEEDED(open) && SUCCEEDED(XSLOT(list4, 9, HRESULT)(list4)),
                      "SDK 0x20004 command list slot 139 resets it, closed (%#lx) or recording (%#lx)", closed, open);
            }
            if (list4) XSLOT(list4, 2, ULONG)(list4);
            if (alloc4) ID3D12CommandAllocator_Release(alloc4);
        }
        if (fence4) ID3D12Fence_Release(fence4);
        if (queue4) XSLOT(queue4, 2, ULONG)(queue4);
        if (dev4) XSLOT(dev4, 2, ULONG)(dev4);
        create(NULL, old, &IID_ID3D12Device, &dev4);
        XSLOT(dev4, 2, ULONG)(dev4);
    }

    /* Device slot 8 (PC 8), 9 (PC, forwarded unwrapped), 12, 14. */
    hr = XSLOT(dev, 8, HRESULT, const D3D12_COMMAND_QUEUE_DESC *, REFIID, void **)(dev, &qd,
            &IID_ID3D12CommandQueue, &queue);
    CHECK(SUCCEEDED(hr), "Xbox CreateCommandQueue (slot 8) %#lx", hr);
    if (queue)
    /* Fence slot 12: the address of the completed value, kept current for a queue's Signal and
     * a CPU Signal (fences made SHARED, as the target title does). */
    {
        void *f = NULL;
        volatile UINT64 *cell = NULL;
        HRESULT fhr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_SHARED,
                &IID_ID3D12Fence, &f);
        if (SUCCEEDED(fhr) && f)
        {
            int i;
            cell = XSLOT(f, 12, UINT64 *)(f);
            fhr = XSLOT(queue, 14, HRESULT, void *, UINT64)(queue, f, 5);
            for (i = 0; i < 500 && *cell != 5; i++)
                Sleep(10);
            CHECK(SUCCEEDED(fhr) && *cell == 5 && XSLOT(f, 12, UINT64 *)(f) == cell,
                  "fence slot 12: the completed value's address follows a queue Signal (%llu, %#lx)",
                  (unsigned long long)*cell, fhr);
            XSLOT(f, 10, HRESULT, UINT64)(f, 9);
            CHECK(*cell == 9, "fence slot 12: and a CPU Signal (%llu)", (unsigned long long)*cell);
            XSLOT(f, 2, ULONG)(f);
        }
        else
            CHECK(0, "a shared fence from device slot 36 (%#lx)", fhr);
    }
    hr = XSLOT(dev, 9, HRESULT, D3D12_COMMAND_LIST_TYPE, REFIID, void **)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT,
            &IID_ID3D12CommandAllocator, (void **)&alloc);
    CHECK(SUCCEEDED(hr), "CreateCommandAllocator through forward thunk (slot 9) %#lx", hr);
    hr = XSLOT(dev, 12, HRESULT, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator *, ID3D12PipelineState *,
            REFIID, void **)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, &IID_ID3D12GraphicsCommandList, &list);
    CHECK(SUCCEEDED(hr), "Xbox CreateCommandList (slot 12) %#lx", hr);
    hr = XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev, &hd,
            &IID_ID3D12DescriptorHeap, &heap);
    CHECK(SUCCEEDED(hr), "Xbox CreateDescriptorHeap (slot 14) %#lx", hr);
    CHECK(queue && alloc && list && heap, "objects created (queue %p, allocator %p, list %p, heap %p)",
          queue, (void *)alloc, list, heap);
    if (!queue || !alloc || !list || !heap) return;

    /* Inline fast paths read cursor +0x18 and limits +0x28/+0x30: all zero. */
    fields = list;
    CHECK(!fields[3] && !fields[5] && !fields[6], "command list cursor and limits are zero");

    hr = XSLOT(list, 7, HRESULT, REFIID, void **)(list, &IID_ID3D12Device, &same);
    CHECK(SUCCEEDED(hr) && same == dev, "command list GetDevice (slot 7) returns the device proxy");

    cpu_start = XSLOT(heap, 9, SIZE_T)(heap);
    gpu_start = XSLOT(heap, 10, UINT64)(heap);
    CHECK(cpu_start && gpu_start, "descriptor heap starts returned in RAX (slots 9, 10): %#llx %#llx",
          (unsigned long long)cpu_start, (unsigned long long)gpu_start);

    hr = serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL);
    if (SUCCEEDED(hr))
        hr = XSLOT(dev, 16, HRESULT, UINT, const void *, SIZE_T, REFIID, void **)(dev, 0,
                ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, &rs);
    CHECK(SUCCEEDED(hr), "root signature through the device proxy (slot 16) %#lx", hr);

    /* Record through Xbox slots: heaps (32), root signature (33), constant
     * (37), embedded data (148). */
    XSLOT(list, 32, void, UINT, void *const *)(list, 1, &heap);
    if (rs)
    {
        XSLOT(list, 33, void, void *)(list, rs);
        XSLOT(list, 37, void, UINT, UINT, UINT)(list, 0, 42, 0);
    }
    embedded = XSLOT(list, 148, void *, UINT64, UINT64)(list, 1000, 256);
    CHECK(embedded && !((ULONG_PTR)embedded & 255), "embedded allocation (slot 148) %p", embedded);
    if (embedded) memset(embedded, 0, 1000);
    hr = XSLOT(list, 9, HRESULT)(list);
    CHECK(SUCCEEDED(hr), "Close (slot 9) after Xbox-slot recording %#lx", hr);

    hr = XSLOT(dev, 36, HRESULT, UINT64, D3D12_FENCE_FLAGS, REFIID, void **)(dev, 0, D3D12_FENCE_FLAG_NONE,
            &IID_ID3D12Fence, (void **)&fence);
    CHECK(SUCCEEDED(hr), "CreateFence (slot 36) %#lx", hr);

    /* Unproxied children report the device proxy. (D3D12CreateDevice on the
     * same adapter returns the same vkd3d-proton device, so there is no
     * separate PC-path device to compare against.) */
    if (fence)
    {
        ID3D12Device *got = NULL;

        hr = ID3D12Fence_GetDevice(fence, &IID_ID3D12Device, (void **)&got);
        CHECK(SUCCEEDED(hr) && (void *)got == dev, "raw fence GetDevice returns the device proxy (%p)", (void *)got);
        if (got) ID3D12Device_Release(got);
    }
    XSLOT(queue, 10, void, UINT, void *const *)(queue, 1, &list);
    if (fence)
    {
        hr = XSLOT(queue, 14, HRESULT, ID3D12Fence *, UINT64)(queue, fence, 1);
        for (i = 0; i < 6000 && ID3D12Fence_GetCompletedValue(fence) < 1; i++)
            Sleep(10);
        CHECK(SUCCEEDED(hr) && ID3D12Fence_GetCompletedValue(fence) == 1,
              "ExecuteCommandLists (queue slot 10) unwraps the list and the GPU finishes it");
    }

    hr = XSLOT(dev, 100, HRESULT, UINT_PTR, UINT_PTR, UINT_PTR)(dev, 1, 2, 3);
    CHECK(hr == E_NOTIMPL, "unidentified device slot traps with E_NOTIMPL (%#lx)", hr);

    test_dxgi_device(dev);
    test_pacing(dev, queue);
    test_texture_va_upload(dev, queue, FALSE);
    test_texture_va_upload(dev, queue, TRUE);
    test_console_draw(dev, queue, FALSE, FALSE, FALSE);
    test_console_draw(dev, queue, TRUE, FALSE, FALSE);
    test_console_draw(dev, queue, FALSE, TRUE, FALSE);
    test_console_draw(dev, queue, FALSE, FALSE, 1);
    SetEnvironmentVariableA("REBOX_RECT_LISTS", "cpu");
    test_console_draw(dev, queue, FALSE, FALSE, 2);
    SetEnvironmentVariableA("REBOX_RECT_LISTS", NULL);
    test_xbox_root_flags(dev);
    test_console_command_signature(dev);
    test_state_object_placeholder(dev);
    test_stream_pso(dev);
    test_prebuild_info(dev);
    test_title_heap_properties(dev);
    test_depth_uav_texture(dev);
    test_console_mesh_stream();
    test_console_mesh_compile();
    test_guest_buffer_entry();
    test_real_heap_va(dev, queue);
    test_placed_texture(dev, queue, D3D12_RESOURCE_FLAG_NONE);
    test_placed_texture(dev, queue, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    test_placed_texture_array(dev, queue);
    test_placed_texture_late(dev, queue);
    test_view_texture_retire(dev, queue);
    test_view_texture_held(dev, queue);
    test_view_texture_evict(dev, queue);
    test_view_buffer_retire(dev, queue);
    test_view_texture_defer(dev, queue);
    test_placed_texture_written(dev, queue);
    test_placed_texture_tracked(dev, queue);
    test_placed_atlas_behind_hot(dev, queue);
    test_list_queries(dev, queue);
    test_mip_major_layout();
    test_console_shader(dev, queue);
    test_console_graphics(dev, queue);
    test_shadow_heap(dev);
    test_shadow_views(dev);
    test_console_bindings(dev);
    test_console_dispatch(dev, queue);
    test_present(dev, queue);
    test_ffx(dev, queue);
    test_present_format(dev, queue);

    /* Slot 60, graphics PSO with an extension array (TomBox): console VS/PS
     * bytes get fallbacks unless REBOX_SHADER_FALLBACK=0. */
    if (rs)
    {
        static const char vs_src[] = "REBOX-FAKE-CONSOLE-VS60", ps_bytes[] = "REBOX-FAKE-CONSOLE-PS60";
        char vs_bytes[sizeof(vs_src)];
        const char *fb = getenv("REBOX_SHADER_FALLBACK");
        BOOL strict = fb && *fb == '0';
        D3D12_GRAPHICS_PIPELINE_STATE_DESC gd = { 0 };
        BYTE ext[0x40] = { 0 };
        void *pso = NULL;

        memcpy(vs_bytes, vs_src, sizeof(vs_src));
        gd.pRootSignature = rs;
        gd.VS = (D3D12_SHADER_BYTECODE){ vs_bytes, sizeof(vs_bytes) };
        gd.PS = (D3D12_SHADER_BYTECODE){ ps_bytes, sizeof(ps_bytes) };
        gd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        gd.SampleMask = ~0u;
        gd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        gd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        gd.RasterizerState.DepthClipEnable = TRUE;
        gd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        gd.NumRenderTargets = 1;
        gd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        gd.SampleDesc.Count = 1;
        hr = XSLOT(dev, 60, HRESULT, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *, UINT, const void *, REFIID, void **)(
                dev, &gd, 1, ext, &IID_ID3D12PipelineState, &pso);
        CHECK(SUCCEEDED(hr) != strict, "graphics PSO through slot 60 with an extension %s (%#lx)",
              strict ? "rejected in strict mode" : "gets fallbacks", hr);

        /* Slot 55 derives variants from it (TomBox): 0x14C-byte patches, a tag and the state at
         * +4. The title's desc and shader bytes may be gone by then. */
        if (pso)
        {
            BYTE patches[6][0x14C] = { { 0 } };
            D3D12_RASTERIZER_DESC raster = gd.RasterizerState;
            D3D12_BLEND_DESC blend = gd.BlendState;
            D3D12_DEPTH_STENCIL_DESC depth = { 0 };
            void *derived = NULL, *bogus = NULL;
            UINT mask = 1;

            memset(vs_bytes, 0, sizeof(vs_bytes));
            raster.CullMode = D3D12_CULL_MODE_BACK;
            blend.RenderTarget[0].BlendEnable = TRUE;
            blend.RenderTarget[0].SrcBlend = blend.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
            blend.RenderTarget[0].DestBlend = blend.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ONE;
            blend.RenderTarget[0].BlendOp = blend.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
            depth.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            *(UINT *)patches[0] = 0, memcpy(patches[0] + 4, &depth, sizeof(depth));
            *(UINT *)patches[1] = 1, memcpy(patches[1] + 4, &raster, sizeof(raster));
            *(UINT *)patches[2] = 2, memcpy(patches[2] + 4, &blend, sizeof(blend));
            *(UINT *)patches[3] = 3, memcpy(patches[3] + 4, &mask, sizeof(mask));
            *(UINT *)patches[4] = 5;
            *(UINT *)patches[5] = 9;
            hr = XSLOT(dev, 55, HRESULT, void *, UINT, const void *, REFIID, void **)(
                    dev, pso, 6, patches, &IID_ID3D12PipelineState, &derived);
            CHECK(SUCCEEDED(hr) && derived && derived != pso,
                  "derived PSO (slot 55) with depth, rasterizer, blend, sample-mask, console and unknown patches (%#lx)", hr);
            hr = XSLOT(dev, 55, HRESULT, void *, UINT, const void *, REFIID, void **)(
                    dev, (void *)&gd, 1, patches, &IID_ID3D12PipelineState, &bogus);
            CHECK(FAILED(hr) && !bogus, "deriving from something that isn't a PSO fails (%#lx)", hr);
            if (derived) XSLOT(derived, 2, ULONG)(derived);
        }
        if (pso) XSLOT(pso, 2, ULONG)(pso);
    }

    /* A committed buffer smaller than a constant buffer view is rounded up to 256 bytes, so a
     * 256-byte CBV on it (as console titles make, per XValkyrie) stays inside. */
    {
        D3D12_HEAP_PROPERTIES up = { D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC small = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 80, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                      D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
        ID3D12Resource *buf = NULL;
        D3D12_RESOURCE_DESC got = { 0 };

        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &up, D3D12_HEAP_FLAG_NONE,
                   &small, D3D12_RESOURCE_STATE_GENERIC_READ, NULL, &IID_ID3D12Resource, (void **)&buf);
        if (buf)
            XSLOT(buf, 10, D3D12_RESOURCE_DESC *, D3D12_RESOURCE_DESC *)(buf, &got); /* GetDesc, MSVC hidden return */
        CHECK(SUCCEEDED(hr) && got.Width == 256, "an 80-byte committed buffer is made 256 bytes (%#lx, %llu)", hr,
              (unsigned long long)got.Width);
        if (buf)
        {
            D3D12_RESOURCE_DESC console_desc = { 0 };
            /* GetDesc in the console order: (result, resource). */
            ((D3D12_RESOURCE_DESC *(STDMETHODCALLTYPE *)(void *, void *))(*(void ***)buf)[10])(&console_desc, buf);
            CHECK(console_desc.Width == 256 && console_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER,
                  "a resource's GetDesc called in the console order, result first (%llu)",
                  (unsigned long long)console_desc.Width);
        }
        if (buf) ID3D12Resource_Release(buf);
    }

    /* Slot 76 as the FidelityFX upscaler calls it (Xbox2Windows): a 0x40-byte
     * desc with Xbox texture layout 0x101 and extension word 3. */
    {
        D3D12_HEAP_PROPERTIES heap_props = { D3D12_HEAP_TYPE_DEFAULT };
        D3D12_RESOURCE_DESC td = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 48, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   { 1, 0 }, (D3D12_TEXTURE_LAYOUT)0x101,
                                   D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };
        D3D12_RESOURCE_ALLOCATION_INFO info = { 0 };
        BYTE desc_x[0x40] = { 0 };
        ID3D12Resource *tex = NULL;

        memcpy(desc_x, &td, 0x34);
        *(UINT *)(desc_x + 0x34) = 3;
        hr = XSLOT(dev, 76, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const void *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &heap_props,
                D3D12_HEAP_FLAG_NONE, desc_x, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource,
                (void **)&tex);
        CHECK(SUCCEEDED(hr) && tex, "slot 76 texture with Xbox layout 0x101 (%#lx)", hr);
        if (tex) ID3D12Resource_Release(tex);
        XSLOT(dev, 25, D3D12_RESOURCE_ALLOCATION_INFO *, D3D12_RESOURCE_ALLOCATION_INFO *, UINT, UINT,
              const D3D12_RESOURCE_DESC *)(dev, &info, 0, 1, &td);
        CHECK(info.SizeInBytes >= 64 * 48 * 4, "GetResourceAllocationInfo (slot 25) with Xbox layout 0x101 (%llu)",
              (unsigned long long)info.SizeInBytes);
        /* Console titles pass the result first and the device second (a UE5 title). */
        {
            D3D12_RESOURCE_ALLOCATION_INFO console_info = { 0 }, *r;
            r = XSLOT(dev, 25, D3D12_RESOURCE_ALLOCATION_INFO *, void *, UINT, UINT,
                      const D3D12_RESOURCE_DESC *)(&console_info, dev, 0, 1, &td);
            CHECK(r == &console_info && console_info.SizeInBytes == info.SizeInBytes,
                  "GetResourceAllocationInfo (slot 25) called in the console order, result first (%llu)",
                  (unsigned long long)console_info.SizeInBytes);
        }
        /* A depth texture that is also a UAV, layout 0 with console flags (the target title). */
        {
            D3D12_RESOURCE_DESC dd = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 256, 144, 1, 1,
                                       DXGI_FORMAT_R32G8X24_TYPELESS, { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                       (D3D12_RESOURCE_FLAGS)0x80300006 };
            D3D12_RESOURCE_ALLOCATION_INFO depth_info = { 0 };
            XSLOT(dev, 25, D3D12_RESOURCE_ALLOCATION_INFO *, void *, UINT, UINT,
                  const D3D12_RESOURCE_DESC *)(&depth_info, dev, 0, 1, &dd);
            CHECK(depth_info.SizeInBytes >= 256 * 144 * 8 && depth_info.SizeInBytes != ~(UINT64)0,
                  "GetResourceAllocationInfo (slot 25) of a depth texture with unordered access (%#llx)",
                  (unsigned long long)depth_info.SizeInBytes);
            /* The console's own D16_UNORM_S8_UINT (118): a desktop depth format here. */
            dd.Format = (DXGI_FORMAT)118;
            dd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | (D3D12_RESOURCE_FLAGS)0x800000;
            depth_info.SizeInBytes = 0;
            XSLOT(dev, 25, D3D12_RESOURCE_ALLOCATION_INFO *, void *, UINT, UINT,
                  const D3D12_RESOURCE_DESC *)(&depth_info, dev, 0, 1, &dd);
            CHECK(depth_info.SizeInBytes && depth_info.SizeInBytes != ~(UINT64)0,
                  "GetResourceAllocationInfo of a D16_UNORM_S8_UINT (118) depth texture (%#llx)",
                  (unsigned long long)depth_info.SizeInBytes);
            {
                D3D12_HEAP_PROPERTIES dh = { D3D12_HEAP_TYPE_DEFAULT };
                D3D12_CLEAR_VALUE cv = { (DXGI_FORMAT)118, .DepthStencil = { 1.0f, 0 } };
                void *tex = NULL;
                HRESULT thr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
                        const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID,
                        void **)(dev, &dh, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv,
                        &IID_ID3D12Resource, &tex);
                CHECK(SUCCEEDED(thr) && tex, "a D16_UNORM_S8_UINT depth texture with its clear value is made (%#lx)", thr);
                if (tex) IUnknown_Release((IUnknown *)tex);
            }
            /* The same with unordered access, in the UAV state (the target title's shadow maps): a
             * depth target here, its plane views in the desktop's formats, its UAV a null view. */
            {
                D3D12_HEAP_PROPERTIES dh = { D3D12_HEAP_TYPE_DEFAULT };
                D3D12_DESCRIPTOR_HEAP_DESC vh = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
                D3D12_SHADER_RESOURCE_VIEW_DESC sv = { DXGI_FORMAT_R16_UNORM, D3D12_SRV_DIMENSION_TEXTURE2D,
                                                       D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
                D3D12_UNORDERED_ACCESS_VIEW_DESC uv = { DXGI_FORMAT_R16_UNORM, D3D12_UAV_DIMENSION_TEXTURE2D };
                void *tex = NULL, *views = NULL;
                HRESULT thr;
                SIZE_T at;
                dd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                thr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
                        const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID,
                        void **)(dev, &dh, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL,
                        &IID_ID3D12Resource, &tex);
                CHECK(SUCCEEDED(thr) && tex, "a D16S8 depth texture with unordered access in the UAV state is made (%#lx)",
                      thr);
                if (tex && SUCCEEDED(XSLOT(dev, 14, HRESULT, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **)(dev,
                        &vh, &IID_ID3D12DescriptorHeap, &views)))
                {
                    UINT inc = XSLOT(dev, 15, UINT, D3D12_DESCRIPTOR_HEAP_TYPE)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    sv.Texture2D.MipLevels = 1;
                    at = XSLOT(views, 9, SIZE_T)(views);
                    XSLOT(dev, 18, void, void *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, tex, &sv, at);
                    sv.Format = DXGI_FORMAT_R8_UINT;
                    XSLOT(dev, 18, void, void *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, SIZE_T)(dev, tex, &sv, at + inc);
                    XSLOT(dev, 19, void, void *, void *, const D3D12_UNORDERED_ACCESS_VIEW_DESC *, SIZE_T)(dev, tex, NULL,
                            &uv, at + 2 * inc);
                    XSLOT(views, 2, ULONG)(views);
                }
                if (tex) IUnknown_Release((IUnknown *)tex);
            }
        }
    }

    /* DEFAULT-heap buffers become CPU-visible, so a title can write through
     * their GPU VA as on the console. */
    {
        D3D12_HEAP_PROPERTIES heap_props = { D3D12_HEAP_TYPE_DEFAULT }, got;
        D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
        D3D12_HEAP_FLAGS flags;
        ID3D12Resource *buf = NULL;

        hr = XSLOT(dev, 27, HRESULT, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC *,
                D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **)(dev, &heap_props,
                D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&buf);
        if (SUCCEEDED(hr))
            hr = ID3D12Resource_GetHeapProperties(buf, &got, &flags);
        /* CPU-visible underneath (it maps, which a DEFAULT heap's can't), DEFAULT to the title. */
        {
            void *mapped = NULL;
            D3D12_RANGE none = { 0, 0 };
            HRESULT map_hr = buf ? ID3D12Resource_Map(buf, 0, &none, &mapped) : E_FAIL;
            CHECK(SUCCEEDED(hr) && got.Type == D3D12_HEAP_TYPE_DEFAULT && SUCCEEDED(map_hr) && mapped,
                  "DEFAULT-heap buffer is created CPU-visible (maps: %#lx), reported as DEFAULT (heap type %u)",
                  (unsigned long)map_hr, got.Type);
            if (SUCCEEDED(map_hr))
                ID3D12Resource_Unmap(buf, 0, &none);
        }
        if (buf)
        {
            UINT64 va = ID3D12Resource_GetGPUVirtualAddress(buf);
            D3D12_RANGE none = { 0, 0 };
            void *cpu = NULL;

            ID3D12Resource_Map(buf, 0, &none, &cpu);
            if (gpu_identity)
                CHECK((UINT64)(ULONG_PTR)cpu == va, "DEFAULT-heap buffer is identity-mapped too (%p, %#llx)", cpu,
                      (unsigned long long)va);
            else
                printf("info DEFAULT-heap buffer at CPU %p, GPU VA %#llx\n", cpu, (unsigned long long)va);
            ID3D12Resource_Release(buf);
        }
    }

    if (fence) ID3D12Fence_Release(fence);
    if (rs) XSLOT(rs, 2, ULONG)(rs);
    if (blob) ID3D10Blob_Release(blob);
    XSLOT(list, 2, ULONG)(list);
    XSLOT(heap, 2, ULONG)(heap);
    XSLOT(queue, 2, ULONG)(queue);
    ID3D12CommandAllocator_Release(alloc);
}

/* XAudio2 via the xaudio2_9.dll shim: XMA2 voices must be creatable (plain
 * Wine fails them) and play out as silence with callbacks on time. */
#pragma pack(push, 1)
typedef struct
{
    WAVEFORMATEX wfx;
    WORD NumStreams;
    DWORD ChannelMask, SamplesEncoded, BytesPerBlock, PlayBegin, PlayLength, LoopBegin, LoopLength;
    BYTE LoopCount, EncoderVersion;
    WORD BlockCount;
} xma2_format;
#pragma pack(pop)

static LONG xma_buffers_ended;

static void STDMETHODCALLTYPE cb_pass_start(IXAudio2VoiceCallback *cb, UINT32 bytes) {}
static void STDMETHODCALLTYPE cb_void(IXAudio2VoiceCallback *cb) {}
static void STDMETHODCALLTYPE cb_ctx(IXAudio2VoiceCallback *cb, void *ctx) {}
static void STDMETHODCALLTYPE cb_buffer_end(IXAudio2VoiceCallback *cb, void *ctx)
{
    if (ctx == (void *)0x1234)
        InterlockedIncrement(&xma_buffers_ended);
}
static void STDMETHODCALLTYPE cb_error(IXAudio2VoiceCallback *cb, void *ctx, HRESULT hr) {}

static IXAudio2VoiceCallbackVtbl cb_vtbl = { cb_pass_start, cb_void, cb_void, cb_ctx, cb_buffer_end, cb_ctx, cb_error };
static IXAudio2VoiceCallback voice_cb = { &cb_vtbl };

/* Wine's CreateMasteringVoice reports success even without an audio device,
 * so ask mmdevapi whether there is a render endpoint. */
static BOOL have_audio_device(void)
{
    HMODULE ole32 = LoadLibraryA("ole32.dll");
    HRESULT (WINAPI *co_init)(void *, DWORD) = (void *)GetProcAddress(ole32, "CoInitializeEx");
    HRESULT (WINAPI *co_create)(REFCLSID, IUnknown *, DWORD, REFIID, void **) =
            (void *)GetProcAddress(ole32, "CoCreateInstance");
    IMMDeviceEnumerator *e = NULL;
    IMMDevice *dev = NULL;
    BOOL ok = FALSE;

    co_init(NULL, COINIT_MULTITHREADED);
    if (SUCCEEDED(co_create(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_INPROC_SERVER, &IID_IMMDeviceEnumerator,
                            (void **)&e)))
    {
        ok = SUCCEEDED(IMMDeviceEnumerator_GetDefaultAudioEndpoint(e, eRender, eConsole, &dev));
        if (dev) IMMDevice_Release(dev);
        IMMDeviceEnumerator_Release(e);
    }
    return ok;
}

static void test_xaudio2(void)
{
    HMODULE m = LoadLibraryA("xaudio2_9.dll");
    HRESULT (WINAPI *create)(IXAudio2 **, UINT32, UINT32) = m ? (void *)GetProcAddress(m, MAKEINTRESOURCEA(1)) : NULL;
    xma2_format fmt = { { 0x166, 2, 48000, 48000 * 4, 2048, 16, sizeof(xma2_format) - sizeof(WAVEFORMATEX) },
                        1, 3, 48000, 65536, 0, 0, 0, 0, 0, 2, 1 };
    static BYTE xma_data[2048];
    XAUDIO2_BUFFER buf = { XAUDIO2_END_OF_STREAM, sizeof(xma_data), xma_data, 0, 4800, 0, 0, 0, (void *)0x1234 };
    IXAudio2MasteringVoice *master = NULL;
    IXAudio2SourceVoice *voice = NULL;
    IXAudio2 *xa = NULL;
    HRESULT hr;

    CHECK(create != NULL, "xaudio2_9 exports XAudio2Create at ordinal 1");
    if (!create) return;
    hr = create(&xa, 0, 0);
    CHECK(SUCCEEDED(hr) && xa, "XAudio2Create (%#lx)", hr);
    if (FAILED(hr)) return;

    if (!have_audio_device() || FAILED(hr = IXAudio2_CreateMasteringVoice(xa, &master, 0, 0, 0, NULL, NULL,
                                                                           AudioCategory_GameEffects)))
    {
        /* No audio device (sandbox): FAudio can't create any source voice
         * then (it reads the missing mastering voice), so the XMA2 path is
         * only checked on machines with audio (tests/manual-test.sh). */
        printf("skip XMA2 voice checks: no audio device (%#lx)\n", hr);
        IXAudio2_Release(xa);
        return;
    }
    hr = IXAudio2_CreateSourceVoice(xa, &voice, &fmt.wfx, 0, 2.0f, &voice_cb, NULL, NULL);
    CHECK(SUCCEEDED(hr) && voice, "XMA2 source voice (%#lx)", hr);
    if (voice)
    {
        hr = IXAudio2SourceVoice_SubmitSourceBuffer(voice, &buf, NULL);
        CHECK(SUCCEEDED(hr), "XMA2 buffer submitted (%#lx)", hr);
        IXAudio2SourceVoice_Start(voice, 0, 0);
        for (int i = 0; i < 300 && !xma_buffers_ended; i++)
            Sleep(10);
        CHECK(xma_buffers_ended == 1, "XMA2 buffer played out as silence: OnBufferEnd with its context");
        IXAudio2SourceVoice_DestroyVoice(voice);
    }
    IXAudio2MasteringVoice_DestroyVoice(master);
    IXAudio2_Release(xa);
}

/* REBOX_BINK_TRACE: a Bink 2 DLL loaded after xmem (build/Bink2Smoke.dll, a stand-in) is traced.
 * Its exports, looked up by name as a title would, are xmem's wrappers, which pass arguments and
 * results through. */
static void test_bink_trace(void)
{
    typedef UINT_PTR (*open_fn)(const char *, const void *, UINT32);
    typedef INT32 (*handle_fn)(UINT_PTR);
    typedef INT32 (*pause_fn)(UINT_PTR, INT32);
    typedef void (*goto_fn)(UINT_PTR, UINT32, INT32);
    HMODULE m = LoadLibraryA("Bink2Smoke.dll");
    MODULEINFO info = { 0 };
    open_fn open;
    handle_fn wait, doframe, next, close_movie;
    pause_fn pause;
    goto_fn seek;
    LONG (*calls)(void);
    UINT_PTR h;
    const UINT32 *head;

    CHECK(m != NULL, "load the stand-in Bink2Smoke.dll");
    if (!m) return;
    K32GetModuleInformation(GetCurrentProcess(), m, &info, sizeof(info));
    open = (open_fn)(void *)GetProcAddress(m, "BinkOpenWithOptions");
    wait = (handle_fn)(void *)GetProcAddress(m, "BinkWait");
    doframe = (handle_fn)(void *)GetProcAddress(m, "BinkDoFrame");
    next = (handle_fn)(void *)GetProcAddress(m, "BinkNextFrame");
    close_movie = (handle_fn)(void *)GetProcAddress(m, "BinkClose");
    pause = (pause_fn)(void *)GetProcAddress(m, "BinkPause");
    seek = (goto_fn)(void *)GetProcAddress(m, "BinkGoto");
    calls = (void *)GetProcAddress(m, "FakeBinkCalls");
    CHECK(open && wait && doframe && next && close_movie && pause && seek && calls, "the stand-in's exports resolve");
    if (!(open && wait && doframe && next && close_movie && pause && seek && calls)) return;
    CHECK((BYTE *)open < (BYTE *)info.lpBaseOfDll || (BYTE *)open >= (BYTE *)info.lpBaseOfDll + info.SizeOfImage,
          "BinkOpenWithOptions by GetProcAddress is xmem's wrapper (REBOX_BINK_TRACE=1)");
    CHECK(!open("missing.bk2", NULL, 0), "a failed open passes through as NULL");
    h = open("movie.bk2", NULL, 0x00100000);
    head = (const UINT32 *)h;
    CHECK(h && head[0] == 64 && head[1] == 32 && head[8] == 0x00100000,
          "an open passes its name and flags through and returns the movie's handle");
    if (!h) return;
    CHECK(wait(h) == 1 && doframe(h) == 0 && wait(h) == 0, "BinkWait and BinkDoFrame results pass through");
    next(h);
    CHECK(head[3] == 2, "BinkNextFrame reaches the DLL");
    CHECK(pause(h, 1) == 1 && head[0x11c / 4] == 1 && pause(h, 0) == 0 && !head[0x11c / 4],
          "BinkPause passes the handle and flag through");
    seek(h, 7, 0);
    CHECK(head[3] == 7, "BinkGoto passes its frame through");
    close_movie(h);
    CHECK(calls() == 10, "every wrapped call reached the stand-in once (%ld calls)", calls());
}

int main(int argc, char **argv)
{
    /* Before xmem loads: its import fixup reads it then. */
    SetEnvironmentVariableA("REBOX_BINK_TRACE", "1");
    test_xmem();
    test_bink_trace();
    /* REBOX_SKIP_ASSERTS: an int3 in the exe is stepped over (xmem crash.c). */
    if (getenv("REBOX_SKIP_ASSERTS"))
    {
        __debugbreak();
        CHECK(1, "an assertion (int3) in the exe was stepped over");
    }
    test_xcrdapi();
    test_xcrdapi_bcpack();
    test_gameinput();
    test_acphal();
    test_acphal_xma();
    test_xapu();
    test_chad();
    if (argc > 1)
        test_d3d12(argv[1], !(argc > 2 && !strcmp(argv[2], "nocache")));
    test_xbox();
    test_xaudio2();
    printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures != 0;
}
