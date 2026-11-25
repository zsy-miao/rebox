/* PresentX for the queue proxy.
 *
 * The console presents a "plane" (a render target) straight from the queue.
 * Here each queue that presents gets a window, on its own message thread,
 * and a flip-model DXGI swapchain (DXVK's dxgi.dll over vkd3d-proton). Each
 * PresentX copies plane 0 into the back buffer on the same queue, so it is
 * ordered after the title's rendering, and then presents.
 *
 * Argument layout (ScarlettX, traced from a retail title):
 * PresentX(plane_count, plane_params, present_params), where the qword at
 * plane_params+16 is the plane's ID3D12Resource, or a pointer to it. Both are
 * accepted. The first call is logged in full to settle which, and what the
 * rest holds.
 *
 * The copy needs the plane's current state. It comes from tracking: initial
 * states at resource creation and every barrier the title records (legacy
 * states and enhanced-barrier layouts), see proxy.c. Untracked resources are
 * assumed COMMON, the state a plane is normally presented in.
 *
 * Environment:
 *   REBOX_PRESENT=0        no window or swapchain (frame dumps still work)
 *   REBOX_VSYNC=N          Present sync interval, default 1
 *   REBOX_PRESENT_WINDOW=own  always present into a window of our own
 *
 * The swapchain goes on the title's own window when it has made one: a GDK
 * title (a UE5 title) creates a top-level window that on the console is the
 * screen and is never painted, so under Wine it stays white on top of a
 * window of ours. Without one, a window of our own is made.
 *
 * A present never blocks the title. vkd3d-proton waits for each present to
 * reach the screen without a timeout, so when the compositor stops taking
 * frames (a Wayland window that is hidden or on another workspace gets no
 * frame callbacks) Present would block the title's render thread for good:
 * A UE5 title stopped after its first frames that way, and Unreal's garbage
 * collector then gave up on textures waiting for the render thread. The
 * swapchain has a frame-latency waitable object instead; when no frame slot
 * frees within 100 ms the frame is dropped, and later ones are dropped
 * without waiting until a slot is free again.
 *   REBOX_DUMP_FRAMES=DIR  write presented frames to DIR\frame_<n>.bmp
 *                             (8-bit RGBA/BGRA planes) or .bin
 *   REBOX_DUMP_FRAMES_AT   which frames, comma-separated; default 1,60,600
 *   REBOX_DUMP_FRAMES_EVERY=N  also a frame every N seconds (the first presented frame, then
 *                             the first one N seconds after the last dump) */

#define COBJMACROS
/* Struct-returning methods (GetDesc) as hidden-pointer calls, matching
 * vkd3d-proton's MSVC ABI. */
#define WIDL_C_INLINE_WRAPPERS
#include <windows.h>
#include <d3d12.h>
/* DXGI IIDs are instantiated here; d3d12_xs.c does the D3D12 ones. */
#include <initguid.h>
#include <dxgi1_6.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "blit.h"
#include "debug_view.h"
#include "present.h"
#include "title_textures.h"
#include "proxy.h"

#define RING 3

struct present_queue
{
    ID3D12CommandQueue *queue;
    ID3D12Device *device;
    ID3D12Fence *fence;
    HANDLE event;
    UINT64 fence_value;
    struct
    {
        ID3D12CommandAllocator *alloc;
        ID3D12GraphicsCommandList *list;
        UINT64 done;
        ID3D12Resource *viewed; /* REBOX_DEBUG_VIEW's target, referenced until the list is done */
    } ring[RING];
    UINT64 frames;

    ID3D12Resource *convert; /* RGBA8 copy of planes a swapchain can't take */
    UINT convert_w, convert_h;
    ID3D12Resource *scaled;  /* RGBA8, the window's size, when the plane isn't */
    UINT scaled_w, scaled_h;

    HWND hwnd;
    BOOL own_window;  /* ours, not the title's: ours to close */
    IDXGISwapChain3 *swapchain;
    UINT width, height;
    DXGI_FORMAT format;
    BOOL swapchain_failed;
    HANDLE latency;   /* the swapchain's frame-latency waitable object */
    BOOL dropping;    /* the last present found no free frame slot */
    UINT64 dropped;
};

/* ------------------------------------------------------------- tracking */

enum { TRACK_LEGACY = 1, TRACK_LAYOUT = 2 };

#define TRACK_SIZE 65536
#define TRACK_PROBE 16

static struct
{
    void *resource;
    UINT kind, value;
} tracked[TRACK_SIZE];
static SRWLOCK track_lock = SRWLOCK_INIT;

static unsigned track_hash(const void *p)
{
    ULONG_PTR h = (ULONG_PTR)p >> 4;
    return (unsigned)((h ^ (h >> 15) ^ (h >> 31)) & (TRACK_SIZE - 1));
}

static void track(void *resource, UINT kind, UINT value)
{
    unsigned h = track_hash(resource), i, slot = h;

    if (!resource)
        return;
    AcquireSRWLockExclusive(&track_lock);
    for (i = 0; i < TRACK_PROBE; i++)
    {
        unsigned s = (h + i) & (TRACK_SIZE - 1);
        if (tracked[s].resource == resource || !tracked[s].resource)
        {
            slot = s;
            break;
        }
    }
    /* Full neighbourhood: overwrite the home slot. Entries for freed
     * resources are only ever stale, never wrong for live ones. */
    tracked[slot].resource = resource;
    tracked[slot].kind = kind;
    tracked[slot].value = value;
    ReleaseSRWLockExclusive(&track_lock);
}

static BOOL lookup(void *resource, UINT *kind, UINT *value)
{
    unsigned h = track_hash(resource);
    BOOL found = FALSE;

    AcquireSRWLockShared(&track_lock);
    for (unsigned i = 0; i < TRACK_PROBE; i++)
    {
        unsigned s = (h + i) & (TRACK_SIZE - 1);
        if (tracked[s].resource == resource)
        {
            *kind = tracked[s].kind;
            *value = tracked[s].value;
            found = TRUE;
            break;
        }
    }
    ReleaseSRWLockShared(&track_lock);
    return found;
}

void present_track_state(void *resource, D3D12_RESOURCE_STATES state)
{
    track(resource, TRACK_LEGACY, state);
}

/* The state the title's barriers last left a resource in (as recorded, not as executed). */
BOOL present_tracked_state(void *resource, D3D12_RESOURCE_STATES *state)
{
    UINT kind, value;
    if (!lookup(resource, &kind, &value) || kind != TRACK_LEGACY)
        return FALSE;
    *state = value;
    return TRUE;
}

BOOL present_tracked(void *resource, UINT *kind, UINT *value)
{
    return lookup(resource, kind, value);
}

void present_track_barriers(UINT count, const D3D12_RESOURCE_BARRIER *barriers)
{
    for (UINT i = 0; barriers && i < count; i++)
    {
        const D3D12_RESOURCE_BARRIER *b = &barriers[i];

        /* Planes are single-subresource; whole-resource or subresource 0
         * transitions are the ones that matter. */
        if (b->Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION
            && (b->Transition.Subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES || !b->Transition.Subresource)
            && !(b->Flags & D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY))
            track(b->Transition.pResource, TRACK_LEGACY, b->Transition.StateAfter);
    }
}

void present_track_enhanced(UINT32 count, const D3D12_BARRIER_GROUP *groups)
{
    for (UINT32 g = 0; groups && g < count; g++)
    {
        if (groups[g].Type != D3D12_BARRIER_TYPE_TEXTURE)
            continue;
        for (UINT32 i = 0; i < groups[g].NumBarriers; i++)
        {
            const D3D12_TEXTURE_BARRIER *t = &groups[g].pTextureBarriers[i];
            if (t->Subresources.IndexOrFirstMipLevel == 0xffffffff || !t->Subresources.IndexOrFirstMipLevel)
                track(t->pResource, TRACK_LAYOUT, t->LayoutAfter);
        }
    }
}

/* ------------------------------------------------------------------ setup */

static int env_int(const char *name, int fallback)
{
    const char *v = getenv(name);
    return v && *v ? atoi(v) : fallback;
}

static void *resource_vtbl(ID3D12Device *device)
{
    static void *vtbl;
    D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                 D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
    ID3D12Resource *probe;

    /* vkd3d-proton shares one vtable between all resources. */
    if (!vtbl && SUCCEEDED(ID3D12Device_CreateCommittedResource(device, &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&probe)))
    {
        vtbl = *(void **)probe;
        ID3D12Resource_Release(probe);
    }
    return vtbl;
}

static struct present_queue *present_create(ID3D12CommandQueue *queue)
{
    struct present_queue *s = calloc(1, sizeof(*s));
    D3D12_COMMAND_QUEUE_DESC qd;
    HRESULT hr;

    s->queue = queue;
    ID3D12CommandQueue_AddRef(queue);
    if (FAILED(hr = ID3D12CommandQueue_GetDevice(queue, &IID_ID3D12Device, (void **)&s->device)))
        goto fail;
    qd = ID3D12CommandQueue_GetDesc(queue);
    if (qd.Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
        rebox_log("d3d12_xs: PresentX on a non-direct queue (type %u)", qd.Type);
    if (FAILED(hr = ID3D12Device_CreateFence(s->device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&s->fence)))
        goto fail;
    s->event = CreateEventW(NULL, FALSE, FALSE, NULL);
    for (int i = 0; i < RING; i++)
    {
        if (FAILED(hr = ID3D12Device_CreateCommandAllocator(s->device, qd.Type, &IID_ID3D12CommandAllocator,
                                                            (void **)&s->ring[i].alloc)))
            goto fail;
        if (FAILED(hr = ID3D12Device_CreateCommandList(s->device, 0, qd.Type, s->ring[i].alloc, NULL,
                                                       &IID_ID3D12GraphicsCommandList, (void **)&s->ring[i].list)))
            goto fail;
        ID3D12GraphicsCommandList_Close(s->ring[i].list);
    }
    return s;

fail:
    rebox_log("d3d12_xs: PresentX setup failed: %#lx", hr);
    present_queue_destroy(s);
    return NULL;
}

/* ---------------------------------------------------------------- window */

static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    if (msg == WM_CLOSE)
    {
        /* As closing a game's window does. TerminateProcess, not
         * ExitProcess: DLL teardown with a live device can hang. */
        rebox_log("d3d12_xs: window closed, exiting");
        TerminateProcess(GetCurrentProcess(), 0);
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

struct window_args
{
    UINT width, height;
    HANDLE ready;
    HWND hwnd;
};

/* The window lives on its own thread so it stays responsive however the
 * title's threads behave. */
static DWORD WINAPI window_thread(void *param)
{
    struct window_args *args = param;
    WNDCLASSW wc = { .lpfnWndProc = window_proc, .hInstance = GetModuleHandleW(NULL),
                     .hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW), .lpszClassName = L"ReboxPresent" };
    UINT max_w = GetSystemMetrics(SM_CXSCREEN) * 9 / 10, max_h = GetSystemMetrics(SM_CYSCREEN) * 9 / 10;
    UINT w = args->width, h = args->height;
    RECT rect;
    MSG msg;

    /* Fit large planes on screen, keeping the aspect ratio; the swapchain
     * scales. */
    if (max_w && max_h && (w > max_w || h > max_h))
    {
        if ((UINT64)w * max_h > (UINT64)h * max_w)
            h = (UINT)((UINT64)h * max_w / w), w = max_w;
        else
            w = (UINT)((UINT64)w * max_h / h), h = max_h;
    }
    SetRect(&rect, 0, 0, w, h);
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    RegisterClassW(&wc);
    args->hwnd = CreateWindowExW(0, wc.lpszClassName, L"rebox", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                                 CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
                                 NULL, NULL, wc.hInstance, NULL);
    SetEvent(args->ready);
    if (!args->hwnd)
        return 1;
    while (GetMessageW(&msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return 0;
}

/* The title's largest visible top-level window, if it made one. */
static BOOL CALLBACK find_title_window(HWND hwnd, LPARAM param)
{
    HWND *best = (HWND *)param;
    WCHAR cls[64];
    DWORD pid = 0;
    RECT r, b;

    GetWindowThreadProcessId(hwnd, &pid);
    if (pid != GetCurrentProcessId() || !IsWindowVisible(hwnd) || GetParent(hwnd)
            || (GetClassNameW(hwnd, cls, ARRAYSIZE(cls)) && !wcscmp(cls, L"ReboxPresent"))
            || !GetClientRect(hwnd, &r) || r.right - r.left < 64 || r.bottom - r.top < 64)
        return TRUE;
    if (!*best || (GetClientRect(*best, &b)
                   && (r.right - r.left) * (r.bottom - r.top) > (b.right - b.left) * (b.bottom - b.top)))
        *best = hwnd;
    return TRUE;
}

static HWND title_window(void)
{
    const char *mode = getenv("REBOX_PRESENT_WINDOW");
    HWND best = NULL;
    WCHAR cls[64] = L"";
    RECT r = { 0 };

    if (mode && !strcmp(mode, "own"))
        return NULL;
    EnumWindows(find_title_window, (LPARAM)&best);
    if (best)
    {
        GetClassNameW(best, cls, ARRAYSIZE(cls));
        GetClientRect(best, &r);
        rebox_log("d3d12_xs: PresentX: presenting into the title's window %p (class %ls, %ldx%ld)", (void *)best,
                     cls, r.right - r.left, r.bottom - r.top);
    }
    return best;
}

/* Console planes are 4K and titles size their windows from that, which can be larger than the
 * screen. The window is left as the title made it (a tiling compositor sizes it anyway, and the
 * swapchain stretches the plane into whatever size it gets). REBOX_PRESENT_FIT=1 makes a window
 * larger than its monitor's work area borderless over the monitor, which Wine shows fullscreen;
 * REBOX_PRESENT_FIT=WxH gives it that size. Done on a thread of its own: a style change sends
 * messages to the window's thread, the title's, which can be waiting on the thread presenting
 * (a UE5 title deadlocked so with this on the present path). */
static DWORD WINAPI fit_thread(void *param)
{
    HWND hwnd = param;
    const char *fit = getenv("REBOX_PRESENT_FIT");
    MONITORINFO mi = { sizeof(mi) };
    HMONITOR monitor;
    LONG_PTR style;
    UINT w, h;
    RECT r;

    if (!(monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST))
            || !GetMonitorInfoW(monitor, &mi) || !GetWindowRect(hwnd, &r))
        return 0;
    rebox_log("d3d12_xs: PresentX: window %ldx%ld; monitor %ldx%ld (work area %ldx%ld), window DPI %u, system DPI %u",
                 r.right - r.left, r.bottom - r.top, mi.rcMonitor.right - mi.rcMonitor.left,
                 mi.rcMonitor.bottom - mi.rcMonitor.top, mi.rcWork.right - mi.rcWork.left,
                 mi.rcWork.bottom - mi.rcWork.top, GetDpiForWindow(hwnd), GetDpiForSystem());
    if (fit && sscanf(fit, "%ux%u", &w, &h) == 2 && w && h)
    {
        rebox_log("d3d12_xs: PresentX: sizing the window to %ux%u (REBOX_PRESENT_FIT)", w, h);
        SetWindowPos(hwnd, NULL, mi.rcWork.left, mi.rcWork.top, w, h,
                     SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS);
        return 0;
    }
    if (!fit || strcmp(fit, "1")
            || (r.right - r.left <= mi.rcWork.right - mi.rcWork.left && r.bottom - r.top <= mi.rcWork.bottom - mi.rcWork.top))
        return 0;
    rebox_log("d3d12_xs: PresentX: window %ldx%ld is larger than its monitor's work area; making it borderless "
                 "over the monitor, %ldx%ld", r.right - r.left, r.bottom - r.top,
                 mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top);
    style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    SetWindowLongPtrW(hwnd, GWL_STYLE, (style & ~(LONG_PTR)WS_OVERLAPPEDWINDOW) | WS_POPUP);
    /* Asynchronous: the window's thread may be the title's, busy in this present. */
    SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                 mi.rcMonitor.bottom - mi.rcMonitor.top,
                 SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS);
    return 0;
}

static void fit_to_monitor(HWND hwnd)
{
    HANDLE thread = CreateThread(NULL, 0, fit_thread, hwnd, 0, NULL);
    if (thread)
        CloseHandle(thread);
}

static HWND create_window(UINT width, UINT height)
{
    struct window_args *args = calloc(1, sizeof(*args)); /* the thread outlives this call */
    HANDLE thread;

    args->width = width;
    args->height = height;
    args->ready = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!(thread = CreateThread(NULL, 0, window_thread, args, 0, NULL)))
        return NULL;
    CloseHandle(thread);
    WaitForSingleObject(args->ready, INFINITE);
    CloseHandle(args->ready);
    return args->hwnd;
}

/* Flip-model swapchains take a few formats; pick the one in the plane's
 * copy-compatible family. */
static DXGI_FORMAT swapchain_format(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

static void wait_idle(struct present_queue *s)
{
    if (ID3D12Fence_GetCompletedValue(s->fence) < s->fence_value)
    {
        ID3D12Fence_SetEventOnCompletion(s->fence, s->fence_value, s->event);
        WaitForSingleObject(s->event, INFINITE); /* reading or reusing before the GPU is done corrupts */
    }
}

/* A frame slot to present into, or FALSE to drop this frame (see the top). */
static BOOL frame_slot_free(struct present_queue *s)
{
    if (!s->latency)
        return TRUE;
    if (WaitForSingleObject(s->latency, s->dropping ? 0 : 100) == WAIT_OBJECT_0)
    {
        if (s->dropping)
            rebox_log("d3d12_xs: PresentX: frames are shown again after %llu dropped",
                         (unsigned long long)s->dropped);
        s->dropping = FALSE;
        return TRUE;
    }
    if (!s->dropping)
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: PresentX: the display isn't taking frames (window hidden?); dropping them "
                         "instead of blocking the title");
        s->dropped = 0;
    }
    s->dropping = TRUE;
    s->dropped++;
    return FALSE;
}

static BOOL ensure_swapchain(struct present_queue *s, UINT width, UINT height, DXGI_FORMAT format)
{
    HRESULT (WINAPI *create_factory)(UINT, REFIID, void **);
    DXGI_SWAP_CHAIN_DESC1 desc = {
        .Width = width, .Height = height, .Format = format, .SampleDesc = { 1, 0 },
        .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT, .BufferCount = RING,
        .Scaling = DXGI_SCALING_STRETCH, .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
        .AlphaMode = DXGI_ALPHA_MODE_IGNORE, .Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT,
    };
    IDXGIFactory2 *factory = NULL;
    IDXGISwapChain1 *sc1 = NULL;
    HMODULE dxgi;
    HRESULT hr;

    if (s->swapchain && s->width == width && s->height == height && s->format == format)
        return TRUE;
    if (s->swapchain_failed || !env_int("REBOX_PRESENT", 1))
        return FALSE;
    if (s->swapchain)
    {
        /* The plane changed size or format: the old back buffers must be
         * idle before the swapchain goes. */
        wait_idle(s);
        IDXGISwapChain3_Release(s->swapchain);
        s->swapchain = NULL;
        if (s->latency)
            CloseHandle(s->latency);
        s->latency = NULL;
    }
    if (!s->hwnd)
        s->hwnd = title_window();
    if (!s->hwnd && (s->own_window = TRUE) && !(s->hwnd = create_window(width, height)))
    {
        rebox_log("d3d12_xs: PresentX: could not create a window (no display?); frames are not shown");
        s->swapchain_failed = TRUE;
        return FALSE;
    }
    fit_to_monitor(s->hwnd);
    if (!(dxgi = LoadLibraryA("dxgi.dll"))
        || !(create_factory = (void *)GetProcAddress(dxgi, "CreateDXGIFactory2"))
        || FAILED(hr = create_factory(0, &IID_IDXGIFactory2, (void **)&factory)))
    {
        rebox_log("d3d12_xs: PresentX: no DXGI factory; frames are not shown");
        s->swapchain_failed = TRUE;
        return FALSE;
    }
    hr = IDXGIFactory2_CreateSwapChainForHwnd(factory, (IUnknown *)s->queue, s->hwnd, &desc, NULL, NULL, &sc1);
    IDXGIFactory2_Release(factory);
    if (SUCCEEDED(hr))
    {
        hr = IDXGISwapChain1_QueryInterface(sc1, &IID_IDXGISwapChain3, (void **)&s->swapchain);
        IDXGISwapChain1_Release(sc1);
    }
    if (FAILED(hr))
    {
        rebox_log("d3d12_xs: PresentX: swapchain creation failed (%#lx); frames are not shown", hr);
        s->swapchain_failed = TRUE;
        return FALSE;
    }
    if (FAILED(IDXGISwapChain3_SetMaximumFrameLatency(s->swapchain, RING - 1))
            || !(s->latency = IDXGISwapChain3_GetFrameLatencyWaitableObject(s->swapchain)))
        rebox_log("d3d12_xs: PresentX: no frame-latency waitable object; a present may block the title");
    s->width = width;
    s->height = height;
    s->format = format;
    rebox_log("d3d12_xs: PresentX: %ux%u swapchain, format %u", width, height, format);
    return TRUE;
}

/* ---------------------------------------------------------------- dumps */

static BOOL dump_wanted(UINT64 frame)
{
    const char *at = getenv("REBOX_DUMP_FRAMES_AT"), *every = getenv("REBOX_DUMP_FRAMES_EVERY");
    static ULONGLONG last_dump;
    char *end;

    if (!getenv("REBOX_DUMP_FRAMES"))
        return FALSE;
    if (every && strtoul(every, NULL, 10))
    {
        ULONGLONG now = GetTickCount64();
        if (!last_dump || now - last_dump >= strtoul(every, NULL, 10) * 1000ull)
        {
            last_dump = now;
            return TRUE;
        }
    }
    if (!at || !*at)
        at = "1,60,600";
    while (*at)
    {
        unsigned long long n = strtoull(at, &end, 10);
        if (end == at)
            break;
        if (n == frame)
            return TRUE;
        at = *end ? end + 1 : end;
    }
    return FALSE;
}

static void write_dump_named(const char *name, UINT64 frame, const D3D12_RESOURCE_DESC *desc,
        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *fp, UINT rows, const BYTE *data);

static void write_dump(UINT64 frame, const D3D12_RESOURCE_DESC *desc, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *fp,
        const BYTE *data)
{
    write_dump_named("frame", frame, desc, fp, desc->Height, data);
}

/* rows is the footprint's row count, which is less than the height for block-compressed formats. */
static void write_dump_named(const char *name, UINT64 frame, const D3D12_RESOURCE_DESC *desc,
        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *fp, UINT rows, const BYTE *data)
{
    const char *dir = getenv("REBOX_DUMP_FRAMES");
    BOOL rgba = swapchain_format(desc->Format) == DXGI_FORMAT_R8G8B8A8_UNORM;
    BOOL bgra = swapchain_format(desc->Format) == DXGI_FORMAT_B8G8R8A8_UNORM;
    UINT w = (UINT)desc->Width, h = desc->Height;
    char path[MAX_PATH];
    FILE *f;

    snprintf(path, sizeof(path), "%s\\%s_%llu.%s", dir, name, (unsigned long long)frame, rgba || bgra ? "bmp" : "bin");
    if (!(f = fopen(path, "wb")))
    {
        rebox_log("d3d12_xs: cannot write %s", path);
        return;
    }
    if (rgba || bgra)
    {
        /* 32-bit top-down BMP; BMP pixels are BGRA. */
        BYTE header[54] = { 'B', 'M' };
        UINT32 v;
        INT32 neg_h = -(INT32)h;
        BYTE *row = malloc(w * 4);

        v = 54 + w * h * 4; memcpy(header + 2, &v, 4);
        v = 54; memcpy(header + 10, &v, 4);
        v = 40; memcpy(header + 14, &v, 4);
        memcpy(header + 18, &w, 4);
        memcpy(header + 22, &neg_h, 4);
        header[26] = 1;
        header[28] = 32;
        fwrite(header, 1, sizeof(header), f);
        for (UINT y = 0; y < h; y++)
        {
            const BYTE *src = data + fp->Offset + (SIZE_T)y * fp->Footprint.RowPitch;
            for (UINT x = 0; x < w; x++)
            {
                row[x * 4 + 0] = src[x * 4 + (rgba ? 2 : 0)];
                row[x * 4 + 1] = src[x * 4 + 1];
                row[x * 4 + 2] = src[x * 4 + (rgba ? 0 : 2)];
                row[x * 4 + 3] = src[x * 4 + 3];
            }
            fwrite(row, 1, w * 4, f);
        }
        free(row);
    }
    else
    {
        /* Raw rows as laid out by the GPU, behind a small header. */
        UINT32 hdr[4] = { w, h, desc->Format, fp->Footprint.RowPitch };
        fwrite(hdr, 1, sizeof(hdr), f);
        fwrite(data + fp->Offset, 1, (SIZE_T)fp->Footprint.RowPitch * rows, f);
    }
    fclose(f);
    rebox_log("d3d12_xs: dumped frame %llu to %s", (unsigned long long)frame, path);
}

/* --------------------------------------------------------------- present */

static ID3D12Resource *plane_resource(ID3D12Device *device, const void *plane_params)
{
    void *vtbl = resource_vtbl(device);
    void *cand;

    if (!plane_params || !vtbl)
        return NULL;
    cand = (void *)((const UINT64 *)plane_params)[2];
    if (!cand)
        return NULL;
    if (*(void **)cand == vtbl)
        return cand;
    cand = *(void **)cand;
    if (cand && *(void **)cand == vtbl)
        return cand;
    return NULL;
}

/* Moves the plane from its tracked state to the one the presenter reads it
 * in (copy source, or shader resource for the conversion blit), or back. */
static void transition(ID3D12GraphicsCommandList *list, ID3D12Resource *res, UINT kind, UINT from, BOOL shader_read,
        BOOL forward)
{
    if (kind == TRACK_LAYOUT)
    {
        UINT layout = shader_read ? D3D12_BARRIER_LAYOUT_SHADER_RESOURCE : D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        UINT sync = shader_read ? D3D12_BARRIER_SYNC_COMPUTE_SHADING : D3D12_BARRIER_SYNC_COPY;
        UINT access = shader_read ? D3D12_BARRIER_ACCESS_SHADER_RESOURCE : D3D12_BARRIER_ACCESS_COPY_SOURCE;
        D3D12_TEXTURE_BARRIER t = {
            .SyncBefore = forward ? D3D12_BARRIER_SYNC_ALL : sync,
            .SyncAfter = forward ? sync : D3D12_BARRIER_SYNC_ALL,
            .AccessBefore = forward ? D3D12_BARRIER_ACCESS_COMMON : access,
            .AccessAfter = forward ? access : D3D12_BARRIER_ACCESS_COMMON,
            .LayoutBefore = forward ? from : layout,
            .LayoutAfter = forward ? layout : from,
            .pResource = res,
            .Subresources = { 0xffffffff },
        };
        D3D12_BARRIER_GROUP g = { .Type = D3D12_BARRIER_TYPE_TEXTURE, .NumBarriers = 1, .pTextureBarriers = &t };

        if (from != layout)
            ID3D12GraphicsCommandList7_Barrier((ID3D12GraphicsCommandList7 *)list, 1, &g);
    }
    else
    {
        UINT state = shader_read ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_COPY_SOURCE;
        D3D12_RESOURCE_BARRIER b = { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };

        if (from == state)
            return;
        b.Transition.pResource = res;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = forward ? from : state;
        b.Transition.StateAfter = forward ? state : from;
        ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &b);
    }
}

/* The RGBA8 target planes in other formats are converted into. */
static BOOL ensure_texture(struct present_queue *s, ID3D12Resource **texture, UINT *tw, UINT *th, UINT width,
                           UINT height)
{
    D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
                                 DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };

    if (*texture && *tw == width && *th == height)
        return TRUE;
    if (*texture)
    {
        wait_idle(s);
        ID3D12Resource_Release(*texture);
        *texture = NULL;
    }
    if (FAILED(ID3D12Device_CreateCommittedResource(s->device, &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void **)texture)))
        return FALSE;
    *tw = width;
    *th = height;
    return TRUE;
}

static BOOL ensure_convert(struct present_queue *s, UINT width, UINT height)
{
    D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
                                 DXGI_FORMAT_R8G8B8A8_UNORM, { 1, 0 }, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };

    if (s->convert && s->convert_w == width && s->convert_h == height)
        return TRUE;
    if (s->convert)
    {
        wait_idle(s);
        ID3D12Resource_Release(s->convert);
        s->convert = NULL;
    }
    if (FAILED(ID3D12Device_CreateCommittedResource(s->device, &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL, &IID_ID3D12Resource, (void **)&s->convert)))
        return FALSE;
    s->convert_w = width;
    s->convert_h = height;
    return TRUE;
}

static void barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *res, D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };

    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &b);
}

HRESULT present_x(ID3D12CommandQueue *queue, struct present_queue **state, UINT planes, const void *plane_params,
        const void *params)
{
    struct present_queue *s = *state;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    ID3D12Resource *plane, *src, *back = NULL, *readback = NULL;
    D3D12_RESOURCE_DESC desc;
    DXGI_FORMAT format;
    UINT kind = TRACK_LEGACY, from = D3D12_RESOURCE_STATE_COMMON;
    BOOL show, dump, convert, scale = FALSE;
    struct debug_view_pick view = { 0 };
    UINT out_w, out_h;
    UINT64 frame, total = 0;
    /* REBOX_DUMP_GPU_TEXTURE=<hex title address>: at each dumped frame, also the GPU contents of the
     * texture placed there (DIR\texture_<n>.bmp), and its mean byte value in the log: whether a
     * target the title samples was drawn at all. */
    ID3D12Resource *gpu_tex = NULL, *gpu_readback = NULL;
    D3D12_RESOURCE_DESC gpu_desc;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT gpu_fp;
    UINT64 gpu_total = 0, gpu_row_bytes = 0;
    UINT gpu_rows = 0;
    unsigned r;
    static LONG logged;

    if (!InterlockedExchange(&logged, 1))
    {
        const UINT64 *q = plane_params, *p = params;
        rebox_log("d3d12_xs: first PresentX: %u planes, plane params %p [%#llx %#llx %#llx %#llx %#llx %#llx], "
                     "params %p [%#llx %#llx %#llx %#llx]", planes, plane_params,
                     q ? q[0] : 0, q ? q[1] : 0, q ? q[2] : 0, q ? q[3] : 0, q ? q[4] : 0, q ? q[5] : 0,
                     params, p ? p[0] : 0, p ? p[1] : 0, p ? p[2] : 0, p ? p[3] : 0);
    }
    proxy_frame_presented();
    {
        /* xmem.dll's hang watchdog (crash.c). */
        static void (WINAPI *heartbeat)(void);
        static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
        BOOL pending = FALSE;
        if (InitOnceBeginInitialize(&once, 0, &pending, NULL) && pending)
        {
            HMODULE xmem = GetModuleHandleA("xmem.dll");
            heartbeat = xmem ? (void *)GetProcAddress(xmem, "ReboxHeartbeat") : NULL;
            InitOnceComplete(&once, 0, NULL);
        }
        if (heartbeat)
            heartbeat();
    }
    if (!s && !(s = *state = present_create(queue)))
        return S_OK;

    if (!planes || !(plane = plane_resource(s->device, plane_params)))
    {
        static LONG once;
        if (!InterlockedExchange(&once, 1))
            rebox_log("d3d12_xs: PresentX: no plane resource found in the plane params; nothing shown");
        return S_OK;
    }
    frame = ++s->frames; /* frames shown, as REBOX_DUMP_FRAMES_AT counts them */
    /* A running count in the log: 1, 2, 5, 10, 20, 50, 100 ... */
    for (UINT64 step = 1; step <= frame; step *= 10)
        if (frame == step || frame == 2 * step || frame == 5 * step)
        {
            rebox_log("d3d12_xs: PresentX: %llu frames", (unsigned long long)frame);
            break;
        }
    if (planes > 1)
    {
        static LONG once;
        if (!InterlockedExchange(&once, 1))
            rebox_log("d3d12_xs: PresentX with %u planes: only plane 0 is shown (an HDR plane, if the second is its SDR GameDVR copy)", planes);
    }

    desc = ID3D12Resource_GetDesc(plane);
    {
        /* Which resource is shown, and whether it is one filled from title memory. */
        static ID3D12Resource *last_plane;
        static LONG logged;
        UINT64 address = 0;
        UINT uploads = 0;
        if (plane != last_plane && InterlockedIncrement(&logged) <= 8)
        {
            BOOL title = title_texture_info(plane, &address, &uploads);
            rebox_log("d3d12_xs: PresentX plane %p: %llux%u, format %u, flags %#x, %s%s", (void *)plane,
                      (unsigned long long)desc.Width, desc.Height, desc.Format, (UINT)desc.Flags,
                      title ? "a placed texture filled from title memory" : "GPU memory of its own",
                      title && !uploads ? " (never uploaded)" : "");
            if (title)
                rebox_log("d3d12_xs:   at title address %#llx, uploaded %u times", (unsigned long long)address, uploads);
        }
        last_plane = plane;
    }
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1)
    {
        static LONG once;
        if (!InterlockedExchange(&once, 1))
            rebox_log("d3d12_xs: PresentX: plane (dimension %u, %u samples) can't be presented",
                         desc.Dimension, desc.SampleDesc.Count);
        return S_OK;
    }
    /* Formats a swapchain can't take are converted to RGBA8 with a blit
     * (HDR values are clamped, not tone-mapped). */
    if ((convert = (format = swapchain_format(desc.Format)) == DXGI_FORMAT_UNKNOWN))
    {
        static LONG once;
        if (!InterlockedExchange(&once, 1))
            rebox_log("d3d12_xs: PresentX: plane format %u is converted to RGBA8 for display", desc.Format);
        format = DXGI_FORMAT_R8G8B8A8_UNORM;
        if (!ensure_convert(s, (UINT)desc.Width, desc.Height))
            return S_OK;
    }
    lookup(plane, &kind, &from);
    /* REBOX_DEBUG_VIEW (debug_view.c): a target of the title's in place of the plane, drawn into our
     * RGBA8 copy as the conversion would be. */
    if (debug_view_pick(&view))
    {
        if (ensure_convert(s, (UINT)desc.Width, desc.Height))
        {
            convert = TRUE;
            format = DXGI_FORMAT_R8G8B8A8_UNORM;
        }
        else
        {
            ID3D12Resource_Release(view.res);
            view.res = NULL;
        }
    }

    /* The window's client size, when it differs from the plane's: vkd3d-proton's swapchain shows a
     * 4K plane unscaled in a smaller window (a quarter of it in a 1920x1080 tile), so the plane is
     * blitted to the window's size and the swapchain made that size. Dumps stay plane-sized. */
    out_w = (UINT)desc.Width;
    out_h = desc.Height;
    {
        RECT rc;
        if (s->hwnd && env_int("REBOX_PRESENT_SCALE", 1) && GetClientRect(s->hwnd, &rc)
                && rc.right - rc.left >= 64 && rc.bottom - rc.top >= 64
                && ((UINT)(rc.right - rc.left) != out_w || (UINT)(rc.bottom - rc.top) != out_h))
        {
            out_w = rc.right - rc.left;
            out_h = rc.bottom - rc.top;
            scale = ensure_texture(s, &s->scaled, &s->scaled_w, &s->scaled_h, out_w, out_h);
            /* The scaling blit reads our RGBA8 copy, which the convert path makes and transitions. */
            if (scale && !convert && !(convert = ensure_convert(s, (UINT)desc.Width, desc.Height)))
                scale = FALSE;
            if (!scale)
                out_w = (UINT)desc.Width, out_h = desc.Height;
        }
    }
    show = ensure_swapchain(s, out_w, out_h, scale ? DXGI_FORMAT_R8G8B8A8_UNORM : format) && frame_slot_free(s);
    dump = dump_wanted(frame);
    if (!show && !dump)
    {
        if (view.res)
            ID3D12Resource_Release(view.res);
        return S_OK;
    }

    /* Reuse the oldest list once the GPU is done with it. */
    r = frame % RING;
    if (ID3D12Fence_GetCompletedValue(s->fence) < s->ring[r].done)
    {
        ID3D12Fence_SetEventOnCompletion(s->fence, s->ring[r].done, s->event);
        WaitForSingleObject(s->event, INFINITE); /* reading or reusing before the GPU is done corrupts */
    }
    ID3D12CommandAllocator_Reset(s->ring[r].alloc);
    ID3D12GraphicsCommandList_Reset(s->ring[r].list, s->ring[r].alloc, NULL);
    if (s->ring[r].viewed)
        ID3D12Resource_Release(s->ring[r].viewed);
    s->ring[r].viewed = view.res;

    src = plane;
    if (view.res)
    {
        /* Our snapshot rests in COMMON; the title's own target is where its barriers left it. */
        UINT view_kind = TRACK_LEGACY, view_from = D3D12_RESOURCE_STATE_COMMON;
        BOOL drawn;
        if (!view.snapshot)
            lookup(view.res, &view_kind, &view_from);
        transition(s->ring[r].list, view.res, view_kind, view_from, TRUE, TRUE);
        drawn = debug_view_record(s->ring[r].list, &view, s->convert, (UINT)desc.Width, desc.Height);
        transition(s->ring[r].list, view.res, view_kind, view_from, TRUE, FALSE);
        if (!drawn)
        {
            /* Not a format it can show: the plane after all. */
            transition(s->ring[r].list, plane, kind, from, TRUE, TRUE);
            ReboxBlit(s->ring[r].list, plane, (UINT)desc.Width, desc.Height, s->convert, (UINT)desc.Width, desc.Height);
            transition(s->ring[r].list, plane, kind, from, TRUE, FALSE);
        }
        barrier(s->ring[r].list, s->convert, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        src = s->convert;
        desc = ID3D12Resource_GetDesc(src);
    }
    else
        transition(s->ring[r].list, plane, kind, from, convert, TRUE);
    if (convert && !view.res)
    {
        ReboxBlit(s->ring[r].list, plane, (UINT)desc.Width, desc.Height, s->convert, (UINT)desc.Width, desc.Height);
        barrier(s->ring[r].list, s->convert, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        src = s->convert;
        desc = ID3D12Resource_GetDesc(src);
    }
    if (show && SUCCEEDED(IDXGISwapChain3_GetBuffer(s->swapchain, IDXGISwapChain3_GetCurrentBackBufferIndex(s->swapchain),
                                                    &IID_ID3D12Resource, (void **)&back)))
    {
        barrier(s->ring[r].list, back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        if (scale)
        {
            if (src == s->convert)
                barrier(s->ring[r].list, src, D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ReboxBlit(s->ring[r].list, src, (UINT)desc.Width, desc.Height, s->scaled, out_w, out_h);
            if (src == s->convert)
                barrier(s->ring[r].list, src, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_SOURCE);
            barrier(s->ring[r].list, s->scaled, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            ID3D12GraphicsCommandList_CopyResource(s->ring[r].list, back, s->scaled);
            barrier(s->ring[r].list, s->scaled, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else
            ID3D12GraphicsCommandList_CopyResource(s->ring[r].list, back, src);
        barrier(s->ring[r].list, back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
    }
    if (dump)
    {
        D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_READBACK };
        D3D12_RESOURCE_DESC bd = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, 0, 1, 1, 1, DXGI_FORMAT_UNKNOWN, { 1, 0 },
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };

        ID3D12Device_GetCopyableFootprints(s->device, &desc, 0, 1, 0, &fp, NULL, NULL, &total);
        bd.Width = total;
        if (SUCCEEDED(ID3D12Device_CreateCommittedResource(s->device, &heap, D3D12_HEAP_FLAG_NONE, &bd,
                D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&readback)))
        {
            D3D12_TEXTURE_COPY_LOCATION dst = { readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT };
            D3D12_TEXTURE_COPY_LOCATION from_loc = { src, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };

            dst.PlacedFootprint = fp;
            from_loc.SubresourceIndex = 0;
            ID3D12GraphicsCommandList_CopyTextureRegion(s->ring[r].list, &dst, 0, 0, 0, &from_loc, NULL);
        }
        {
            const char *v = getenv("REBOX_DUMP_GPU_TEXTURE");
            UINT64 at = v ? _strtoui64(v, NULL, 16) : 0;
            if (at && (gpu_tex = title_texture_at(at)))
            {
                gpu_desc = ID3D12Resource_GetDesc(gpu_tex);
                ID3D12Device_GetCopyableFootprints(s->device, &gpu_desc, 0, 1, 0, &gpu_fp, &gpu_rows, &gpu_row_bytes, &gpu_total);
                bd.Width = gpu_total;
                if (gpu_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                        && SUCCEEDED(ID3D12Device_CreateCommittedResource(s->device, &heap, D3D12_HEAP_FLAG_NONE, &bd,
                                D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&gpu_readback)))
                {
                    D3D12_TEXTURE_COPY_LOCATION dst = { gpu_readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT };
                    D3D12_TEXTURE_COPY_LOCATION from_loc = { gpu_tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };
                    dst.PlacedFootprint = gpu_fp;
                    /* Its state is the title's: taken as shader readable, as the composite samples it. */
                    barrier(s->ring[r].list, gpu_tex, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COPY_SOURCE);
                    ID3D12GraphicsCommandList_CopyTextureRegion(s->ring[r].list, &dst, 0, 0, 0, &from_loc, NULL);
                    barrier(s->ring[r].list, gpu_tex, D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
                }
            }
        }
    }
    if (convert)
        barrier(s->ring[r].list, s->convert, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!view.res)
        transition(s->ring[r].list, plane, kind, from, convert, FALSE);

    if (SUCCEEDED(ID3D12GraphicsCommandList_Close(s->ring[r].list)))
    {
        ID3D12CommandList *lists[] = { (ID3D12CommandList *)s->ring[r].list };
        ID3D12CommandQueue_ExecuteCommandLists(s->queue, 1, lists);
        if (back)
            IDXGISwapChain3_Present(s->swapchain, env_int("REBOX_VSYNC", 1), 0);
        ID3D12CommandQueue_Signal(s->queue, s->fence, ++s->fence_value);
        s->ring[r].done = s->fence_value;
    }
    if (back)
        ID3D12Resource_Release(back);

    if (readback)
    {
        void *data;
        D3D12_RANGE range = { 0, (SIZE_T)total };

        wait_idle(s);
        if (SUCCEEDED(ID3D12Resource_Map(readback, 0, &range, &data)))
        {
            write_dump(frame, &desc, &fp, data);
            ID3D12Resource_Unmap(readback, 0, NULL);
        }
        ID3D12Resource_Release(readback);
    }
    if (gpu_readback)
    {
        void *data;
        D3D12_RANGE range = { 0, (SIZE_T)gpu_total };

        wait_idle(s);
        if (SUCCEEDED(ID3D12Resource_Map(gpu_readback, 0, &range, &data)))
        {
            UINT64 sum = 0, n = 0;
            for (UINT y = 0; y < gpu_rows; y += 16)
            {
                const BYTE *row = (const BYTE *)data + gpu_fp.Offset + (SIZE_T)y * gpu_fp.Footprint.RowPitch;
                for (UINT x = 0; x + 2 < gpu_row_bytes; x += 64, n++)
                    sum += row[x] + row[x + 1] + row[x + 2];
            }
            rebox_log("d3d12_xs: frame %llu: the GPU texture at %s (%llux%u, format %u) has mean %.2f",
                      (unsigned long long)frame, getenv("REBOX_DUMP_GPU_TEXTURE"),
                      (unsigned long long)gpu_desc.Width, gpu_desc.Height, gpu_desc.Format,
                      n ? (double)sum / (3.0 * n) : 0.0);
            write_dump_named("texture", frame, &gpu_desc, &gpu_fp, gpu_rows, data);
            ID3D12Resource_Unmap(gpu_readback, 0, NULL);
        }
        ID3D12Resource_Release(gpu_readback);
    }
    return S_OK;
}

void present_queue_destroy(struct present_queue *s)
{
    if (!s)
        return;
    if (s->fence)
        wait_idle(s);
    if (s->swapchain)
        IDXGISwapChain3_Release(s->swapchain);
    if (s->latency)
        CloseHandle(s->latency);
    if (s->convert)
        ID3D12Resource_Release(s->convert);
    if (s->scaled)
        ID3D12Resource_Release(s->scaled);
    if (s->hwnd && s->own_window)
        PostMessageW(s->hwnd, WM_QUIT, 0, 0);
    for (int i = 0; i < RING; i++)
    {
        if (s->ring[i].list)
            ID3D12GraphicsCommandList_Release(s->ring[i].list);
        if (s->ring[i].alloc)
            ID3D12CommandAllocator_Release(s->ring[i].alloc);
    }
    if (s->fence)
        ID3D12Fence_Release(s->fence);
    if (s->event)
        CloseHandle(s->event);
    if (s->device)
        ID3D12Device_Release(s->device);
    ID3D12CommandQueue_Release(s->queue);
    free(s);
}
