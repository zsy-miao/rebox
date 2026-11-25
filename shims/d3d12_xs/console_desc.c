/* Console-format (RDNA2 / GFX10) descriptors for the shadow heaps
 * (docs/anyps5.md). AnyPS5 reads descriptors as the console wrote them, so
 * every view the title creates is also encoded here, with the field layout
 * AnyPS5's decoders use:
 *   T# (8 dwords): base >> 8 in words 0-1, FORMAT word1[20:28], WIDTH-1
 *      word1[30:31] + word2[0:11], HEIGHT-1 word2[14:27], DST_SEL word3[0:11],
 *      BASE_LEVEL word3[12:15], LAST_LEVEL word3[16:19], SW_MODE word3[20:24],
 *      TYPE word3[28:31], DEPTH word4[0:12], BASE_ARRAY word4[16:28],
 *      MAX_MIP word5[4:7];
 *   V# (4 dwords): BASE word0 + word1[0:15], STRIDE word1[16:29],
 *      NUM_RECORDS word2, DST_SEL word3[0:11], FORMAT word3[12:18],
 *      RESOURCE_LEVEL word3[24] (1 on GFX10), OOB_SELECT word3[28:29];
 *   S# (4 dwords): CLAMP_X/Y/Z word0[0:8], MAX_ANISO word0[9:11],
 *      DEPTH_COMPARE word0[12:14], MIN_LOD/MAX_LOD word1 (4.8 fixed),
 *      LOD_BIAS word2[0:13] (5.8 signed), MAG/MIN/MIP filters word2[20:27],
 *      BORDER_COLOR_TYPE word3[30:31].
 * Format codes are GFX10's (Mesa's register database). Textures are described
 * as linear (SW_MODE 0): the descriptor only has to identify the view, since
 * bindings map back to vkd3d-proton's own descriptor (lookup.c). */

#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#include <windows.h>
#include <d3d12.h>
#include <stdlib.h>
#include <string.h>

#include "console_compile.h"
#include "console_desc.h"
#include "proxy.h"

/* DST_SEL values */
#define SEL_0 0
#define SEL_1 1
#define SEL_X 4
#define SEL_Y 5
#define SEL_Z 6
#define SEL_W 7
#define SWIZ(x, y, z, w) ((x) | (y) << 3 | (z) << 6 | (w) << 9)
#define RGBA SWIZ(SEL_X, SEL_Y, SEL_Z, SEL_W)
#define BGRA SWIZ(SEL_Z, SEL_Y, SEL_X, SEL_W)
#define RGB1 SWIZ(SEL_X, SEL_Y, SEL_Z, SEL_1)
#define RG01 SWIZ(SEL_X, SEL_Y, SEL_0, SEL_1)
#define R001 SWIZ(SEL_X, SEL_0, SEL_0, SEL_1)

static const struct { DXGI_FORMAT dxgi; UINT gfx10, swizzle; } formats[] =
{
    { DXGI_FORMAT_R8G8B8A8_UNORM, 56, RGBA },      { DXGI_FORMAT_R8G8B8A8_TYPELESS, 56, RGBA },
    { DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 130, RGBA }, { DXGI_FORMAT_R8G8B8A8_SNORM, 57, RGBA },
    { DXGI_FORMAT_R8G8B8A8_UINT, 60, RGBA },
    { DXGI_FORMAT_B8G8R8A8_UNORM, 56, BGRA },      { DXGI_FORMAT_B8G8R8A8_TYPELESS, 56, BGRA },
    { DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 130, BGRA }, { DXGI_FORMAT_B8G8R8X8_UNORM, 56, SWIZ(SEL_Z, SEL_Y, SEL_X, SEL_1) },
    { DXGI_FORMAT_R16G16B16A16_FLOAT, 71, RGBA },  { DXGI_FORMAT_R16G16B16A16_TYPELESS, 71, RGBA },
    { DXGI_FORMAT_R16G16B16A16_UNORM, 65, RGBA },  { DXGI_FORMAT_R16G16B16A16_UINT, 69, RGBA },
    { DXGI_FORMAT_R32G32B32A32_FLOAT, 77, RGBA },  { DXGI_FORMAT_R32G32B32A32_TYPELESS, 77, RGBA },
    { DXGI_FORMAT_R32G32B32A32_UINT, 75, RGBA },   { DXGI_FORMAT_R32G32B32_FLOAT, 74, RGB1 },
    { DXGI_FORMAT_R32G32_FLOAT, 64, RG01 },        { DXGI_FORMAT_R32G32_UINT, 62, RG01 },
    { DXGI_FORMAT_R16G16_FLOAT, 29, RG01 },        { DXGI_FORMAT_R16G16_UNORM, 23, RG01 },
    { DXGI_FORMAT_R16G16_UINT, 27, RG01 },         { DXGI_FORMAT_R8G8_UNORM, 14, RG01 },
    { DXGI_FORMAT_R32_FLOAT, 22, R001 },           { DXGI_FORMAT_R32_TYPELESS, 22, R001 },
    { DXGI_FORMAT_D32_FLOAT, 22, R001 },           { DXGI_FORMAT_R32_UINT, 20, R001 },
    { DXGI_FORMAT_R32_SINT, 21, R001 },            { DXGI_FORMAT_R16_FLOAT, 13, R001 },
    { DXGI_FORMAT_R16_UNORM, 7, R001 },            { DXGI_FORMAT_D16_UNORM, 7, R001 },
    { DXGI_FORMAT_R16_UINT, 11, R001 },            { DXGI_FORMAT_R8_UNORM, 1, R001 },
    { DXGI_FORMAT_R8_UINT, 5, R001 },
    { DXGI_FORMAT_R10G10B10A2_UNORM, 50, RGBA },   { DXGI_FORMAT_R10G10B10A2_TYPELESS, 50, RGBA },
    { DXGI_FORMAT_R10G10B10A2_UINT, 54, RGBA },    { DXGI_FORMAT_R11G11B10_FLOAT, 36, RGB1 },
    { DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 132, RGB1 },
    { DXGI_FORMAT_BC1_UNORM, 169, RGBA },          { DXGI_FORMAT_BC1_UNORM_SRGB, 170, RGBA },
    { DXGI_FORMAT_BC2_UNORM, 171, RGBA },          { DXGI_FORMAT_BC2_UNORM_SRGB, 172, RGBA },
    { DXGI_FORMAT_BC3_UNORM, 173, RGBA },          { DXGI_FORMAT_BC3_UNORM_SRGB, 174, RGBA },
    { DXGI_FORMAT_BC4_UNORM, 175, R001 },          { DXGI_FORMAT_BC4_SNORM, 176, R001 },
    { DXGI_FORMAT_BC5_UNORM, 177, RG01 },          { DXGI_FORMAT_BC5_SNORM, 178, RG01 },
    { DXGI_FORMAT_BC6H_UF16, 179, RGB1 },          { DXGI_FORMAT_BC6H_SF16, 180, RGB1 },
    { DXGI_FORMAT_BC7_UNORM, 181, RGBA },          { DXGI_FORMAT_BC7_UNORM_SRGB, 182, RGBA },
};

static BOOL format_info(DXGI_FORMAT f, UINT *gfx10, UINT *swizzle)
{
    for (UINT i = 0; i < ARRAYSIZE(formats); i++)
        if (formats[i].dxgi == f)
        {
            *gfx10 = formats[i].gfx10;
            *swizzle = formats[i].swizzle;
            return TRUE;
        }
    *gfx10 = 56; /* unknown: describe as RGBA8 so the view is still identifiable */
    *swizzle = RGBA;
    return FALSE;
}

/* T# image types */
enum { T_1D = 8, T_2D = 9, T_3D = 10, T_CUBE = 11, T_1D_ARRAY = 12, T_2D_ARRAY = 13, T_2D_MSAA = 14, T_2D_MSAA_ARRAY = 15 };

struct view
{
    UINT type, first_mip, mips, first_layer, layers, depth;
    DXGI_FORMAT format;
};

static void encode_texture(UINT64 va, const D3D12_RESOURCE_DESC *rd, const struct view *v, UINT32 out[8])
{
    UINT gfx10, swizzle, w = (UINT)rd->Width, h = rd->Height;
    UINT last_mip = v->first_mip + (v->mips ? v->mips : 1) - 1;
    UINT64 base = va >> 8;

    format_info(v->format != DXGI_FORMAT_UNKNOWN ? v->format : rd->Format, &gfx10, &swizzle);
    memset(out, 0, 8 * sizeof(*out));
    out[0] = (UINT32)base;
    out[1] = (UINT32)(base >> 32) & 0xff;
    out[1] |= (gfx10 & 0x1ff) << 20 | ((w - 1) & 3) << 30;
    out[2] = ((w - 1) >> 2 & 0xfff) | ((h - 1) & 0x3fff) << 14;
    out[3] = swizzle | (v->first_mip & 0xf) << 12 | (last_mip & 0xf) << 16 | (v->type & 0xf) << 28;
    out[4] = ((v->type == T_3D ? v->depth - 1 : v->first_layer + v->layers - 1) & 0x1fff)
           | (v->first_layer & 0x1fff) << 16;
    out[5] = ((rd->MipLevels ? rd->MipLevels : 1) - 1) << 4;
}

static void encode_buffer(UINT64 va, UINT64 size, UINT stride, DXGI_FORMAT format, UINT32 out[8])
{
    UINT gfx10, swizzle;

    format_info(format, &gfx10, &swizzle);
    if (format == DXGI_FORMAT_UNKNOWN)
    {
        gfx10 = 20; /* raw and structured buffers: 32-bit elements */
        swizzle = RGBA;
    }
    memset(out, 0, 8 * sizeof(*out));
    out[0] = (UINT32)va;
    out[1] = (UINT32)(va >> 32) & 0xffff;
    out[1] |= (stride & 0x3fff) << 16;
    out[2] = stride ? (UINT32)(size / stride) : (UINT32)size;
    out[3] = swizzle | (gfx10 & 0x7f) << 12 | 1u << 24 /* RESOURCE_LEVEL */ | 3u << 28 /* OOB_SELECT: raw */;
}

/* A buffer's address is its GPU VA. Textures have none here (and asking
 * proxy.c's GetGPUVirtualAddress would give them scratch memory), so each gets
 * a synthetic one, unique per resource object, above every real VA: the T#
 * only has to tell views apart. */
static SRWLOCK texture_lock = SRWLOCK_INIT;
static struct texture_id { ID3D12Resource *res; UINT64 va; struct texture_id *next, *va_next; } *texture_ids[4096];
/* The same by address: console_texture_of, for every T# a draw binds that isn't in a heap (one the
 * title wrote itself), walked every texture ever viewed, tens of thousands in a world. */
#define TEXTURE_VA_BUCKETS 16384
static struct texture_id *texture_vas[TEXTURE_VA_BUCKETS];

static SIZE_T texture_va_bucket(UINT64 va)
{
    return (SIZE_T)((va >> 24) * 0x9e3779b97f4a7c15ull >> 50);
}
static UINT64 next_texture_va = 1ull << 47;

static UINT64 resource_va(ID3D12Resource *res, const D3D12_RESOURCE_DESC *rd)
{
    struct texture_id **bucket, *t;
    UINT64 va = 0;

    if (rd->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        return ID3D12Resource_GetGPUVirtualAddress(res);
    bucket = &texture_ids[((ULONG_PTR)res >> 4) & 4095];
    AcquireSRWLockExclusive(&texture_lock);
    for (t = *bucket; t && t->res != res; t = t->next)
        ;
    if (!t && (t = malloc(sizeof(*t))))
    {
        /* A released resource's pointer can come back and reuse its address;
         * its views are rewritten anyway. */
        t->res = res;
        t->va = next_texture_va;
        next_texture_va += 1ull << 24;
        t->next = *bucket;
        *bucket = t;
        t->va_next = texture_vas[texture_va_bucket(t->va)];
        texture_vas[texture_va_bucket(t->va)] = t;
        console_bindings_changed(TRUE);
    }
    if (t)
        va = t->va;
    ReleaseSRWLockExclusive(&texture_lock);
    return va;
}

/* res is being released: a resource made later at its pointer gets an address of its own, so T#s
 * of the old one aren't taken for the new one's. */
void console_texture_forget(ID3D12Resource *res)
{
    struct texture_id **p, *t;

    AcquireSRWLockExclusive(&texture_lock);
    for (p = &texture_ids[((ULONG_PTR)res >> 4) & 4095]; (t = *p); p = &t->next)
        if (t->res == res)
        {
            struct texture_id **v;
            *p = t->next;
            for (v = &texture_vas[texture_va_bucket(t->va)]; *v; v = &(*v)->va_next)
                if (*v == t)
                {
                    *v = t->va_next;
                    break;
                }
            free(t);
            console_bindings_changed(TRUE);
            break;
        }
    ReleaseSRWLockExclusive(&texture_lock);
}

/* For tests: console_texture_of. */
ID3D12Resource *WINAPI ReboxConsoleTextureOf(const UINT32 words[2])
{
    return console_texture_of(words);
}

ID3D12Resource *console_texture_by_id(const UINT32 words[2])
{
    UINT64 va = ((UINT64)(words[1] & 0xff) << 32 | words[0]) << 8;
    ID3D12Resource *res = NULL;
    struct texture_id *t;

    AcquireSRWLockShared(&texture_lock);
    for (t = texture_vas[texture_va_bucket(va)]; t && !res; t = t->va_next)
        if (t->va == va)
            res = t->res;
    ReleaseSRWLockShared(&texture_lock);
    return res;
}

ID3D12Resource *console_texture_of(const UINT32 words[2])
{
    UINT64 va = ((UINT64)(words[1] & 0xff) << 32 | words[0]) << 8;
    ID3D12Resource *res = NULL;
    struct texture_id *t;

    AcquireSRWLockShared(&texture_lock);
    for (t = texture_vas[texture_va_bucket(va)]; t && !res; t = t->va_next)
        if (t->va == va)
            res = t->res;
    ReleaseSRWLockShared(&texture_lock);
    /* A T# the title wrote itself holds the texture's address as the title knows it. */
    return res ? res : proxy_texture_at(va);
}

void console_encode_srv(ID3D12Resource *res, const D3D12_SHADER_RESOURCE_VIEW_DESC *d, UINT32 out[8])
{
    D3D12_RESOURCE_DESC rd;
    struct view v = { 0 };

    memset(out, 0, 8 * sizeof(*out));
    if (!res)
        return; /* null descriptor */
    rd = ID3D12Resource_GetDesc(res);
    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        UINT stride = d && d->Buffer.StructureByteStride ? d->Buffer.StructureByteStride : 4;
        UINT64 first = d ? d->Buffer.FirstElement : 0, count = d ? d->Buffer.NumElements : rd.Width / stride;
        encode_buffer(resource_va(res, &rd) + first * stride, count * stride, stride, d ? d->Format : DXGI_FORMAT_UNKNOWN,
                      out);
        return;
    }
    v.format = d ? d->Format : rd.Format;
    v.mips = rd.MipLevels;
    v.layers = rd.DepthOrArraySize;
    v.depth = rd.DepthOrArraySize;
    switch (d ? d->ViewDimension : D3D12_SRV_DIMENSION_TEXTURE2D)
    {
    case D3D12_SRV_DIMENSION_TEXTURE1D:
        v.type = T_1D; v.first_mip = d->Texture1D.MostDetailedMip; v.mips = d->Texture1D.MipLevels; v.layers = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
        v.type = T_1D_ARRAY; v.first_mip = d->Texture1DArray.MostDetailedMip; v.mips = d->Texture1DArray.MipLevels;
        v.first_layer = d->Texture1DArray.FirstArraySlice; v.layers = d->Texture1DArray.ArraySize; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
        v.type = T_2D_ARRAY; v.first_mip = d->Texture2DArray.MostDetailedMip; v.mips = d->Texture2DArray.MipLevels;
        v.first_layer = d->Texture2DArray.FirstArraySlice; v.layers = d->Texture2DArray.ArraySize; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS:
        v.type = T_2D_MSAA; v.mips = 1; v.layers = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
        v.type = T_2D_MSAA_ARRAY; v.mips = 1; v.first_layer = d->Texture2DMSArray.FirstArraySlice;
        v.layers = d->Texture2DMSArray.ArraySize; break;
    case D3D12_SRV_DIMENSION_TEXTURE3D:
        v.type = T_3D; v.first_mip = d->Texture3D.MostDetailedMip; v.mips = d->Texture3D.MipLevels; v.layers = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
        v.type = T_CUBE; v.first_mip = d->TextureCube.MostDetailedMip; v.mips = d->TextureCube.MipLevels; v.layers = 6; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
        v.type = T_CUBE; v.first_mip = d->TextureCubeArray.MostDetailedMip; v.mips = d->TextureCubeArray.MipLevels;
        v.first_layer = d->TextureCubeArray.First2DArrayFace; v.layers = d->TextureCubeArray.NumCubes * 6; break;
    case D3D12_SRV_DIMENSION_TEXTURE2D:
    default:
        v.type = T_2D; v.first_mip = d ? d->Texture2D.MostDetailedMip : 0;
        v.mips = d ? d->Texture2D.MipLevels : rd.MipLevels; v.layers = 1; break;
    }
    if (v.mips == (UINT)-1 || !v.mips)
        v.mips = rd.MipLevels - v.first_mip;
    encode_texture(resource_va(res, &rd), &rd, &v, out);
}

void console_encode_uav(ID3D12Resource *res, const D3D12_UNORDERED_ACCESS_VIEW_DESC *d, UINT32 out[8])
{
    D3D12_RESOURCE_DESC rd;
    struct view v = { 0 };

    memset(out, 0, 8 * sizeof(*out));
    if (!res)
        return;
    rd = ID3D12Resource_GetDesc(res);
    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        UINT stride = d && d->Buffer.StructureByteStride ? d->Buffer.StructureByteStride : 4;
        UINT64 first = d ? d->Buffer.FirstElement : 0, count = d ? d->Buffer.NumElements : rd.Width / stride;
        encode_buffer(resource_va(res, &rd) + first * stride, count * stride, stride, d ? d->Format : DXGI_FORMAT_UNKNOWN,
                      out);
        return;
    }
    v.format = d ? d->Format : rd.Format;
    v.mips = 1;
    v.layers = 1;
    v.depth = rd.DepthOrArraySize;
    switch (d ? d->ViewDimension : D3D12_UAV_DIMENSION_TEXTURE2D)
    {
    case D3D12_UAV_DIMENSION_TEXTURE1D:
        v.type = T_1D; v.first_mip = d->Texture1D.MipSlice; break;
    case D3D12_UAV_DIMENSION_TEXTURE1DARRAY:
        v.type = T_1D_ARRAY; v.first_mip = d->Texture1DArray.MipSlice;
        v.first_layer = d->Texture1DArray.FirstArraySlice; v.layers = d->Texture1DArray.ArraySize; break;
    case D3D12_UAV_DIMENSION_TEXTURE2DARRAY:
        v.type = T_2D_ARRAY; v.first_mip = d->Texture2DArray.MipSlice;
        v.first_layer = d->Texture2DArray.FirstArraySlice; v.layers = d->Texture2DArray.ArraySize; break;
    case D3D12_UAV_DIMENSION_TEXTURE3D:
        v.type = T_3D; v.first_mip = d->Texture3D.MipSlice; break;
    case D3D12_UAV_DIMENSION_TEXTURE2D:
    default:
        v.type = T_2D; v.first_mip = d ? d->Texture2D.MipSlice : 0; break;
    }
    encode_texture(resource_va(res, &rd), &rd, &v, out);
}

/* A constant buffer as the console's runtime writes one: 16-byte rows (stride 16, the record count
 * in rows) with an index-only bounds check. A shader may read rows by index (buffer_load idxen: a
 * UI vertex shader's transform rows); with stride 0 every row was row 0 and each vertex had x = y,
 * the whole UI on one diagonal (as xcl had: ivall/xcl #175). Scalar loads are bounded by stride
 * times records, the same bytes. */
void console_encode_cbv(const D3D12_CONSTANT_BUFFER_VIEW_DESC *d, UINT32 out[8])
{
    memset(out, 0, 8 * sizeof(*out));
    if (d && d->BufferLocation)
    {
        encode_buffer(d->BufferLocation, (d->SizeInBytes + 15) & ~15u, 16, DXGI_FORMAT_R32G32B32A32_FLOAT, out);
        out[3] = (out[3] & ~(3u << 28)) | 1u << 28; /* OOB_SELECT: index only */
    }
}

static UINT clamp_mode(D3D12_TEXTURE_ADDRESS_MODE m)
{
    switch (m)
    {
    case D3D12_TEXTURE_ADDRESS_MODE_MIRROR: return 1;       /* SQ_TEX_MIRROR */
    case D3D12_TEXTURE_ADDRESS_MODE_CLAMP: return 2;        /* SQ_TEX_CLAMP_LAST_TEXEL */
    case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE: return 3;  /* SQ_TEX_MIRROR_ONCE_LAST_TEXEL */
    case D3D12_TEXTURE_ADDRESS_MODE_BORDER: return 6;       /* SQ_TEX_CLAMP_BORDER */
    case D3D12_TEXTURE_ADDRESS_MODE_WRAP:
    default: return 0;                                      /* SQ_TEX_WRAP */
    }
}

static UINT lod_4_8(float lod)
{
    if (lod < 0.0f) lod = 0.0f;
    if (lod > 15.0f) lod = 15.0f;
    return (UINT)(lod * 256.0f) & 0xfff;
}

void console_encode_sampler(const D3D12_SAMPLER_DESC *d, UINT32 out[4])
{
    UINT filter = d->Filter, min_linear = (filter >> 4) & 1, mag_linear = (filter >> 2) & 1, mip_linear = filter & 1;
    UINT aniso = (filter & 0x40) != 0, compare = (filter & 0x80) != 0, ratio = 0;
    INT bias = (INT)(d->MipLODBias * 256.0f);

    for (UINT n = d->MaxAnisotropy; aniso && n > 1 && ratio < 4; n >>= 1)
        ratio++;
    out[0] = clamp_mode(d->AddressU) | clamp_mode(d->AddressV) << 3 | clamp_mode(d->AddressW) << 6 | ratio << 9
           | (compare ? (d->ComparisonFunc - 1) & 7 : 0) << 12;
    out[1] = lod_4_8(d->MinLOD) | lod_4_8(d->MaxLOD) << 12;
    /* XY filter: 0 point, 1 bilinear, 2 aniso point, 3 aniso linear; MIP: 1 point, 2 linear. */
    out[2] = ((UINT)bias & 0x3fff) | (aniso ? 3 : mag_linear) << 20 | (aniso ? 3 : min_linear) << 22
           | (mip_linear ? 2u : 1u) << 26;
    out[3] = 0; /* BORDER_COLOR_TYPE 0: transparent black */
    if (d->BorderColor[3] == 1.0f)
        out[3] = (d->BorderColor[0] == 1.0f ? 2u : 1u) << 30; /* opaque white / opaque black */
}

static D3D12_TEXTURE_ADDRESS_MODE address_mode(UINT clamp)
{
    switch (clamp)
    {
    case 0: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case 1: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case 3: case 4: case 5: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    case 6: case 7: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
}

void console_decode_sampler(const UINT32 w[4], D3D12_SAMPLER_DESC *d)
{
    UINT mag = (w[2] >> 20) & 3, min = (w[2] >> 22) & 3, mip = (w[2] >> 26) & 3, compare = (w[0] >> 12) & 7;
    UINT aniso = mag >= 2 || min >= 2, filter = 0;
    INT bias = (INT)((w[2] & 0x3fff) << 18) >> 18;

    memset(d, 0, sizeof(*d));
    d->AddressU = address_mode(w[0] & 7);
    d->AddressV = address_mode((w[0] >> 3) & 7);
    d->AddressW = address_mode((w[0] >> 6) & 7);
    d->MaxAnisotropy = aniso ? 1u << ((w[0] >> 9) & 7) : 1;
    if (d->MaxAnisotropy > 16)
        d->MaxAnisotropy = 16;
    d->MipLODBias = bias / 256.0f;
    d->MinLOD = (w[1] & 0xfff) / 256.0f;
    d->MaxLOD = ((w[1] >> 12) & 0xfff) / 256.0f;
    if (aniso)
        filter = 0x55; /* D3D12_FILTER_ANISOTROPIC */
    else
        filter = (min & 1) << 4 | (mag & 1) << 2 | (mip == 2);
    if (compare)
    {
        filter |= 0x80;
        d->ComparisonFunc = (D3D12_COMPARISON_FUNC)(compare + 1);
    }
    else
        d->ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    d->Filter = (D3D12_FILTER)filter;
    switch (w[3] >> 30)
    {
    case 1: d->BorderColor[3] = 1.0f; break;
    case 2: d->BorderColor[0] = d->BorderColor[1] = d->BorderColor[2] = d->BorderColor[3] = 1.0f; break;
    default: break; /* transparent black; register-table colours aren't modelled */
    }
}
