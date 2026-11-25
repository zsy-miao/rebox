// Fits a recompiled shader to vkd3d-proton's bindings (docs/anyps5.md). GPL-2.0-only.
//
// AnyPS5's own driver binds a result in a descriptor set of its own (set 0, bindings
// `stage*65 + local`) and passes VkSpecializationInfo. Console pipelines here use the title's root
// signature layout instead, where vkd3d-proton already has every descriptor the title created bound
// in one runtime array per descriptor type (its bindless sets, indexed by heap slot). So:
//
// 1. Specialization constants are frozen to the result's values (SPIRV-Tools). AnyPS5 indexes
//    descriptor arrays through them, so this turns those indices into plain constants.
// 2. Each guest descriptor array `T name[count]` moves to the host's set and binding as `T name[]`,
//    and every access-chain index i becomes heap_index[i]: a constant when i is one, otherwise a
//    load from a private table of the binding's heap indices.
// 3. Each host buffer (FlattenedSrt, BDA page table, fault buffer, ...) becomes a
//    PhysicalStorageBuffer pointer read from the push block: AnyPS5's 128 bytes of push data, then
//    one 64-bit address per host buffer from byte 128; or, given its address, a constant (graphics:
//    each stage's push data takes 128 bytes of the block).
// With a draw block (sa5_finalize_draw), the heap indices of 2. and the pushed addresses of 3. are
// read from a per-draw buffer whose address is in the push block instead: the module has nothing
// that changes from draw to draw, so a title that copies its descriptors to new heap slots every
// draw (a UE5 title) gets one module and one pipeline per shader, not one per draw. A mesh module's
// draw parameters, which AnyPS5 reads from the push block, are frozen too (FreezeMeshDraw).
#define REBOX_ANYPS5_BUILD
#include <cstdlib>
#include <atomic>
#include "rebox_anyps5.h"

#include "sa5_internal.hpp"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

enum : std::uint32_t {
    OpName = 5, OpMemberName = 6, OpExtension = 10, OpMemoryModel = 14, OpEntryPoint = 15, OpCapability = 17,
    OpTypeInt = 21, OpTypeVector = 23, OpTypeArray = 28, OpTypeRuntimeArray = 29, OpTypeStruct = 30,
    OpTypePointer = 32, OpConstant = 43, OpConstantComposite = 44, OpFunction = 54, OpFunctionEnd = 56,
    OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65, OpInBoundsAccessChain = 66,
    OpPtrAccessChain = 67, OpArrayLength = 68, OpDecorate = 71, OpMemberDecorate = 72, OpCopyObject = 83,
    OpBitcast = 124, OpLabel = 248, OpCompositeConstruct = 80, OpIAdd = 128, OpIMul = 132,
};
enum : std::uint32_t {
    DecorationBlock = 2, DecorationArrayStride = 6, DecorationBinding = 33, DecorationDescriptorSet = 34,
    DecorationOffset = 35, DecorationNonUniform = 5300,
};
enum : std::uint32_t {
    StorageClassFunction = 7, StorageClassPrivate = 6, StorageClassPushConstant = 9, StorageClassStorageBuffer = 12,
    StorageClassPhysicalStorageBuffer = 5349,
};
enum : std::uint32_t {
    CapUniformBufferArrayDynamicIndexing = 28, CapSampledImageArrayDynamicIndexing = 29,
    CapStorageBufferArrayDynamicIndexing = 30, CapStorageImageArrayDynamicIndexing = 31, CapShaderNonUniform = 5301,
    CapRuntimeDescriptorArray = 5302, CapUniformTexelBufferArrayDynamicIndexing = 5304,
    CapStorageTexelBufferArrayDynamicIndexing = 5305, CapPhysicalStorageBufferAddresses = 5347,
};
constexpr std::uint32_t AddressingPhysicalStorageBuffer64 = 5348;
constexpr std::uint32_t MemoryAccessAligned = 2;
constexpr std::uint32_t HostPushOffset = 128;
constexpr std::uint32_t MaxHostBuffers = 8;

using Inst = std::vector<std::uint32_t>;

std::uint32_t Op(const Inst& inst) { return inst[0] & 0xffffu; }

void Seal(Inst& inst) { inst[0] = static_cast<std::uint32_t>(inst.size()) << 16u | Op(inst); }

Inst Make(std::uint32_t op, std::initializer_list<std::uint32_t> operands)
{
    Inst inst{op};
    inst.insert(inst.end(), operands);
    Seal(inst);
    return inst;
}

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

// The word of an instruction that holds a pointer operand, or 0: loads, stores, access chains,
// copies, array lengths, image texel pointers, function-call arguments and atomics. Other words may
// be literals that merely equal a variable's id, so only these are compared or replaced.
std::vector<std::size_t> PointerOperands(const Inst& inst)
{
    const auto op = Op(inst);
    switch (op) {
    case 61 /* OpLoad */: case 65: case 66: case 67: case 68 /* OpArrayLength */: case 83 /* OpCopyObject */:
    case 60 /* OpImageTexelPointer */:
        return {3};
    case 62 /* OpStore */:
        return {1};
    case 63 /* OpCopyMemory */:
        return {1, 2};
    case 57 /* OpFunctionCall */: {
        std::vector<std::size_t> args;
        for (std::size_t w = 4; w < inst.size(); w++) args.push_back(w);
        return args;
    }
    default:
        if (op >= 227 && op <= 242) return {3};  // OpAtomicLoad .. OpAtomicXor (OpAtomicStore: word 1)
        return {};
    }
}

std::vector<std::size_t> PointerWords(const Inst& inst)
{
    if (Op(inst) == 228 /* OpAtomicStore */) return {1};
    auto words = PointerOperands(inst);
    words.erase(std::remove_if(words.begin(), words.end(), [&](std::size_t w) { return w >= inst.size(); }), words.end());
    return words;
}

struct Module {
    std::vector<std::uint32_t> header;
    std::vector<Inst> insts;
    std::uint32_t bound = 0;

    explicit Module(const std::vector<std::uint32_t>& words)
    {
        if (words.size() < 5 || words[0] != 0x07230203u) Fail("not a SPIR-V module");
        header.assign(words.begin(), words.begin() + 5);
        bound = words[3];
        for (std::size_t i = 5; i < words.size();) {
            const std::uint32_t len = words[i] >> 16u;
            if (!len || i + len > words.size()) Fail("truncated SPIR-V instruction");
            insts.emplace_back(words.begin() + static_cast<std::ptrdiff_t>(i),
                               words.begin() + static_cast<std::ptrdiff_t>(i + len));
            i += len;
        }
    }

    std::uint32_t NewId() { return bound++; }

    std::vector<std::uint32_t> Words() const
    {
        std::vector<std::uint32_t> out = header;
        out[3] = bound;
        for (const auto& inst : insts) out.insert(out.end(), inst.begin(), inst.end());
        return out;
    }

    // The instruction defining a type (result in word 1) or a global variable or constant (word 2).
    const Inst& Def(std::uint32_t id) const
    {
        for (const auto& inst : insts) {
            const auto op = Op(inst);
            if (op >= 19 && op <= 39 && inst.size() > 1 && inst[1] == id) return inst;
            if ((op == OpVariable || op == OpConstant) && inst.size() > 2 && inst[2] == id) return inst;
        }
        Fail("SPIR-V id " + std::to_string(id) + " has no definition");
    }

    std::size_t FirstFunction() const
    {
        for (std::size_t i = 0; i < insts.size(); i++)
            if (Op(insts[i]) == OpFunction) return i;
        return insts.size();
    }

    // Types and constants go at the end of the global section, after everything they may use.
    void AddGlobal(Inst inst) { insts.insert(insts.begin() + static_cast<std::ptrdiff_t>(FirstFunction()), std::move(inst)); }

    void AddDecoration(Inst inst)
    {
        std::size_t at = 0;
        for (std::size_t i = 0; i < insts.size(); i++) {
            const auto op = Op(insts[i]);
            if (op == OpDecorate || op == OpMemberDecorate || op == OpName || op == OpMemberName || op == 16 /* OpExecutionMode */ || op == OpEntryPoint || op == OpMemoryModel || op == OpCapability || op == OpExtension || op == 11 /* OpExtInstImport */ || op == 7 /* OpString */ || op == 3 /* OpSource */) at = i + 1;
        }
        insts.insert(insts.begin() + static_cast<std::ptrdiff_t>(at), std::move(inst));
    }

    void AddCapability(std::uint32_t cap)
    {
        std::size_t last = 0;
        for (std::size_t i = 0; i < insts.size(); i++) {
            if (Op(insts[i]) != OpCapability) continue;
            if (insts[i][1] == cap) return;
            last = i + 1;
        }
        insts.insert(insts.begin() + static_cast<std::ptrdiff_t>(last), Make(OpCapability, {cap}));
    }

    void AddInterface(std::uint32_t id)
    {
        for (auto& inst : insts) {
            if (Op(inst) != OpEntryPoint) continue;
            inst.push_back(id);
            Seal(inst);
        }
    }

    std::uint32_t UintType()
    {
        for (const auto& inst : insts)
            if (Op(inst) == OpTypeInt && inst[2] == 32 && inst[3] == 0) return inst[1];
        const std::uint32_t id = NewId();
        AddGlobal(Make(OpTypeInt, {id, 32, 0}));
        return id;
    }

    std::uint32_t Constant(std::uint32_t value)
    {
        const std::uint32_t type = UintType();
        for (const auto& inst : insts)
            if (Op(inst) == OpConstant && inst.size() == 4 && inst[1] == type && inst[3] == value) return inst[2];
        const std::uint32_t id = NewId();
        AddGlobal(Make(OpConstant, {type, id, value}));
        return id;
    }

    std::uint32_t Type(std::uint32_t op, std::initializer_list<std::uint32_t> operands)
    {
        for (const auto& inst : insts) {
            if (Op(inst) != op || inst.size() != operands.size() + 2) continue;
            if (std::equal(operands.begin(), operands.end(), inst.begin() + 2)) return inst[1];
        }
        const std::uint32_t id = NewId();
        Inst inst{op, id};
        inst.insert(inst.end(), operands);
        Seal(inst);
        AddGlobal(std::move(inst));
        return id;
    }

    std::uint32_t Pointer(std::uint32_t storageClass, std::uint32_t pointee) { return Type(OpTypePointer, {storageClass, pointee}); }

    // Every variable at a set and binding. AnyPS5 declares several at one binding: a view of the
    // guest buffers per element type (buffers, buffers_coherent, buffers_u64, buffers_u8) and an
    // image table per image type (table_images_N). Each one must move, not just the first.
    std::vector<std::uint32_t> VariablesAt(std::uint32_t set, std::uint32_t binding) const
    {
        std::map<std::uint32_t, std::uint32_t> sets, bindings;
        for (const auto& inst : insts) {
            if (Op(inst) != OpDecorate || inst.size() < 4) continue;
            if (inst[2] == DecorationDescriptorSet) sets[inst[1]] = inst[3];
            if (inst[2] == DecorationBinding) bindings[inst[1]] = inst[3];
        }
        std::vector<std::uint32_t> found;
        for (const auto& [id, b] : bindings)
            if (b == binding && sets.count(id) && sets.at(id) == set) found.push_back(id);
        if (found.empty()) Fail("no variable at set " + std::to_string(set) + " binding " + std::to_string(binding));
        return found;
    }

    std::string NameOf(std::uint32_t id) const
    {
        for (const auto& inst : insts) {
            if (Op(inst) != OpName || inst.size() < 3 || inst[1] != id) continue;
            std::string name;
            for (std::size_t w = 2; w < inst.size(); w++)
                for (std::uint32_t b = 0; b < 4; b++) {
                    const char c = static_cast<char>((inst[w] >> (8u * b)) & 0xffu);
                    if (!c) return name;
                    name += c;
                }
            return name;
        }
        return "%" + std::to_string(id);
    }
};

std::pair<std::uint32_t, std::uint32_t> PushBlock(Module& m, std::uint32_t base, std::uint32_t slots);

// The per-draw block (sa5_finalize_draw): {uint64 addresses[SA5_DRAW_ADDRESSES]; uint32 indices[]},
// read as a uint array through a PhysicalStorageBuffer pointer that is the push block's one address.
struct DrawBlock {
    std::uint32_t pushVar = 0, memberConstant = 0, pushUvec2 = 0, uvec2 = 0, uintType = 0, psbBlock = 0, psbUint = 0,
                  c0 = 0;

    DrawBlock(Module& m, std::uint32_t base)
    {
        const auto [var, member] = PushBlock(m, base, 1);
        pushVar = var;
        memberConstant = m.Constant(member);
        uintType = m.UintType();
        uvec2 = m.Type(OpTypeVector, {uintType, 2});
        pushUvec2 = m.Pointer(StorageClassPushConstant, uvec2);
        const std::uint32_t runtime = m.NewId(), block = m.NewId();
        m.AddGlobal(Make(OpTypeRuntimeArray, {runtime, uintType}));
        m.AddDecoration(Make(OpDecorate, {runtime, DecorationArrayStride, 4}));
        m.AddGlobal(Make(OpTypeStruct, {block, runtime}));
        m.AddDecoration(Make(OpDecorate, {block, DecorationBlock}));
        m.AddDecoration(Make(OpMemberDecorate, {block, 0, DecorationOffset, 0}));
        psbBlock = m.Pointer(StorageClassPhysicalStorageBuffer, block);
        psbUint = m.Pointer(StorageClassPhysicalStorageBuffer, uintType);
        c0 = m.Constant(0);
        m.AddCapability(CapPhysicalStorageBufferAddresses);
        for (auto& inst : m.insts)
            if (Op(inst) == OpMemoryModel) inst[1] = AddressingPhysicalStorageBuffer64;
    }

    // Instructions loading the block's pointer; returns its id.
    std::uint32_t Pointer(Module& m, std::vector<Inst>& out) const
    {
        const std::uint32_t chain = m.NewId(), value = m.NewId(), pointer = m.NewId();
        out.push_back(Make(OpAccessChain, {pushUvec2, chain, pushVar, memberConstant, c0}));
        out.push_back(Make(OpLoad, {uvec2, value, chain}));
        out.push_back(Make(OpBitcast, {psbBlock, pointer, value}));
        return pointer;
    }

    // Instructions loading word `index` (an id) of the block; returns the loaded id.
    std::uint32_t Word(Module& m, std::vector<Inst>& out, std::uint32_t pointer, std::uint32_t index) const
    {
        const std::uint32_t chain = m.NewId(), loaded = m.NewId();
        out.push_back(Make(OpAccessChain, {psbUint, chain, pointer, c0, index}));
        out.push_back(Make(OpLoad, {uintType, loaded, chain, MemoryAccessAligned, 4}));
        return loaded;
    }
};

// ---------------------------------------------------------------- 1. specialization constants

std::vector<std::uint32_t> Freeze(const std::vector<std::uint32_t>& words,
                                  const std::vector<ShaderRecompiler::PipelineSpecializationConstant>& constants)
{
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> values;
    for (const auto& c : constants) values[c.id] = {c.value};
    std::string log;
    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_2);
    optimizer.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        log += message;
        log += "; ";
    });
    optimizer.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass(values))
        .RegisterPass(spvtools::CreateFreezeSpecConstantValuePass())
        .RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass())
        .RegisterPass(spvtools::CreateUnifyConstantPass());
    std::vector<std::uint32_t> out;
    spvtools::OptimizerOptions options;
    options.set_run_validator(false);
    if (!optimizer.Run(words.data(), words.size(), &out, options)) Fail("freezing specialization constants failed: " + log);
    return out;
}

// ---------------------------------------------------------------- 2. guest descriptor arrays

std::uint32_t DynamicIndexingCapability(std::uint32_t kind)
{
    switch (kind) {
    case 0: return CapUniformBufferArrayDynamicIndexing;
    case 1: return CapStorageBufferArrayDynamicIndexing;
    case 2: return CapUniformTexelBufferArrayDynamicIndexing;
    case 3: return CapStorageTexelBufferArrayDynamicIndexing;
    case 5: return CapStorageImageArrayDynamicIndexing;
    default: return CapSampledImageArrayDynamicIndexing;  // sampled images and samplers
    }
}

// One variable of a remapped binding. Its heap-index table (made on first dynamic use, without a
// draw block) is shared with the binding's other variables.
void RebindVariable(Module& m, const sa5_remap& remap, const DrawBlock* draw, std::uint32_t var, std::uint32_t& table,
                    std::uint32_t& privateUintPtr)
{
    const Inst varDef = m.Def(var);
    const std::uint32_t storageClass = varDef[3];
    const Inst ptrType = m.Def(varDef[1]);
    const Inst arrType = m.Def(ptrType[3]);
    // AnyPS5 declares an image table that can grow (binding 261) as T[] already.
    if (Op(arrType) != OpTypeArray && Op(arrType) != OpTypeRuntimeArray)
        Fail("binding " + std::to_string(remap.binding) + " is not a descriptor array");
    const std::uint32_t uintType = m.UintType();
    const auto heapIndex = [&](std::uint32_t i) {
        if (i >= remap.count) Fail("descriptor index " + std::to_string(i) + " out of range for binding " + std::to_string(remap.binding));
        return remap.heap_indices ? remap.heap_indices[i] : 0u;  // none with a draw block
    };

    // T[] and a pointer to it.
    const std::uint32_t runtimeArray = m.NewId(), runtimePtr = m.NewId();
    m.AddGlobal(Make(OpTypeRuntimeArray, {runtimeArray, arrType[2]}));
    m.AddGlobal(Make(OpTypePointer, {runtimePtr, storageClass, runtimeArray}));

    std::map<std::uint32_t, std::uint32_t> constantValues;
    for (const auto& inst : m.insts)
        if (Op(inst) == OpConstant && inst.size() == 4 && inst[1] == uintType) constantValues[inst[2]] = inst[3];
    std::set<std::uint32_t> nonUniform;
    for (const auto& inst : m.insts)
        if (Op(inst) == OpDecorate && inst.size() >= 3 && inst[2] == DecorationNonUniform) nonUniform.insert(inst[1]);

    // The heap-index table for dynamic indices, made on first use.
    const auto makeTable = [&] {
        if (table) return;
        const std::uint32_t arr = m.Type(OpTypeArray, {uintType, m.Constant(remap.count)});
        Inst composite{OpConstantComposite, arr, m.NewId()};
        for (std::uint32_t i = 0; i < remap.count; i++) composite.push_back(m.Constant(remap.heap_indices[i]));
        Seal(composite);
        const std::uint32_t init = composite[2];
        m.AddGlobal(std::move(composite));
        privateUintPtr = m.Pointer(StorageClassPrivate, uintType);
        table = m.NewId();
        m.AddGlobal(Make(OpVariable, {m.Pointer(StorageClassPrivate, arr), table, StorageClassPrivate, init}));
        m.AddInterface(table);
    };

    // Constants for the rewritten indices are collected first: adding globals while walking would
    // move the instructions being walked.
    std::map<std::size_t, std::uint32_t> constantIndex;
    bool dynamic = false;
    for (std::size_t i = 0; i < m.insts.size(); i++) {
        const auto& inst = m.insts[i];
        const auto op = Op(inst);
        if ((op == OpAccessChain || op == OpInBoundsAccessChain) && inst.size() >= 5 && inst[3] == var) {
            if (constantValues.count(inst[4])) constantIndex[inst[2]] = heapIndex(constantValues[inst[4]]);
            else dynamic = true;
        } else {
            for (const auto w : PointerWords(inst))
                if (inst[w] == var) Fail("binding " + std::to_string(remap.binding) + " is used by opcode " + std::to_string(op));
        }
    }
    std::map<std::uint32_t, std::uint32_t> constantIds;
    if (draw) {
        // Every index comes from the draw block: element i is word 2 * SA5_DRAW_ADDRESSES +
        // draw_index_base + i. The constants for constant elements are made first.
        constantIndex.clear();
        dynamic = true;
        for (const auto& inst : m.insts)
            if ((Op(inst) == OpAccessChain || Op(inst) == OpInBoundsAccessChain) && inst.size() >= 5 && inst[3] == var &&
                constantValues.count(inst[4])) {
                if (constantValues[inst[4]] >= remap.count) heapIndex(constantValues[inst[4]]);  // fails
                constantIds[inst[2]] = m.Constant(2u * SA5_DRAW_ADDRESSES + remap.draw_index_base + constantValues[inst[4]]);
            }
    }
    const std::uint32_t drawBase = draw ? m.Constant(2u * SA5_DRAW_ADDRESSES + remap.draw_index_base) : 0u;
    for (const auto& [result, value] : constantIndex) constantIds[result] = m.Constant(value);
    if (dynamic && !draw) makeTable();

    std::vector<Inst> out, decorations;
    out.reserve(m.insts.size() + 8);
    for (auto& inst : m.insts) {
        const auto op = Op(inst);
        if ((op == OpAccessChain || op == OpInBoundsAccessChain) && inst.size() >= 5 && inst[3] == var) {
            if (draw) {
                std::uint32_t word = constantIds.count(inst[2]) ? constantIds[inst[2]] : 0u;
                if (!word) {
                    word = m.NewId();
                    out.push_back(Make(OpIAdd, {uintType, word, inst[4], drawBase}));
                }
                const std::uint32_t pointer = draw->Pointer(m, out);
                const std::uint32_t loaded = draw->Word(m, out, pointer, word);
                if (nonUniform.count(inst[4])) decorations.push_back(Make(OpDecorate, {loaded, DecorationNonUniform}));
                inst[4] = loaded;
            } else if (constantIds.count(inst[2])) {
                inst[4] = constantIds[inst[2]];
            } else {
                const std::uint32_t slot = m.NewId(), loaded = m.NewId();
                out.push_back(Make(OpAccessChain, {privateUintPtr, slot, table, inst[4]}));
                out.push_back(Make(OpLoad, {uintType, loaded, slot}));
                if (nonUniform.count(inst[4])) decorations.push_back(Make(OpDecorate, {loaded, DecorationNonUniform}));
                inst[4] = loaded;
            }
        } else if (op == OpVariable && inst[2] == var) {
            inst[1] = runtimePtr;
        } else if (op == OpDecorate && inst.size() >= 4 && inst[1] == var) {
            if (inst[2] == DecorationDescriptorSet) inst[3] = remap.new_set;
            if (inst[2] == DecorationBinding) inst[3] = remap.new_binding;
        }
        out.push_back(std::move(inst));
    }
    m.insts = std::move(out);
    // The variable's new type was added at the end of the global section: so goes the variable.
    for (auto it = m.insts.begin(); it != m.insts.end(); ++it) {
        if (Op(*it) != OpVariable || (*it)[2] != var) continue;
        Inst moved = std::move(*it);
        m.insts.erase(it);
        m.AddGlobal(std::move(moved));
        break;
    }
    for (auto& d : decorations) m.AddDecoration(std::move(d));
    m.AddCapability(CapRuntimeDescriptorArray);
    if (dynamic) {
        m.AddCapability(DynamicIndexingCapability(remap.kind));
        if (!decorations.empty()) m.AddCapability(CapShaderNonUniform);
    }
}

void Rebind(Module& m, const sa5_remap& remap, const DrawBlock* draw, std::set<std::uint32_t>& rebound)
{
    std::uint32_t table = 0, privateUintPtr = 0;
    for (const std::uint32_t var : m.VariablesAt(remap.set, remap.binding)) {
        RebindVariable(m, remap, draw, var, table, privateUintPtr);
        rebound.insert(var);
    }
}

// Whether a caller binds only some of a module's bindings on purpose (sa5_check's offline sweeps,
// which have no heap): SA5_FINALIZE_PARTIAL=1.
bool PartialFinalize()
{
    static const bool partial = [] {
        const char* v = std::getenv("SA5_FINALIZE_PARTIAL");
        return v && *v == '1';
    }();
    return partial;
}

// After the rewrite, every variable the module uses that still has a descriptor set must be one
// Rebind moved; host buffers have no variable left. Any other is still at AnyPS5's own set and
// binding, which vkd3d-proton's layout doesn't have: RADV then reads its descriptor from wherever
// that set's address points (a GPU page fault at a constant offset past set 0). Failing the compile
// skips the draw or dispatch instead.
void CheckBindings(const Module& m, const std::set<std::uint32_t>& rebound)
{
    if (PartialFinalize()) return;
    std::map<std::uint32_t, std::uint32_t> sets, bindings;
    for (const auto& inst : m.insts) {
        if (Op(inst) != OpDecorate || inst.size() < 4) continue;
        if (inst[2] == DecorationDescriptorSet) sets[inst[1]] = inst[3];
        if (inst[2] == DecorationBinding) bindings[inst[1]] = inst[3];
    }
    std::set<std::uint32_t> used;
    for (const auto& inst : m.insts)
        for (const auto w : PointerWords(inst)) used.insert(inst[w]);
    for (const auto& inst : m.insts) {
        if (Op(inst) != OpVariable || inst.size() < 4) continue;
        const std::uint32_t id = inst[2];
        if (rebound.count(id) || !used.count(id) || (!sets.count(id) && !bindings.count(id))) continue;
        Fail("variable " + m.NameOf(id) + " is left at set " + (sets.count(id) ? std::to_string(sets[id]) : "?") +
             " binding " + (bindings.count(id) ? std::to_string(bindings[id]) : "?") +
             ", which vkd3d-proton's layout doesn't have");
    }
}

// ---------------------------------------------------------------- mesh draw parameters

// A mesh module (sa5_compile_mesh) reads AnyPS5's draw parameters from push dwords 26-31
// (MeshDrawPushOffsetBytes): index count, base vertex, base instance, index word, indirect argument
// address. Each workgroup here is one console group, never indexed or indirect: they become
// constants (no index limit, the rest 0), so the stage's push data may end at the draw block's
// address (PushBlock cuts AnyPS5's dword array there) and the bytes aren't pushed per draw.
void FreezeMeshDraw(Module& m)
{
    constexpr std::uint32_t ExecutionModelMeshEXT = 5365, First = 104 / 4, Count = 6;
    if (std::none_of(m.insts.begin(), m.insts.end(),
                     [](const Inst& inst) { return Op(inst) == OpEntryPoint && inst[1] == ExecutionModelMeshEXT; }))
        return;
    const std::uint32_t all = m.Constant(0xffffffffu), zero = m.Constant(0);
    std::set<std::uint32_t> pushVars;
    std::map<std::uint32_t, std::uint32_t> constants, chains;  // id -> value; access chain -> dword
    for (const auto& inst : m.insts) {
        if (Op(inst) == OpVariable && inst[3] == StorageClassPushConstant) pushVars.insert(inst[2]);
        if (Op(inst) == OpConstant && inst.size() == 4) constants[inst[2]] = inst[3];
    }
    std::vector<Inst> out;
    for (auto& inst : m.insts) {
        const auto op = Op(inst);
        if ((op == OpAccessChain || op == OpInBoundsAccessChain) && inst.size() == 6 && pushVars.count(inst[3])
                && constants.count(inst[4]) && constants.count(inst[5]) && !constants[inst[4]]
                && constants[inst[5]] >= First && constants[inst[5]] < First + Count) {
            chains[inst[2]] = constants[inst[5]] - First;
            continue;
        }
        if (op == OpLoad && chains.count(inst[3])) {
            out.push_back(Make(OpCopyObject, {inst[1], inst[2], chains[inst[3]] ? zero : all}));
            continue;
        }
        out.push_back(std::move(inst));
    }
    m.insts = std::move(out);
    for (const auto& inst : m.insts)
        for (const auto w : PointerWords(inst))
            if (chains.count(inst[w])) Fail("a mesh draw parameter is used other than loaded");
}

// ---------------------------------------------------------------- 3. host buffers through BDA

// The push block, with room for the host buffer addresses at byte 128: returns the variable and
// the member index of the address array.
std::pair<std::uint32_t, std::uint32_t> PushBlock(Module& m, std::uint32_t base, std::uint32_t slots)
{
    const std::uint32_t uintType = m.UintType();
    const std::uint32_t uvec2 = m.Type(OpTypeVector, {uintType, 2});
    const std::uint32_t addresses = m.NewId();
    m.AddGlobal(Make(OpTypeArray, {addresses, uvec2, m.Constant(slots)}));
    m.AddDecoration(Make(OpDecorate, {addresses, DecorationArrayStride, 8}));

    for (auto& inst : m.insts) {
        if (Op(inst) != OpVariable || inst[3] != StorageClassPushConstant) continue;
        const std::uint32_t var = inst[2];
        const std::uint32_t block = m.Def(m.Def(inst[1])[3])[1];
        // AnyPS5 declares its push data as a full 32-dword array whatever the stage's push size; a
        // graphics stage's addresses go right after its 28 (base = offset + 112), so a member that
        // runs past the base is cut to end there (AnyPS5 was told its data ends there too).
        {
            std::map<std::uint32_t, std::uint32_t> offsets, strides;
            for (const auto& d : m.insts) {
                if (Op(d) == OpMemberDecorate && d[1] == block && d[3] == DecorationOffset) offsets[d[2]] = d[4];
                if (Op(d) == OpDecorate && d[2] == DecorationArrayStride) strides[d[1]] = d[3];
            }
            Inst structDef = m.Def(block);
            for (std::uint32_t k = 0; k + 2 < structDef.size(); k++) {
                const std::uint32_t type = structDef[2 + k];
                const Inst arr = m.Def(type);
                if (Op(arr) != OpTypeArray || !offsets.count(k) || offsets[k] >= base) continue;
                const Inst length = m.Def(arr[3]);
                if (Op(length) != OpConstant) continue;
                // AnyPS5's dword array; a type it shares may carry no stride of its own.
                const std::uint32_t stride = strides.count(type) ? strides[type] : 4u;
                if (offsets[k] + length[3] * stride <= base) continue;
                const std::uint32_t shorter = m.NewId();
                m.AddGlobal(Make(OpTypeArray, {shorter, arr[2], m.Constant((base - offsets[k]) / stride)}));
                m.AddDecoration(Make(OpDecorate, {shorter, DecorationArrayStride, stride}));
                for (auto& def : m.insts)
                    if (Op(def) == OpTypeStruct && def[1] == block) {
                        def[2 + k] = shorter;
                        break;
                    }
            }
        }
        // The block gains a member. It now uses the address array, so it moves to the end of the
        // global section, with the pointer types to it and the variable.
        std::vector<Inst> moved, kept;
        std::set<std::uint32_t> movedIds{block};
        for (auto& inst2 : m.insts) {
            const auto op = Op(inst2);
            if (op == OpFunction) {
                kept.insert(kept.end(), moved.begin(), moved.end());
                moved.clear();
            }
            const bool move = (op == OpTypeStruct && inst2[1] == block) ||
                              (op == OpTypePointer && movedIds.count(inst2[3])) ||
                              (op == OpVariable && movedIds.count(inst2[1]));
            if (move) {
                if (op == OpTypePointer) movedIds.insert(inst2[1]);
                moved.push_back(std::move(inst2));
            } else {
                kept.push_back(std::move(inst2));
            }
        }
        kept.insert(kept.end(), moved.begin(), moved.end());
        m.insts = std::move(kept);
        std::uint32_t member = 0;
        for (auto& s : m.insts) {
            if (Op(s) != OpTypeStruct || s[1] != block) continue;
            member = static_cast<std::uint32_t>(s.size() - 2);
            s.push_back(addresses);
            Seal(s);
            break;
        }
        m.AddDecoration(Make(OpMemberDecorate, {block, member, DecorationOffset, base}));
        return {var, member};
    }
    // No push data: unused dwords up to the base, then the addresses.
    const std::uint32_t pad = m.Type(OpTypeArray, {uintType, m.Constant(base / 4u)});
    const std::uint32_t block = m.NewId(), var = m.NewId();
    m.AddGlobal(Make(OpTypeStruct, {block, pad, addresses}));
    m.AddDecoration(Make(OpDecorate, {pad, DecorationArrayStride, 4}));
    m.AddDecoration(Make(OpDecorate, {block, DecorationBlock}));
    m.AddDecoration(Make(OpMemberDecorate, {block, 0, DecorationOffset, 0}));
    m.AddDecoration(Make(OpMemberDecorate, {block, 1, DecorationOffset, base}));
    m.AddGlobal(Make(OpVariable, {m.Pointer(StorageClassPushConstant, block), var, StorageClassPushConstant}));
    m.AddInterface(var);
    return {var, 1};
}

// Loads and stores through physical-storage-buffer pointers declare an alignment of at least their
// largest scalar: the rewrite gives every one 4, which a 64-bit view (buffers_u64, gds_u64) needs
// to be 8.
void AlignPhysicalAccesses(Module& m)
{
    std::map<std::uint32_t, const Inst*> types;  // type id -> definition
    for (const auto& inst : m.insts) {
        const auto op = Op(inst);
        if (op >= 19 && op <= 39 && inst.size() > 1) types[inst[1]] = &inst;
    }
    std::map<std::uint32_t, std::uint32_t> scalarBytes;
    const std::function<std::uint32_t(std::uint32_t)> bytesOf = [&](std::uint32_t type) -> std::uint32_t {
        if (scalarBytes.count(type)) return scalarBytes[type];
        std::uint32_t bytes = 4;
        if (types.count(type)) {
            const Inst& t = *types[type];
            switch (Op(t)) {
            case OpTypeInt: case 22 /* OpTypeFloat */: bytes = std::max(1u, t[2] / 8u); break;
            case OpTypeVector: case OpTypeArray: case OpTypeRuntimeArray: bytes = bytesOf(t[2]); break;
            case OpTypeStruct:
                bytes = 1;
                for (std::size_t w = 2; w < t.size(); w++) bytes = std::max(bytes, bytesOf(t[w]));
                break;
            case OpTypePointer: bytes = t[2] == StorageClassPhysicalStorageBuffer ? 8u : 4u; break;
            default: break;
            }
        }
        return scalarBytes[type] = bytes;
    };
    // Each pointer value's type, from the instructions that make pointers in functions.
    std::map<std::uint32_t, std::uint32_t> pointerType;
    for (const auto& inst : m.insts) {
        const auto op = Op(inst);
        if ((op == OpAccessChain || op == OpInBoundsAccessChain || op == OpPtrAccessChain || op == OpCopyObject ||
             op == OpBitcast) && inst.size() >= 3)
            pointerType[inst[2]] = inst[1];
    }
    const auto pointee = [&](std::uint32_t pointer) -> std::uint32_t {
        if (!pointerType.count(pointer) || !types.count(pointerType[pointer])) return 0;
        const Inst& t = *types[pointerType[pointer]];
        return Op(t) == OpTypePointer && t[2] == StorageClassPhysicalStorageBuffer ? t[3] : 0u;
    };
    for (auto& inst : m.insts) {
        const auto op = Op(inst);
        std::size_t mask = 0, value = 0;
        if (op == OpLoad && inst.size() >= 6) mask = 4, value = pointee(inst[3]) ? inst[1] : 0u;
        else if (op == OpStore && inst.size() >= 5) mask = 3, value = pointee(inst[1]);
        if (!value || !(inst[mask] & MemoryAccessAligned)) continue;
        inst[mask + 1] = std::max(inst[mask + 1], bytesOf(static_cast<std::uint32_t>(value)));
    }
}

void HostBuffers(Module& m, const sa5_host_buffer* buffers, std::uint32_t count, const DrawBlock* draw)
{
    if (!count) return;
    if (count > MaxHostBuffers) Fail("too many host buffers");
    const std::uint32_t uintType = m.UintType();
    const std::uint32_t uvec2 = m.Type(OpTypeVector, {uintType, 2});
    bool pushed = false;
    std::uint32_t base = HostPushOffset, slots = 0;
    for (std::uint32_t i = 0; i < count; i++) {
        if (buffers[i].address) continue;
        pushed = true;
        if (buffers[i].push_base) base = buffers[i].push_base;
        slots = std::max(slots, buffers[i].push_slot + 1u);
        if (draw && buffers[i].push_slot >= SA5_DRAW_ADDRESSES) Fail("more host buffers than the draw block has addresses for");
    }
    if (base % 8u) Fail("host buffer addresses must be 8-byte aligned in the push block");
    // Only buffers without a baked address need the push block's address array (with a draw block,
    // its address array holds them).
    const auto [pushVar, member] = pushed && !draw ? PushBlock(m, base, base == HostPushOffset ? MaxHostBuffers : slots)
                                                   : std::pair<std::uint32_t, std::uint32_t>{0u, 0u};
    const std::uint32_t pushUvec2 = m.Pointer(StorageClassPushConstant, uvec2);

    // stride: the byte size of the block's runtime array elements. Lengths are given in dwords;
    // OpArrayLength on a view of other elements (buffers_u64, buffers_u8, gds_u64) counts those.
    // tablePtr: for a guest buffer array, the type its address table is read through (a uint array).
    struct Buffer { std::uint32_t var, block, psbPtr, slot, length; bool table; std::uint32_t address, stride, tablePtr; };
    std::vector<Buffer> list;
    std::set<std::uint32_t> vars;
    const std::uint32_t psbUint = m.Pointer(StorageClassPhysicalStorageBuffer, uintType);
    std::uint32_t uintTablePtr = 0;
    const auto elementStride = [&](std::uint32_t block) {
        const Inst s = m.Def(block);
        if (Op(s) != OpTypeStruct || s.size() < 3) return 4u;
        const std::uint32_t last = s.back();
        if (Op(m.Def(last)) != OpTypeRuntimeArray) return 4u;
        for (const auto& d : m.insts)
            if (Op(d) == OpDecorate && d.size() >= 4 && d[1] == last && d[2] == DecorationArrayStride && d[3]) return d[3];
        return 4u;
    };
    for (std::uint32_t i = 0; i < count; i++) {
        std::uint32_t address = 0;
        if (buffers[i].address) {
            address = m.NewId();
            m.AddGlobal(Make(OpConstantComposite, {uvec2, address, m.Constant(static_cast<std::uint32_t>(buffers[i].address)),
                                                   m.Constant(static_cast<std::uint32_t>(buffers[i].address >> 32u))}));
        }
        for (const std::uint32_t var : m.VariablesAt(buffers[i].set, buffers[i].binding)) {
            const Inst varDef = m.Def(var);
            if (varDef[3] != StorageClassStorageBuffer) Fail("host buffer at binding " + std::to_string(buffers[i].binding) + " is not a storage buffer");
            std::uint32_t block = m.Def(varDef[1])[3];
            if (buffers[i].guest_array) {
                // A guest buffer array: element i comes from an address table, {address, u32 count, 0}.
                const Inst arr = m.Def(block);
                if (Op(arr) != OpTypeArray) Fail("guest buffers at binding " + std::to_string(buffers[i].binding) + " are not an array");
                block = arr[2];
                const Inst s = m.Def(block);
                if (Op(s) != OpTypeStruct || s.size() != 3 || Op(m.Def(s[2])) != OpTypeRuntimeArray)
                    Fail("guest buffers at binding " + std::to_string(buffers[i].binding) + " are not runtime arrays");
            } else if (Op(m.Def(block)) != OpTypeStruct) {
                Fail("host buffer at binding " + std::to_string(buffers[i].binding) + " is an array");
            }
            const std::uint32_t psbPtr = m.Pointer(StorageClassPhysicalStorageBuffer, block);
            std::uint32_t tablePtr = psbPtr;
            if (buffers[i].guest_array && m.Def(m.Def(block)[2])[2] != uintType) {
                // A view of other elements (buffers_u64, buffers_u8): its table is still read as uints.
                if (!uintTablePtr) {
                    const std::uint32_t runtime = m.NewId(), s = m.NewId();
                    m.AddGlobal(Make(OpTypeRuntimeArray, {runtime, uintType}));
                    m.AddDecoration(Make(OpDecorate, {runtime, DecorationArrayStride, 4}));
                    m.AddGlobal(Make(OpTypeStruct, {s, runtime}));
                    m.AddDecoration(Make(OpDecorate, {s, DecorationBlock}));
                    m.AddDecoration(Make(OpMemberDecorate, {s, 0, DecorationOffset, 0}));
                    uintTablePtr = m.Pointer(StorageClassPhysicalStorageBuffer, s);
                }
                tablePtr = uintTablePtr;
            }
            list.push_back({var, block, psbPtr, buffers[i].push_slot, buffers[i].array_length,
                            buffers[i].guest_array != 0, address, elementStride(block), tablePtr});
            vars.insert(var);
        }
    }
    std::map<std::uint32_t, const Buffer*> tableOf;
    for (const auto& b : list)
        if (b.table) tableOf[b.var] = &b;
    const std::uint32_t c0 = m.Constant(0), c1 = m.Constant(1), c2 = m.Constant(2), c4 = m.Constant(4);
    // Physical-storage-buffer twins of every StorageBuffer pointer type, for derived pointers.
    std::map<std::uint32_t, std::uint32_t> psbOf;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> sbPointers;
    for (const auto& inst : m.insts)
        if (Op(inst) == OpTypePointer && inst[2] == StorageClassStorageBuffer) sbPointers.emplace_back(inst[1], inst[3]);
    for (const auto& [ptr, pointee] : sbPointers) psbOf[ptr] = m.Pointer(StorageClassPhysicalStorageBuffer, pointee);
    // Each buffer's length: a constant, or (SA5_LENGTH_IN_DRAW_BLOCK) the index of its draw block word.
    std::map<std::uint32_t, std::uint32_t> lengthConstants, lengthWords;
    // A length in dwords as a count of a view's elements: divided (wider elements) or multiplied
    // (narrower), by a constant made here, before the walk below.
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> scaleOf;  // var -> opcode (0: none), factor
    for (const auto& b : list) {
        if (b.stride == 4) scaleOf[b.var] = {0u, 0u};
        else if (b.stride % 4 == 0) scaleOf[b.var] = {134u /* OpUDiv */, m.Constant(b.stride / 4)};
        else if (4 % b.stride == 0) scaleOf[b.var] = {OpIMul, m.Constant(4 / b.stride)};
        else Fail("host buffer elements of " + std::to_string(b.stride) + " bytes");
    }
    for (const auto& b : list) {
        if (b.length != SA5_LENGTH_IN_DRAW_BLOCK) {
            const std::uint64_t elements = static_cast<std::uint64_t>(b.length) * 4u / b.stride;
            lengthConstants[b.var] = m.Constant(static_cast<std::uint32_t>(std::min<std::uint64_t>(elements, 0xffffffffu)));
            continue;
        }
        if (!draw) Fail("a host buffer's length is in a draw block, but there is none");
        if (b.slot >= SA5_DRAW_ADDRESSES) Fail("a host buffer's length is past the draw block's lengths");
        lengthWords[b.var] = m.Constant(2u * SA5_DRAW_ADDRESSES + b.slot);
    }
    std::map<std::uint32_t, std::uint32_t> slotConstants;
    for (const auto& b : list) slotConstants[b.slot] = m.Constant(b.slot);
    const std::uint32_t memberConstant = pushed && !draw ? m.Constant(member) : 0u;
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> wordConstants;  // slot -> its two words
    if (draw)
        for (const auto& b : list) wordConstants[b.slot] = {m.Constant(2u * b.slot), m.Constant(2u * b.slot + 1u)};

    std::vector<Inst> out;
    std::map<std::uint32_t, std::uint32_t> current;  // host buffer variable -> this function's pointer
    std::map<std::uint32_t, std::uint32_t> rootOf;   // derived pointer -> host buffer variable
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> elementOf;  // guest buffer pointer -> table, entry*4
    bool inFunction = false, needPrologue = false;
    std::uint32_t drawPointer = 0;  // this function's draw block pointer
    for (auto& inst : m.insts) {
        const auto op = Op(inst);
        // Drop the variables and everything that names them outside functions.
        if (op == OpVariable && vars.count(inst[2])) continue;
        if ((op == OpDecorate || op == OpName) && inst.size() > 1 && vars.count(inst[1])) continue;
        if (op == OpEntryPoint) {
            Inst kept(inst.begin(), inst.begin() + 3);
            std::size_t w = 3;
            while (w < inst.size()) {  // the name: a nul-terminated string
                kept.push_back(inst[w]);
                if ((inst[w++] >> 24u) == 0) break;
            }
            for (; w < inst.size(); w++)
                if (!vars.count(inst[w])) kept.push_back(inst[w]);
            Seal(kept);
            out.push_back(std::move(kept));
            continue;
        }
        if (op == OpFunction) {
            inFunction = true;
            needPrologue = true;
            current.clear();
            rootOf.clear();
            elementOf.clear();
        } else if (op == OpFunctionEnd) {
            inFunction = false;
        }
        if (inFunction && needPrologue && op != OpFunction && op != OpLabel && op != 55 /* OpFunctionParameter */ &&
            !(op == OpVariable && inst[3] == StorageClassFunction)) {
            // First instruction after the entry block's variables: load each host buffer's address.
            const std::uint32_t block = draw ? draw->Pointer(m, out) : 0u;
            drawPointer = block;
            for (const auto& b : list) {
                const std::uint32_t pointer = m.NewId();
                std::uint32_t value = b.address;
                if (!value && draw) {
                    const std::uint32_t lo = draw->Word(m, out, block, wordConstants[b.slot].first);
                    const std::uint32_t hi = draw->Word(m, out, block, wordConstants[b.slot].second);
                    value = m.NewId();
                    out.push_back(Make(OpCompositeConstruct, {uvec2, value, lo, hi}));
                } else if (!value) {
                    const std::uint32_t chain = m.NewId();
                    value = m.NewId();
                    out.push_back(Make(OpAccessChain, {pushUvec2, chain, pushVar, memberConstant, slotConstants[b.slot]}));
                    out.push_back(Make(OpLoad, {uvec2, value, chain}));
                }
                out.push_back(Make(OpBitcast, {b.table ? b.tablePtr : b.psbPtr, pointer, value}));
                current[b.var] = pointer;
                rootOf[pointer] = b.var;
            }
            needPrologue = false;
        }
        if (inFunction) {
            const auto derived = [&](std::uint32_t id) { return rootOf.count(id) != 0; };
            // `dwords` as the result of OpArrayLength `inst` on a view of buffer variable `var`.
            const auto lengthOf = [&](std::uint32_t var, std::uint32_t dwords) {
                const auto [scale, factor] = scaleOf[var];
                if (scale) out.push_back(Make(scale, {inst[1], inst[2], dwords, factor}));
                else out.push_back(Make(OpCopyObject, {inst[1], inst[2], dwords}));
            };
            if (op == OpArrayLength && inst.size() >= 5 && vars.count(inst[3])) {
                if (lengthConstants.count(inst[3])) {
                    out.push_back(Make(OpCopyObject, {inst[1], inst[2], lengthConstants[inst[3]]}));
                } else {
                    lengthOf(inst[3], draw->Word(m, out, drawPointer, lengthWords[inst[3]]));
                }
                continue;
            }
            if (op == OpArrayLength && inst.size() >= 5 && elementOf.count(inst[3])) {
                // A guest buffer's length: the table entry's third word.
                const auto [table, entry] = elementOf[inst[3]];
                const std::uint32_t at = m.NewId(), chain = m.NewId(), dwords = m.NewId();
                out.push_back(Make(OpIAdd, {uintType, at, entry, c2}));
                out.push_back(Make(OpAccessChain, {psbUint, chain, table, c0, at}));
                out.push_back(Make(OpLoad, {uintType, dwords, chain, MemoryAccessAligned, 4}));
                lengthOf(rootOf[inst[3]], dwords);
                continue;
            }
            if ((op == OpAccessChain || op == OpInBoundsAccessChain) && inst.size() >= 5 && tableOf.count(inst[3])) {
                // &buffers[i]...: the address in table entry i, as a pointer to the element.
                const Buffer& b = *tableOf[inst[3]];
                const std::uint32_t table = current[b.var];
                const std::uint32_t entry = m.NewId(), loChain = m.NewId(), lo = m.NewId(), hiAt = m.NewId(),
                                    hiChain = m.NewId(), hi = m.NewId(), pair = m.NewId(), pointer = m.NewId();
                out.push_back(Make(OpIMul, {uintType, entry, inst[4], c4}));
                out.push_back(Make(OpAccessChain, {psbUint, loChain, table, c0, entry}));
                out.push_back(Make(OpLoad, {uintType, lo, loChain, MemoryAccessAligned, 4}));
                out.push_back(Make(OpIAdd, {uintType, hiAt, entry, c1}));
                out.push_back(Make(OpAccessChain, {psbUint, hiChain, table, c0, hiAt}));
                out.push_back(Make(OpLoad, {uintType, hi, hiChain, MemoryAccessAligned, 4}));
                out.push_back(Make(OpCompositeConstruct, {uvec2, pair, lo, hi}));
                out.push_back(Make(OpBitcast, {b.psbPtr, pointer, pair}));
                rootOf[pointer] = b.var;
                elementOf[pointer] = {table, entry};
                if (inst.size() == 5) {
                    out.push_back(Make(OpCopyObject, {b.psbPtr, inst[2], pointer}));
                    elementOf[inst[2]] = {table, entry};
                } else {
                    if (!psbOf.count(inst[1])) Fail("pointer into a guest buffer has an unexpected type");
                    Inst chain{op, psbOf[inst[1]], inst[2], pointer};
                    chain.insert(chain.end(), inst.begin() + 5, inst.end());
                    Seal(chain);
                    out.push_back(std::move(chain));
                }
                rootOf[inst[2]] = b.var;
                continue;
            }
            for (const auto w : PointerWords(inst))
                if (vars.count(inst[w])) {
                    if (tableOf.count(inst[w])) Fail("a guest buffer array is used whole by opcode " + std::to_string(op));
                    rootOf[current[inst[w]]] = inst[w];
                    inst[w] = current[inst[w]];
                }
            if ((op == OpAccessChain || op == OpInBoundsAccessChain || op == OpPtrAccessChain || op == OpCopyObject) &&
                inst.size() >= 4 && derived(inst[3])) {
                if (!psbOf.count(inst[1])) Fail("pointer into a host buffer has an unexpected type");
                inst[1] = psbOf[inst[1]];
                rootOf[inst[2]] = rootOf[inst[3]];
            } else if (op == OpLoad && inst.size() >= 4 && derived(inst[3])) {
                if (inst.size() == 4) inst.push_back(0);
                if (!(inst[4] & MemoryAccessAligned)) {
                    inst[4] |= MemoryAccessAligned;
                    inst.insert(inst.begin() + 5, 4);  // Aligned's literal comes first (lowest bit with one)
                }
                Seal(inst);
            } else if (op == OpStore && inst.size() >= 3 && derived(inst[1])) {
                if (inst.size() == 3) inst.push_back(0);
                if (!(inst[3] & MemoryAccessAligned)) {
                    inst[3] |= MemoryAccessAligned;
                    inst.insert(inst.begin() + 4, 4);
                }
                Seal(inst);
            }
        }
        out.push_back(std::move(inst));
    }
    m.insts = std::move(out);
    m.AddCapability(CapPhysicalStorageBufferAddresses);
    for (auto& inst : m.insts)
        if (Op(inst) == OpMemoryModel) inst[1] = AddressingPhysicalStorageBuffer64;
    AlignPhysicalAccesses(m);
}

void Validate(const std::vector<std::uint32_t>& words)
{
    std::string log;
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        log += message;
        log += "; ";
    });
    spvtools::ValidatorOptions options;
    options.SetScalarBlockLayout(true);
    if (!tools.Validate(words.data(), words.size(), options)) Fail("rewritten module is invalid: " + log);
}

// Whether a finalized module is validated: every one for the first 256, then every 32nd.
// Validation is much of the allocation a compile does, and under Wine allocations take the
// process heap's one lock, so it kept a world's thousand new shaders queued for minutes.
// REBOX_SPIRV_VALIDATE=1 validates every module, =0 none.
bool ShouldValidate()
{
    static const int setting = [] {
        const char* v = std::getenv("REBOX_SPIRV_VALIDATE");
        return v && *v == '1' ? 1 : v && *v == '0' ? 0 : -1;
    }();
    static std::atomic<std::uint64_t> finalized{0};
    const std::uint64_t n = finalized.fetch_add(1, std::memory_order_relaxed);
    return setting == 1 || (setting < 0 && (n < 256 || n % 32 == 0));
}

// The rewrite after freezing: draw_block_base SA5_NO_DRAW_BLOCK is sa5_finalize's, without a draw
// block. Writes the module to *out.
void Finish(Module& m, const sa5_remap* remaps, std::uint32_t remap_count, const sa5_host_buffer* buffers,
            std::uint32_t buffer_count, std::uint32_t draw_block_base, std::uint32_t** out, std::uint32_t* out_count)
{
    if (remap_count && !remaps) Fail("bad arguments");
    if (draw_block_base != SA5_NO_DRAW_BLOCK && draw_block_base % 8u)
        Fail("the draw block's address must be 8-byte aligned in the push block");
    FreezeMeshDraw(m);
    std::set<std::uint32_t> rebound;
    if (draw_block_base == SA5_NO_DRAW_BLOCK) {
        for (std::uint32_t i = 0; i < remap_count; i++) Rebind(m, remaps[i], nullptr, rebound);
        HostBuffers(m, buffers, buffer_count, nullptr);
    } else {
        const DrawBlock draw(m, draw_block_base);
        for (std::uint32_t i = 0; i < remap_count; i++) Rebind(m, remaps[i], &draw, rebound);
        HostBuffers(m, buffers, buffer_count, &draw);
    }
    CheckBindings(m, rebound);
    const auto words = m.Words();
    if (ShouldValidate()) Validate(words);
    *out = new std::uint32_t[words.size()];
    std::memcpy(*out, words.data(), words.size() * sizeof(std::uint32_t));
    *out_count = static_cast<std::uint32_t>(words.size());
}

int Report(const std::exception& e, char* error, size_t error_size)
{
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return -1;
}

}  // namespace

extern "C" {

SA5_API int sa5_finalize(const struct sa5_result* result, const struct sa5_remap* remaps, uint32_t remap_count,
                         const struct sa5_host_buffer* buffers, uint32_t buffer_count, uint32_t** out,
                         uint32_t* out_count, char* error, size_t error_size)
{
    try {
        if (!result || !out || !out_count) Fail("bad arguments");
        Module m(Freeze(const_cast<sa5_result*>(result)->Spirv(), result->result.specialization));
        Finish(m, remaps, remap_count, buffers, buffer_count, SA5_NO_DRAW_BLOCK, out, out_count);
        return 0;
    } catch (const std::exception& e) {
        return Report(e, error, error_size);
    }
}

SA5_API int sa5_finalize_draw(const struct sa5_result* result, const struct sa5_remap* remaps, uint32_t remap_count,
                              const struct sa5_host_buffer* buffers, uint32_t buffer_count, uint32_t draw_block_base,
                              uint32_t** out, uint32_t* out_count, char* error, size_t error_size)
{
    try {
        if (!result || !out || !out_count || draw_block_base == SA5_NO_DRAW_BLOCK) Fail("bad arguments");
        Module m(Freeze(const_cast<sa5_result*>(result)->Spirv(), result->result.specialization));
        Finish(m, remaps, remap_count, buffers, buffer_count, draw_block_base, out, out_count);
        return 0;
    } catch (const std::exception& e) {
        return Report(e, error, error_size);
    }
}

SA5_API int sa5_finalize_words(const uint32_t* words, uint32_t word_count, const struct sa5_remap* remaps,
                               uint32_t remap_count, const struct sa5_host_buffer* buffers, uint32_t buffer_count,
                               uint32_t draw_block_base, uint32_t** out, uint32_t* out_count, char* error,
                               size_t error_size)
{
    try {
        if (!words || !out || !out_count) Fail("bad arguments");
        Module m(std::vector<std::uint32_t>(words, words + word_count));
        Finish(m, remaps, remap_count, buffers, buffer_count, draw_block_base, out, out_count);
        return 0;
    } catch (const std::exception& e) {
        return Report(e, error, error_size);
    }
}

SA5_API int sa5_validate(const uint32_t* words, uint32_t word_count, char* error, size_t error_size)
{
    try {
        Validate(std::vector<std::uint32_t>(words, words + word_count));
        return 0;
    } catch (const std::exception& e) {
        if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
        return -1;
    }
}

SA5_API void sa5_free_words(uint32_t* words)
{
    delete[] words;
}

}  // extern "C"
