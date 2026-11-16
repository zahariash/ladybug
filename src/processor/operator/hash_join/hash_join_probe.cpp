#include "processor/operator/hash_join/hash_join_probe.h"

#include "binder/expression/expression_util.h"
#include "function/hash/hash_functions.h"
#include "processor/execution_context.h"
#include "processor/operator/flatten.h"
#include "storage/buffer_manager/memory_manager.h"

#if !defined(__GNUC__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

using namespace lbug::common;

namespace lbug {
namespace processor {

std::string HashJoinProbePrintInfo::toString() const {
    std::string result = "Keys: ";
    result += binder::ExpressionUtil::toString(keys);
    return result;
}

static void prefetchForRead(const void* address) {
#if defined(__GNUC__)
    __builtin_prefetch(address);
#elif defined(_M_X64) || defined(_M_IX86)
    _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#else
    (void)address;
#endif
}

void HashJoinProbe::initLocalStateInternal(ResultSet* resultSet, ExecutionContext* context) {
    // Clear member vectors so re-execution on a reused operator starts clean.
    keyVectors.clear();
    vectorsToReadInto.clear();
    probeState = std::make_unique<ProbeState>();
    for (auto& keyDataPos : probeDataInfo.keysDataPos) {
        keyVectors.push_back(resultSet->getValueVector(keyDataPos).get());
    }
    if (probeDataInfo.markDataPos.isValid()) {
        markVector = resultSet->getValueVector(probeDataInfo.markDataPos).get();
    } else {
        markVector = nullptr;
    }
    for (auto& dataPos : probeDataInfo.payloadsOutPos) {
        vectorsToReadInto.push_back(resultSet->getValueVector(dataPos).get());
    }
    // We only need to read nonKeys from the factorizedTable. Key columns are always kept as first k
    // columns in the factorizedTable, so we skip the first k columns.
    DASSERT(probeDataInfo.keysDataPos.size() + probeDataInfo.getNumPayloads() + 2 ==
            sharedState->getHashTable()->getTableSchema()->getNumColumns());
    columnIdxsToReadFrom.resize(probeDataInfo.getNumPayloads());
    iota(columnIdxsToReadFrom.begin(), columnIdxsToReadFrom.end(),
        probeDataInfo.keysDataPos.size());
    auto mm = storage::MemoryManager::Get(*context->clientContext);
    hashVector = std::make_unique<ValueVector>(LogicalType::HASH(), mm);
    if (keyVectors.size() > 1) {
        tmpHashVector = std::make_unique<ValueVector>(LogicalType::HASH(), mm);
    }
    keyFlatten = nullptr;
    prefetchSizeChecked = false;
    prefetchBatchSeq = 0;
    numSampledProbes = 0;
    numShortBatchProbes = 0;
    // Only a Flatten right below the probe: anything in between could rewrite the key after the
    // batch was hashed.
    auto child = children[0].get();
    if (flatProbe && keyVectors.size() == 1 &&
        keyVectors[0]->dataType.getPhysicalType() == PhysicalTypeID::INTERNAL_ID &&
        child->getOperatorType() == PhysicalOperatorType::FLATTEN &&
        child->constCast<Flatten>().getDataChunkToFlattenPos() ==
            probeDataInfo.keysDataPos[0].dataChunkPos) {
        keyFlatten = &child->constCast<Flatten>();
    }
}

bool HashJoinProbe::probeFlatKeyWithPrefetch() {
    auto& hashTable = *sharedState->getHashTable();
    if (!prefetchSizeChecked) {
        prefetchSizeChecked = true;
        if (hashTable.getNumEntries() == 0 || hashTable.getCapacity() == 0 ||
            hashTable.getProbedBytes() <= PREFETCH_MIN_TABLE_BYTES) {
            keyFlatten = nullptr;
            return false;
        }
        // Kept across batches, so resize does not allocate again.
        prefetchHashes.reserve(DEFAULT_VECTOR_CAPACITY);
    }
    const auto batch = keyFlatten->getBatchView();
    auto& keyVector = *keyVectors[0];
    // A batch this short has no key far enough ahead to prefetch. Stop checking if most of the
    // first probes come from such batches.
    const auto shortBatch = batch.size <= PREFETCH_DISTANCE;
    if (numSampledProbes < PREFETCH_SAMPLE_PROBES) {
        numSampledProbes++;
        numShortBatchProbes += shortBatch;
        if (numSampledProbes == PREFETCH_SAMPLE_PROBES &&
            2 * numShortBatchProbes > PREFETCH_SAMPLE_PROBES) {
            keyFlatten = nullptr;
        }
    }
    if (shortBatch || batch.nextIdx == 0 || batch.nextIdx > batch.size ||
        keyVector.state->getSelVector()[0] != (*batch.selVector)[batch.nextIdx - 1]) {
        return false;
    }
    if (batch.seq != prefetchBatchSeq) {
        prefetchBatchSeq = batch.seq;
        prefetchHashes.resize(batch.size);
        for (auto i = 0u; i < batch.size; i++) {
            const auto pos = (*batch.selVector)[i];
            prefetchHashes[i] = 0;
            if (!keyVector.isNull(pos)) {
                function::Hash::operation(keyVector.getValue<internalID_t>(pos), prefetchHashes[i]);
            }
        }
    }
    const auto idx = batch.nextIdx - 1;
    if (idx + 2 * PREFETCH_DISTANCE < batch.size) {
        prefetchForRead(reinterpret_cast<const char*>(
            hashTable.getSlotForHash(prefetchHashes[idx + 2 * PREFETCH_DISTANCE])));
    }
    if (idx + PREFETCH_DISTANCE < batch.size) {
        if (auto chainHead = hashTable.getTupleForHash(prefetchHashes[idx + PREFETCH_DISTANCE])) {
            prefetchForRead(chainHead);
        }
    }
    // The probe itself reads the key's current value; the batch hashes only steer prefetches.
    const auto pos = keyVector.state->getSelVector()[0];
    if (keyVector.isNull(pos)) {
        probeState->probedTuples[0] = nullptr;
    } else {
        // The INTERNAL_ID hash that VectorHashFunction computes in JoinHashTable::probe.
        hash_t hash = 0;
        function::Hash::operation(keyVector.getValue<internalID_t>(pos), hash);
        probeState->probedTuples[0] = hashTable.getTupleForHash(hash);
    }
    return true;
}

bool HashJoinProbe::getMatchedTuplesForFlatKey(ExecutionContext* context) {
    if (probeState->nextMatchedTupleIdx < probeState->matchedSelVector.getSelSize()) {
        // Not all matched tuples have been shipped. Continue shipping.
        return true;
    }
    if (probeState->probedTuples[0] == nullptr) { // No more matched tuples on the chain.
        if (!children[0]->getNextTuple(context)) {
            return false;
        }
        if (keyFlatten == nullptr || !probeFlatKeyWithPrefetch()) {
            sharedState->getHashTable()->probe(keyVectors, *hashVector, hashSelVec,
                tmpHashVector.get(), probeState->probedTuples.get());
        }
    }
    auto numMatchedTuples = sharedState->getHashTable()->matchFlatKeys(keyVectors,
        probeState->probedTuples.get(), probeState->matchedTuples.get());
    probeState->matchedSelVector.setSelSize(numMatchedTuples);
    probeState->nextMatchedTupleIdx = 0;
    return true;
}

bool HashJoinProbe::getMatchedTuplesForUnFlatKey(ExecutionContext* context) {
    DASSERT(keyVectors.size() == 1);
    auto keyVector = keyVectors[0];
    restoreSelVector(*keyVector->state);
    if (!children[0]->getNextTuple(context)) {
        return false;
    }
    saveSelVector(*keyVector->state);
    sharedState->getHashTable()->probe(keyVectors, *hashVector, hashSelVec, tmpHashVector.get(),
        probeState->probedTuples.get());
    auto numMatchedTuples =
        sharedState->getHashTable()->matchUnFlatKey(keyVector, probeState->probedTuples.get(),
            probeState->matchedTuples.get(), probeState->matchedSelVector);
    probeState->matchedSelVector.setSelSize(numMatchedTuples);
    probeState->nextMatchedTupleIdx = 0;
    return true;
}

uint64_t HashJoinProbe::getInnerJoinResultForFlatKey() {
    if (probeState->matchedSelVector.getSelSize() == 0) {
        return 0;
    }
    auto numTuplesToRead = 1;
    sharedState->getHashTable()->lookup(vectorsToReadInto, columnIdxsToReadFrom,
        probeState->matchedTuples.get(), probeState->nextMatchedTupleIdx, numTuplesToRead);
    probeState->nextMatchedTupleIdx += numTuplesToRead;
    return numTuplesToRead;
}

uint64_t HashJoinProbe::getInnerJoinResultForUnFlatKey() {
    auto numTuplesToRead = probeState->matchedSelVector.getSelSize();
    if (numTuplesToRead == 0) {
        return 0;
    }
    auto& keySelVector = keyVectors[0]->state->getSelVectorUnsafe();
    if (keySelVector.getSelSize() != numTuplesToRead) {
        // Some keys have no matched tuple. So we modify selected position.
        auto buffer = keySelVector.getMutableBuffer();
        for (auto i = 0u; i < numTuplesToRead; i++) {
            buffer[i] = probeState->matchedSelVector[i];
        }
        keySelVector.setToFiltered(numTuplesToRead);
    }
    sharedState->getHashTable()->lookup(vectorsToReadInto, columnIdxsToReadFrom,
        probeState->matchedTuples.get(), probeState->nextMatchedTupleIdx, numTuplesToRead);
    probeState->nextMatchedTupleIdx += numTuplesToRead;
    return numTuplesToRead;
}

static void writeLeftJoinMarkVector(ValueVector* markVector, bool flag) {
    if (markVector == nullptr) {
        return;
    }
    DASSERT(markVector->state->getSelVector().getSelSize() == 1);
    auto pos = markVector->state->getSelVector()[0];
    markVector->setValue<bool>(pos, flag);
}

uint64_t HashJoinProbe::getLeftJoinResult() {
    if (getInnerJoinResult() == 0) {
        for (auto& vector : vectorsToReadInto) {
            vector->setAsSingleNullEntry();
        }
        probeState->probedTuples[0] = nullptr;
        writeLeftJoinMarkVector(markVector, false);
        return 1;
    }
    writeLeftJoinMarkVector(markVector, true);
    return 1;
}

uint64_t HashJoinProbe::getCountJoinResult() {
    DASSERT(vectorsToReadInto.size() == 1);
    if (getInnerJoinResult() == 0) {
        auto pos = vectorsToReadInto[0]->state->getSelVector()[0];
        vectorsToReadInto[0]->setValue<int64_t>(pos, 0);
        probeState->probedTuples[0] = nullptr;
    }
    return 1;
}

uint64_t HashJoinProbe::getMarkJoinResult() {
    auto markValues = (bool*)markVector->getData();
    if (markVector->state->isFlat()) {
        auto pos = markVector->state->getSelVector()[0];
        markValues[pos] = probeState->matchedSelVector.getSelSize() != 0;
    } else {
        std::fill(markValues, markValues + DEFAULT_VECTOR_CAPACITY, false);
        for (auto i = 0u; i < probeState->matchedSelVector.getSelSize(); i++) {
            auto pos = probeState->matchedSelVector[i];
            markValues[pos] = true;
        }
    }
    probeState->probedTuples[0] = nullptr;
    probeState->nextMatchedTupleIdx = probeState->matchedSelVector.getSelSize();
    return 1;
}

uint64_t HashJoinProbe::getJoinResult() {
    switch (joinType) {
    case JoinType::LEFT: {
        return getLeftJoinResult();
    }
    case JoinType::COUNT: {
        return getCountJoinResult();
    }
    case JoinType::MARK: {
        return getMarkJoinResult();
    }
    case JoinType::INNER: {
        return getInnerJoinResult();
    }
    default:
        throw InternalException("Unimplemented join type for HashJoinProbe::getJoinResult()");
    }
}

// The general flow of a hash join probe:
// 1) find matched tuples of probe side key from ht.
// 2) populate values from matched tuples into resultKeyDataChunk , buildSideFlatResultDataChunk
// (all flat data chunks from the build side are merged into one) and buildSideVectorPtrs (each
// VectorPtr corresponds to one unFlat build side data chunk that is appended to the resultSet).
bool HashJoinProbe::getNextTuplesInternal(ExecutionContext* context) {
    uint64_t numPopulatedTuples = 0;
    do {
        if (!getMatchedTuples(context)) {
            return false;
        }
        numPopulatedTuples = getJoinResult();
    } while (numPopulatedTuples == 0);
    metrics->numOutputTuple.increase(numPopulatedTuples);
    return true;
}

} // namespace processor
} // namespace lbug
