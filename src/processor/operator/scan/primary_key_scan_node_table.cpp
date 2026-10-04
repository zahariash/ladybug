#include "processor/operator/scan/primary_key_scan_node_table.h"

#include <algorithm>
#include <limits>

#include "binder/expression/expression_util.h"
#include "processor/execution_context.h"

using namespace lbug::common;
using namespace lbug::storage;

namespace lbug {
namespace processor {

std::string PrimaryKeyScanPrintInfo::toString() const {
    std::string result = "Key: ";
    result += key;
    if (!alias.empty()) {
        result += ",Alias: ";
        result += alias;
    }
    if (!indexType.empty()) {
        result += ", Index: ";
        result += indexType;
    }
    result += ", Expressions: ";
    result += binder::ExpressionUtil::toString(expressions);
    return result;
}

idx_t PrimaryKeyScanSharedState::getTableIdx() {
    std::unique_lock lck{mtx};
    if (cursor < numTables) {
        return cursor++;
    }
    return numTables;
}

void PrimaryKeyScanNodeTable::initLocalStateInternal(ResultSet* resultSet,
    ExecutionContext* context) {
    ScanTable::initLocalStateInternal(resultSet, context);
    auto nodeIDVector = resultSet->getValueVector(opInfo.nodeIDPos).get();
    scanState = std::make_unique<NodeTableScanState>(nodeIDVector, std::vector<ValueVector*>{},
        nodeIDVector->state);
    if (indexEvaluator != nullptr) {
        indexEvaluator->init(*resultSet, context->clientContext);
    }
    if (upperBoundEvaluator != nullptr) {
        upperBoundEvaluator->init(*resultSet, context->clientContext);
    }
}

void PrimaryKeyScanNodeTable::initGlobalStateInternal(ExecutionContext* /*context*/) {
    // Repeated executions of a cached physical plan clone the operator tree but share this
    // shared state, so the table cursor must be re-armed for every execution (mirrors
    // ScanNodeTable::initGlobalStateInternal).
    sharedState->resetForReuse();
}

bool PrimaryKeyScanNodeTable::getNextTuplesInternal(ExecutionContext* context) {
    if (isRange || isKeyList) {
        return lookupRange(context);
    }
    auto transaction = transaction::Transaction::Get(*context->clientContext);
    auto tableIdx = sharedState->getTableIdx();
    if (tableIdx >= tableInfos.size()) {
        return false;
    }
    DASSERT(tableIdx < tableInfos.size());
    auto& tableInfo = tableInfos[tableIdx];
    // Look up index
    indexEvaluator->evaluate();
    auto indexVector = indexEvaluator->resultVector.get();
    auto& selVector = indexVector->state->getSelVector();
    DASSERT(selVector.getSelSize() == 1);
    auto pos = selVector.getSelectedPositions()[0];
    if (indexVector->isNull(pos)) {
        return false;
    }
    offset_t nodeOffset = 0;
    auto& table = tableInfo.table->cast<NodeTable>();
    if (!table.lookupPK(transaction, indexVector, pos, nodeOffset)) {
        return false;
    }
    auto nodeID = nodeID_t{nodeOffset, table.getTableID()};
    scanState->nodeIDVector->setValue<nodeID_t>(pos, nodeID);
    // Look up properties
    tableInfo.initScanState(*scanState, outVectors, context->clientContext);
    table.initScanState(transaction, *scanState, nodeID.tableID, nodeOffset);
    auto succeeded = table.lookup(transaction, *scanState);
    tableInfo.castColumns();
    metrics->numOutputTuple.incrementByOne();
    return succeeded;
}

bool PrimaryKeyScanNodeTable::lookupRange(ExecutionContext* context) {
    auto transaction = transaction::Transaction::Get(*context->clientContext);
    while (currentRangeTableIdx < tableInfos.size()) {
        auto& tableInfo = tableInfos[currentRangeTableIdx];
        auto& table = tableInfo.table->cast<NodeTable>();
        if (rangeOffsets.empty() && isKeyList) {
            lookupKeyList(transaction, table);
            if (rangeOffsets.empty()) {
                currentRangeTableIdx++;
                continue;
            }
        } else if (rangeOffsets.empty()) {
            ValueVector* lowerBoundVector = nullptr;
            uint64_t lowerPos = 0;
            if (indexEvaluator != nullptr) {
                indexEvaluator->evaluate();
                lowerBoundVector = indexEvaluator->resultVector.get();
                auto& lowerSelVector = lowerBoundVector->state->getSelVector();
                DASSERT(lowerSelVector.getSelSize() == 1);
                lowerPos = lowerSelVector.getSelectedPositions()[0];
                if (lowerBoundVector->isNull(lowerPos)) {
                    currentRangeTableIdx++;
                    continue;
                }
            }

            ValueVector* upperBoundVector = nullptr;
            uint64_t upperPos = 0;
            if (upperBoundEvaluator != nullptr) {
                upperBoundEvaluator->evaluate();
                upperBoundVector = upperBoundEvaluator->resultVector.get();
                auto& upperSelVector = upperBoundVector->state->getSelVector();
                DASSERT(upperSelVector.getSelSize() == 1);
                upperPos = upperSelVector.getSelectedPositions()[0];
                if (upperBoundVector->isNull(upperPos)) {
                    currentRangeTableIdx++;
                    continue;
                }
            }
            static constexpr auto maxRangeResults = std::numeric_limits<common::idx_t>::max();
            const auto found =
                indexName.empty() ?
                    table.lookupPKRange(transaction, lowerBoundVector, lowerPos, lowerInclusive,
                        upperBoundVector, upperPos, upperInclusive, maxRangeResults, rangeOffsets) :
                isIndexEquality ? table.lookupIndex(transaction, indexName, lowerBoundVector,
                                      lowerPos, rangeOffsets) :
                                  table.lookupIndexRange(transaction, indexName, lowerBoundVector,
                                      lowerPos, lowerInclusive, upperBoundVector, upperPos,
                                      upperInclusive, maxRangeResults, rangeOffsets);
            if (!found) {
                currentRangeTableIdx++;
                continue;
            }
        }

        const auto remaining = rangeOffsets.size() - rangeOffsetCursor;
        if (remaining == 0) {
            rangeOffsets.clear();
            rangeOffsetCursor = 0;
            currentRangeTableIdx++;
            continue;
        }
        const auto outputSize = std::min<uint64_t>(remaining, common::DEFAULT_VECTOR_CAPACITY);
        auto& selVector = scanState->nodeIDVector->state->getSelVectorUnsafe();
        selVector.setToUnfiltered(outputSize);
        for (auto i = 0u; i < outputSize; ++i) {
            const auto nodeOffset = rangeOffsets[rangeOffsetCursor + i];
            scanState->nodeIDVector->setValue<nodeID_t>(i, {nodeOffset, table.getTableID()});
        }
        rangeOffsetCursor += outputSize;
        tableInfo.initScanState(*scanState, outVectors, context->clientContext);
        table.lookupMultiple(transaction, *scanState);
        tableInfo.castColumns();
        scanState->outState->setToUnflat();
        metrics->numOutputTuple.increase(outputSize);
        return true;
    }
    return false;
}

void PrimaryKeyScanNodeTable::lookupKeyList(const transaction::Transaction* transaction,
    const NodeTable& table) {
    indexEvaluator->evaluate();
    auto listVector = indexEvaluator->resultVector.get();
    auto& selVector = listVector->state->getSelVector();
    DASSERT(selVector.getSelSize() == 1);
    auto pos = selVector.getSelectedPositions()[0];
    if (listVector->isNull(pos)) {
        return;
    }
    auto& list = listVector->getValue<list_entry_t>(pos);
    auto keyVector = ListVector::getDataVector(listVector);
    std::vector<uint64_t> keyPositions;
    for (auto i = list.offset; i < list.offset + list.size; ++i) {
        if (keyVector->isNull(i)) {
            continue;
        }
        if (indexName.empty()) {
            offset_t offset = 0;
            if (table.lookupPK(transaction, keyVector, i, offset)) {
                rangeOffsets.push_back(offset);
            }
        } else {
            keyPositions.push_back(i);
        }
    }
    if (!keyPositions.empty()) {
        table.lookupIndex(transaction, indexName, keyVector, keyPositions, rangeOffsets);
    }
    // Repeated keys must not repeat rows; sorted offsets also read the table in storage order.
    std::sort(rangeOffsets.begin(), rangeOffsets.end());
    rangeOffsets.erase(std::unique(rangeOffsets.begin(), rangeOffsets.end()), rangeOffsets.end());
}

} // namespace processor
} // namespace lbug
