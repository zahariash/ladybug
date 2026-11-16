#pragma once

#include "processor/operator/filtering_operator.h"
#include "processor/operator/physical_operator.h"

namespace lbug {
namespace processor {

struct FlattenLocalState {
    uint64_t currentIdx = 0;
    uint64_t sizeToFlatten = 0;
    // Counts the batches pulled from the child.
    uint64_t batchSeq = 0;
};

// The batch a Flatten walks, for an operator above it that looks ahead (HashJoinProbe's prefetch).
// The batch's positions are selVector[0, size); the last one handed out is selVector[nextIdx - 1].
struct FlattenBatchView {
    const common::SelectionVector* selVector;
    uint64_t size;
    uint64_t nextIdx;
    uint64_t seq;
};

class Flatten final : public PhysicalOperator, SelVectorOverWriter {
    static constexpr PhysicalOperatorType type_ = PhysicalOperatorType::FLATTEN;

public:
    Flatten(data_chunk_pos_t dataChunkToFlattenPos, std::unique_ptr<PhysicalOperator> child,
        uint32_t id, std::unique_ptr<OPPrintInfo> printInfo)
        : PhysicalOperator{type_, std::move(child), id, std::move(printInfo)},
          dataChunkToFlattenPos{dataChunkToFlattenPos}, dataChunkState{nullptr} {}

    void initLocalStateInternal(ResultSet* resultSet, ExecutionContext* context) override;

    bool getNextTuplesInternal(ExecutionContext* context) override;

    std::unique_ptr<PhysicalOperator> copy() override {
        return make_unique<Flatten>(dataChunkToFlattenPos, children[0]->copy(), id,
            printInfo->copy());
    }

    data_chunk_pos_t getDataChunkToFlattenPos() const { return dataChunkToFlattenPos; }
    FlattenBatchView getBatchView() const {
        return {prevSelVector.get(), localState->sizeToFlatten, localState->currentIdx,
            localState->batchSeq};
    }

private:
    void resetCurrentSelVector(const common::SelectionVector& selVector) override;

private:
    data_chunk_pos_t dataChunkToFlattenPos;
    common::DataChunkState* dataChunkState;
    std::unique_ptr<FlattenLocalState> localState;
};

} // namespace processor
} // namespace lbug
