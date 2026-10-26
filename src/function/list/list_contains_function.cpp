#include <algorithm>
#include <string_view>

#include "binder/expression/expression_util.h"
#include "common/exception/binder.h"
#include "common/type_utils.h"
#include "function/list/functions/list_position_function.h"
#include "function/list/vector_list_functions.h"
#include "function/scalar_function.h"
#include <format>

using namespace lbug::common;
using namespace lbug::binder;

namespace lbug {
namespace function {

// `x IN list` is parsed into list_contains, so this also defines IN over a list: a NULL element
// never matches, and a list without a match is false even if it holds NULLs, not NULL as SQL's
// three-valued IN would give. DuckDB's list_contains behaves the same way.
struct ListContains {
    template<typename T>
    static void operation(common::list_entry_t& list, T& element, uint8_t& result,
        common::ValueVector& listVector, common::ValueVector& elementVector,
        common::ValueVector& resultVector) {
        int64_t pos = 0;
        ListPosition::operation(list, element, pos, listVector, elementVector, resultVector);
        result = (pos != 0);
    }
};

// The members of a list, sorted for binary search, so a build allocates one buffer per vector
// rather than a node per member.
struct ListContainsBindData final : FunctionBindData {
    std::vector<uint64_t> integerKeys;
    std::vector<std::string_view> stringKeys;
    // The bytes of the string members, which stringKeys points into.
    std::string stringData;
    bool isBuilt = false;

    ListContainsBindData(std::vector<LogicalType> paramTypes, LogicalType resultType)
        : FunctionBindData{std::move(paramTypes), std::move(resultType)} {}

    // A copy starts without keys, so no two copies share views into one stringData.
    std::unique_ptr<FunctionBindData> copy() const override {
        return std::make_unique<ListContainsBindData>(LogicalType::copy(paramTypes),
            resultType.copy());
    }
};

template<typename T>
concept KeyedListElement =
    std::is_same_v<T, string_t> || (std::is_integral_v<T> && !std::is_same_v<T, bool>);

// The linear membership test, as a select operation.
struct ListContainsScan {
    template<typename T>
    static void operation(list_entry_t& list, T& element, uint8_t& result, ValueVector& listVector,
        ValueVector& elementVector, ValueVector& resultVector, void* /*dataPtr*/) {
        ListContains::operation(list, element, result, listVector, elementVector, resultVector);
    }
};

// list_contains over a literal or parameter list, the same for every row of an execution:
// binary-search the list's sorted members instead of scanning the list per row. Each execution
// copies the plan's operators and their bind data, so a cached plan run with another parameter
// list builds its keys again.
struct ListContainsConstantList {
    // The keys hold one list, so a list that is not flat, or whose element type is not the
    // element's, is scanned instead.
    static bool canUseKeys(const std::vector<std::shared_ptr<ValueVector>>& params, void* dataPtr) {
        return dataPtr != nullptr && params[0]->state->isFlat() &&
               ListType::getChildType(params[0]->dataType) == params[1]->dataType;
    }

    template<KeyedListElement T>
    static void operation(list_entry_t& list, T& element, uint8_t& result, ValueVector& listVector,
        ValueVector& /*elementVector*/, ValueVector& /*resultVector*/, void* dataPtr) {
        auto& bindData = *static_cast<ListContainsBindData*>(dataPtr);
        if (!bindData.isBuilt) {
            build<T>(bindData, list, listVector);
        }
        if constexpr (std::is_same_v<T, string_t>) {
            result = std::binary_search(bindData.stringKeys.begin(), bindData.stringKeys.end(),
                element.getAsStringView());
        } else {
            result = std::binary_search(bindData.integerKeys.begin(), bindData.integerKeys.end(),
                static_cast<uint64_t>(element));
        }
    }

    template<KeyedListElement T>
    static void build(ListContainsBindData& bindData, const list_entry_t& list,
        ValueVector& listVector) {
        auto dataVector = ListVector::getDataVector(&listVector);
        if constexpr (std::is_same_v<T, string_t>) {
            bindData.stringKeys.clear();
            bindData.stringData.clear();
            bindData.stringKeys.reserve(list.size);
            uint64_t numBytes = 0;
            for (auto i = list.offset; i < list.offset + list.size; ++i) {
                if (!dataVector->isNull(i)) {
                    numBytes += dataVector->getValue<string_t>(i).len;
                }
            }
            // Reserved up front, so appending never moves the bytes the views point into.
            bindData.stringData.reserve(numBytes);
            for (auto i = list.offset; i < list.offset + list.size; ++i) {
                if (dataVector->isNull(i)) {
                    continue;
                }
                const auto value = dataVector->getValue<string_t>(i).getAsStringView();
                const auto start = bindData.stringData.size();
                bindData.stringData.append(value);
                bindData.stringKeys.emplace_back(bindData.stringData.data() + start, value.size());
            }
            std::sort(bindData.stringKeys.begin(), bindData.stringKeys.end());
        } else {
            bindData.integerKeys.clear();
            bindData.integerKeys.reserve(list.size);
            for (auto i = list.offset; i < list.offset + list.size; ++i) {
                if (!dataVector->isNull(i)) {
                    bindData.integerKeys.push_back(
                        static_cast<uint64_t>(dataVector->getValue<T>(i)));
                }
            }
            std::sort(bindData.integerKeys.begin(), bindData.integerKeys.end());
        }
        bindData.isBuilt = true;
    }

    template<KeyedListElement T>
    static void exec(const std::vector<std::shared_ptr<ValueVector>>& params,
        const std::vector<SelectionVector*>& paramSelVectors, ValueVector& result,
        SelectionVector* resultSelVector, void* dataPtr) {
        if (canUseKeys(params, dataPtr)) {
            ScalarFunction::BinaryExecWithBindData<list_entry_t, T, uint8_t,
                ListContainsConstantList>(params, paramSelVectors, result, resultSelVector,
                dataPtr);
            return;
        }
        ScalarFunction::BinaryExecListStructFunction<list_entry_t, T, uint8_t, ListContains>(params,
            paramSelVectors, result, resultSelVector, dataPtr);
    }

    template<KeyedListElement T>
    static bool select(const std::vector<std::shared_ptr<ValueVector>>& params,
        SelectionVector& selVector, void* dataPtr) {
        if (canUseKeys(params, dataPtr)) {
            return ScalarFunction::BinarySelectWithBindData<list_entry_t, T,
                ListContainsConstantList>(params, selVector, dataPtr);
        }
        return ScalarFunction::BinarySelectWithBindData<list_entry_t, T, ListContainsScan>(params,
            selVector, dataPtr);
    }
};

static bool isConstantList(const Expression& listExpr) {
    return listExpr.expressionType == ExpressionType::LITERAL ||
           listExpr.expressionType == ExpressionType::PARAMETER;
}

static std::unique_ptr<FunctionBindData> bindFunc(const ScalarBindFuncInput& input) {
    auto scalarFunction = input.definition->ptrCast<ScalarFunction>();
    // for list_contains(list, input), we expect input and list child have the same type, if list
    // is empty, we use in the input type. Otherwise, we use list child type because casting list
    // is more expensive.
    std::vector<LogicalType> paramTypes;
    LogicalType childType;
    auto listExpr = input.arguments[0];
    auto elementExpr = input.arguments[1];
    // A NULL literal (or unbound parameter) list has type ANY and no child type.
    if (ExpressionUtil::isEmptyList(*listExpr) ||
        listExpr->getDataType().getLogicalTypeID() == LogicalTypeID::ANY) {
        childType = elementExpr->getDataType().copy();
    } else {
        auto& listChildType = ListType::getChildType(listExpr->getDataType());
        auto& elementType = elementExpr->getDataType();
        if (!LogicalTypeUtils::tryGetMaxLogicalType(listChildType, elementType, childType)) {
            throw BinderException(std::format("Cannot compare {} and {} in list_contains function.",
                listChildType.toString(), elementType.toString()));
        }
    }
    if (childType.getLogicalTypeID() == LogicalTypeID::ANY) {
        childType = LogicalType::STRING();
    }
    auto listType = LogicalType::LIST(childType.copy());
    paramTypes.push_back(listType.copy());
    paramTypes.push_back(childType.copy());
    auto useKeys = false;
    TypeUtils::visit(childType.getPhysicalType(), [&]<typename T>(T) {
        scalarFunction->selectFunc = nullptr;
        if constexpr (KeyedListElement<T>) {
            if (isConstantList(*listExpr)) {
                useKeys = true;
                scalarFunction->execFunc = ListContainsConstantList::exec<T>;
                scalarFunction->selectFunc = ListContainsConstantList::select<T>;
                return;
            }
        }
        scalarFunction->execFunc =
            ScalarFunction::BinaryExecListStructFunction<list_entry_t, T, uint8_t, ListContains>;
    });
    if (useKeys) {
        return std::make_unique<ListContainsBindData>(std::move(paramTypes), LogicalType::BOOL());
    }
    return std::make_unique<FunctionBindData>(std::move(paramTypes), LogicalType::BOOL());
}

function_set ListContainsFunction::getFunctionSet() {
    function_set result;
    auto function = std::make_unique<ScalarFunction>(name,
        std::vector<LogicalTypeID>{LogicalTypeID::LIST, LogicalTypeID::ANY}, LogicalTypeID::BOOL);
    function->bindFunc = bindFunc;
    result.push_back(std::move(function));
    return result;
}

} // namespace function
} // namespace lbug
