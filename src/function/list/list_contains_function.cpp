#include <optional>
#include <string_view>
#include <unordered_set>

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

struct StringViewHash {
    using is_transparent = void;
    size_t operator()(std::string_view value) const { return std::hash<std::string_view>{}(value); }
};

// Holds the members of a list as a hash set. A constant list's set is built on first use; for
// other lists the hashed list is kept too, so the set is rebuilt only when the list changes.
struct ListContainsBindData final : FunctionBindData {
    std::unordered_set<uint64_t> integerKeys;
    std::unordered_set<std::string, StringViewHash, std::equal_to<>> stringKeys;
    std::vector<std::optional<uint64_t>> hashedIntegerList;
    std::vector<std::optional<std::string>> hashedStringList;
    bool isBuilt = false;

    template<typename T>
    const auto& hashedList() const {
        if constexpr (std::is_same_v<T, string_t>) {
            return hashedStringList;
        } else {
            return hashedIntegerList;
        }
    }

    ListContainsBindData(std::vector<LogicalType> paramTypes, LogicalType resultType)
        : FunctionBindData{std::move(paramTypes), std::move(resultType)} {}

    std::unique_ptr<FunctionBindData> copy() const override {
        return std::make_unique<ListContainsBindData>(LogicalType::copy(paramTypes),
            resultType.copy());
    }
};

template<typename T>
concept HashableListElement =
    std::is_same_v<T, string_t> || (std::is_integral_v<T> && !std::is_same_v<T, bool>);

// list_contains over a list that is the same for every row: probe a hash set instead of scanning
// the list per row. Short lists keep the linear scan.
struct ListContainsConstantList {
    static constexpr uint64_t MIN_LIST_SIZE_TO_HASH = 16;

    template<HashableListElement T>
    static void operation(list_entry_t& list, T& element, uint8_t& result, ValueVector& listVector,
        ValueVector& elementVector, ValueVector& resultVector, void* dataPtr) {
        if (list.size < MIN_LIST_SIZE_TO_HASH || dataPtr == nullptr) {
            ListContains::operation(list, element, result, listVector, elementVector, resultVector);
            return;
        }
        auto& bindData = *static_cast<ListContainsBindData*>(dataPtr);
        if (!bindData.isBuilt) {
            build<T>(bindData, list, listVector);
        }
        if constexpr (std::is_same_v<T, string_t>) {
            result = bindData.stringKeys.contains(element.getAsStringView());
        } else {
            result = bindData.integerKeys.contains(static_cast<uint64_t>(element));
        }
    }

    template<HashableListElement T>
    static void build(ListContainsBindData& bindData, const list_entry_t& list,
        ValueVector& listVector) {
        auto dataVector = ListVector::getDataVector(&listVector);
        for (auto i = list.offset; i < list.offset + list.size; ++i) {
            if (dataVector->isNull(i)) {
                continue;
            }
            if constexpr (std::is_same_v<T, string_t>) {
                bindData.stringKeys.emplace(dataVector->getValue<string_t>(i).getAsStringView());
            } else {
                bindData.integerKeys.insert(static_cast<uint64_t>(dataVector->getValue<T>(i)));
            }
        }
        bindData.isBuilt = true;
    }
};

// list_contains over a list that can differ between calls. When one list applies to many rows of
// a call (a flat list against enough unflat elements to pay for a rebuild), the constant-list path
// is used with a set refreshed for that list.
struct ListContainsFlatList {
    static bool canHash(const std::vector<std::shared_ptr<ValueVector>>& params, void* dataPtr) {
        const auto& elementState = *params[1]->state;
        return dataPtr != nullptr && params[0]->state->isFlat() && !elementState.isFlat() &&
               elementState.getSelVector().getSelSize() >=
                   ListContainsConstantList::MIN_LIST_SIZE_TO_HASH;
    }

    // Linear membership test, for lists that differ per row.
    template<typename T>
    static void operation(list_entry_t& list, T& element, uint8_t& result, ValueVector& listVector,
        ValueVector& elementVector, ValueVector& resultVector, void* /*dataPtr*/) {
        ListContains::operation(list, element, result, listVector, elementVector, resultVector);
    }

    template<HashableListElement T>
    static void refresh(ListContainsBindData& bindData, ValueVector& listVector,
        const SelectionVector& listSelVector) {
        const auto pos = listSelVector[0];
        if (listVector.isNull(pos)) {
            return;
        }
        auto& list = listVector.getValue<list_entry_t>(pos);
        if (list.size < ListContainsConstantList::MIN_LIST_SIZE_TO_HASH) {
            return;
        }
        auto dataVector = ListVector::getDataVector(&listVector);
        if (isHashed<T>(bindData, list, *dataVector)) {
            return;
        }
        bindData.integerKeys.clear();
        bindData.stringKeys.clear();
        bindData.hashedIntegerList.clear();
        bindData.hashedStringList.clear();
        for (auto i = list.offset; i < list.offset + list.size; ++i) {
            const auto isNull = dataVector->isNull(i);
            if constexpr (std::is_same_v<T, string_t>) {
                bindData.hashedStringList.push_back(
                    isNull ? std::nullopt :
                             std::optional{dataVector->getValue<string_t>(i).getAsString()});
            } else {
                bindData.hashedIntegerList.push_back(
                    isNull ? std::nullopt :
                             std::optional{static_cast<uint64_t>(dataVector->getValue<T>(i))});
            }
        }
        ListContainsConstantList::build<T>(bindData, list, listVector);
    }

    template<HashableListElement T>
    static bool isHashed(const ListContainsBindData& bindData, const list_entry_t& list,
        ValueVector& dataVector) {
        const auto& hashed = bindData.hashedList<T>();
        if (hashed.size() != list.size) {
            return false;
        }
        for (auto i = 0u; i < list.size; ++i) {
            const auto pos = list.offset + i;
            if (dataVector.isNull(pos)) {
                if (hashed[i].has_value()) {
                    return false;
                }
                continue;
            }
            if constexpr (std::is_same_v<T, string_t>) {
                if (hashed[i] != dataVector.getValue<string_t>(pos).getAsStringView()) {
                    return false;
                }
            } else {
                if (hashed[i] != static_cast<uint64_t>(dataVector.getValue<T>(pos))) {
                    return false;
                }
            }
        }
        return true;
    }

    template<HashableListElement T>
    static void exec(const std::vector<std::shared_ptr<ValueVector>>& params,
        const std::vector<SelectionVector*>& paramSelVectors, ValueVector& result,
        SelectionVector* resultSelVector, void* dataPtr) {
        if (canHash(params, dataPtr)) {
            refresh<T>(*static_cast<ListContainsBindData*>(dataPtr), *params[0],
                *paramSelVectors[0]);
            ScalarFunction::BinaryExecWithBindData<list_entry_t, T, uint8_t,
                ListContainsConstantList>(params, paramSelVectors, result, resultSelVector,
                dataPtr);
            return;
        }
        ScalarFunction::BinaryExecListStructFunction<list_entry_t, T, uint8_t, ListContains>(params,
            paramSelVectors, result, resultSelVector, dataPtr);
    }

    template<HashableListElement T>
    static bool select(const std::vector<std::shared_ptr<ValueVector>>& params,
        SelectionVector& selVector, void* dataPtr) {
        if (canHash(params, dataPtr)) {
            refresh<T>(*static_cast<ListContainsBindData*>(dataPtr), *params[0],
                params[0]->state->getSelVector());
            return ScalarFunction::BinarySelectWithBindData<list_entry_t, T,
                ListContainsConstantList>(params, selVector, dataPtr);
        }
        return ScalarFunction::BinarySelectWithBindData<list_entry_t, T, ListContainsFlatList>(
            params, selVector, dataPtr);
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
    auto hashList = false;
    TypeUtils::visit(childType.getPhysicalType(), [&]<typename T>(T) {
        scalarFunction->selectFunc = nullptr;
        if constexpr (HashableListElement<T>) {
            hashList = true;
            if (isConstantList(*listExpr)) {
                scalarFunction->execFunc = ScalarFunction::BinaryExecWithBindData<list_entry_t, T,
                    uint8_t, ListContainsConstantList>;
                scalarFunction->selectFunc = ScalarFunction::BinarySelectWithBindData<list_entry_t,
                    T, ListContainsConstantList>;
            } else {
                scalarFunction->execFunc = ListContainsFlatList::exec<T>;
                scalarFunction->selectFunc = ListContainsFlatList::select<T>;
            }
            return;
        }
        scalarFunction->execFunc =
            ScalarFunction::BinaryExecListStructFunction<list_entry_t, T, uint8_t, ListContains>;
    });
    if (hashList) {
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
