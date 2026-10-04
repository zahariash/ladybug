#include "optimizer/filter_push_down_optimizer.h"

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <unordered_set>

#include "binder/expression/literal_expression.h"
#include "binder/expression/parameter_expression.h"
#include "binder/expression/property_expression.h"
#include "binder/expression/scalar_function_expression.h"
#include "catalog/catalog.h"
#include "catalog/catalog_entry/index_catalog_entry.h"
#include "catalog/catalog_entry/table_catalog_entry.h"
#include "common/string_utils.h"
#include "function/list/vector_list_functions.h"
#include "main/attached_database.h"
#include "main/client_context.h"
#include "main/database_manager.h"
#include "planner/join_order/cardinality_estimator.h"
#include "planner/operator/extend/logical_extend.h"
#include "planner/operator/logical_empty_result.h"
#include "planner/operator/logical_filter.h"
#include "planner/operator/logical_hash_join.h"
#include "planner/operator/logical_table_function_call.h"
#include "planner/operator/scan/logical_scan_node_table.h"
#include "storage/index/art_index.h"
#include "storage/local_storage/local_storage.h"
#include "storage/local_storage/local_table.h"
#include "storage/storage_manager.h"
#include "storage/table/columnar_node_table_base.h"
#include "storage/table/node_table.h"

using namespace lbug::binder;
using namespace lbug::common;
using namespace lbug::planner;
using namespace lbug::storage;
using namespace lbug::catalog;

namespace lbug {
namespace optimizer {

void FilterPushDownOptimizer::rewrite(LogicalPlan* plan) {
    visitOperator(plan->getLastOperator());
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::visitOperator(
    const std::shared_ptr<LogicalOperator>& op) {
    switch (op->getOperatorType()) {
    case LogicalOperatorType::FILTER: {
        return visitFilterReplace(op);
    }
    case LogicalOperatorType::CROSS_PRODUCT: {
        return visitCrossProductReplace(op);
    }
    case LogicalOperatorType::EXTEND: {
        return visitExtendReplace(op);
    }
    case LogicalOperatorType::SCAN_NODE_TABLE: {
        return visitScanNodeTableReplace(op);
    }
    case LogicalOperatorType::TABLE_FUNCTION_CALL: {
        return visitTableFunctionCallReplace(op);
    }
    default: { // Stop current push down for unhandled operator.
        return visitChildren(op);
    }
    }
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::visitChildren(
    const std::shared_ptr<LogicalOperator>& op) {
    for (auto i = 0u; i < op->getNumChildren(); ++i) {
        // Start new push down for child.
        auto optimizer = FilterPushDownOptimizer(context, cardinalityEstimator);
        op->setChild(i, optimizer.visitOperator(op->getChild(i)));
    }
    op->computeFlatSchema();
    return finishPushDown(op);
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::visitFilterReplace(
    const std::shared_ptr<LogicalOperator>& op) {
    auto& filter = op->constCast<LogicalFilter>();
    auto predicate = filter.getPredicate();
    if (predicate->expressionType == ExpressionType::LITERAL) {
        // Avoid executing child plan if literal is Null or False.
        auto& literalExpr = predicate->constCast<LiteralExpression>();
        if (literalExpr.isNull() || !literalExpr.getValue().getValue<bool>()) {
            return std::make_shared<LogicalEmptyResult>(*op->getSchema());
        }
        // Ignore if literal is True.
    } else {
        predicateSet.addPredicate(predicate);
    }
    return visitOperator(filter.getChild(0));
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::visitCrossProductReplace(
    const std::shared_ptr<LogicalOperator>& op) {
    auto remainingPSet = PredicateSet();
    auto probePSet = PredicateSet();
    auto buildPSet = PredicateSet();
    for (auto& p : predicateSet.getAllPredicates()) {
        auto inProbe = op->getChild(0)->getSchema()->evaluable(*p);
        auto inBuild = op->getChild(1)->getSchema()->evaluable(*p);
        if (inProbe && !inBuild) {
            probePSet.addPredicate(p);
        } else if (!inProbe && inBuild) {
            buildPSet.addPredicate(p);
        } else {
            remainingPSet.addPredicate(p);
        }
    }
    DASSERT(op->getNumChildren() == 2);
    // Push probe side
    auto probeOptimizer =
        FilterPushDownOptimizer(context, cardinalityEstimator, std::move(probePSet));
    op->setChild(0, probeOptimizer.visitOperator(op->getChild(0)));
    // Push build side
    auto buildOptimizer =
        FilterPushDownOptimizer(context, cardinalityEstimator, std::move(buildPSet));
    op->setChild(1, buildOptimizer.visitOperator(op->getChild(1)));

    auto probeSchema = op->getChild(0)->getSchema();
    auto buildSchema = op->getChild(1)->getSchema();
    expression_vector predicates;
    std::vector<join_condition_t> joinConditions;
    for (auto& predicate : remainingPSet.equalityPredicates) {
        auto left = predicate->getChild(0);
        auto right = predicate->getChild(1);
        // TODO(Xiyang): this can only rewrite left = right, we should also be able to do
        // expr(left), expr(right)
        if (probeSchema->isExpressionInScope(*left) && buildSchema->isExpressionInScope(*right)) {
            joinConditions.emplace_back(left, right);
        } else if (probeSchema->isExpressionInScope(*right) &&
                   buildSchema->isExpressionInScope(*left)) {
            joinConditions.emplace_back(right, left);
        } else {
            // Collect predicates that cannot be rewritten as join conditions.
            predicates.push_back(predicate);
        }
    }
    if (joinConditions.empty()) { // Nothing to push down. Terminate.
        return finishPushDown(op);
    }
    auto hashJoin = std::make_shared<LogicalHashJoin>(joinConditions, JoinType::INNER,
        nullptr /* mark */, op->getChild(0), op->getChild(1), 0 /* cardinality */);
    // For non-id based joins, we disable side way information passing.
    hashJoin->getSIPInfoUnsafe().position = SemiMaskPosition::PROHIBIT;
    hashJoin->computeFlatSchema();
    if (cardinalityEstimator != nullptr) {
        hashJoin->setCardinality(cardinalityEstimator->estimateHashJoin(joinConditions,
            *op->getChild(0), *op->getChild(1)));
    }
    // Apply remaining predicates.
    predicates.insert(predicates.end(), remainingPSet.nonEqualityPredicates.begin(),
        remainingPSet.nonEqualityPredicates.end());
    if (predicates.empty()) {
        return hashJoin;
    }
    return appendFiltersStatsAware(std::move(predicates), hashJoin);
}

static ColumnPredicateSet getPredicateSet(const Expression& column,
    const binder::expression_vector& predicates) {
    auto predicateSet = ColumnPredicateSet();
    for (auto& predicate : predicates) {
        auto columnPredicate = ColumnPredicateUtil::tryConvert(column, *predicate);
        if (columnPredicate == nullptr) {
            continue;
        }
        predicateSet.addPredicate(std::move(columnPredicate));
    }
    return predicateSet;
}

static std::vector<ColumnPredicateSet> getColumnPredicateSets(const expression_vector& columns,
    const expression_vector& predicates) {
    std::vector<ColumnPredicateSet> predicateSets;
    for (auto& column : columns) {
        predicateSets.push_back(getPredicateSet(*column, predicates));
    }
    return predicateSets;
}

static bool isConstantExpression(const std::shared_ptr<Expression> expression) {
    switch (expression->expressionType) {
    case ExpressionType::LITERAL:
    case ExpressionType::PARAMETER: {
        return true;
    }
    // TODO(Xiyang): fold parameter expression in binder.
    case ExpressionType::FUNCTION: {
        auto& func = expression->constCast<ScalarFunctionExpression>();
        if (func.getFunction().name.starts_with("CAST")) {
            return isConstantExpression(func.getChild(0));
        } else {
            return false;
        }
    }
    default:
        return false;
    }
}

static bool isNodeProperty(const Expression& expression, const Expression& nodeID) {
    if (expression.expressionType != ExpressionType::PROPERTY) {
        return false;
    }
    auto& property = expression.constCast<PropertyExpression>();
    return property.getVariableName() == nodeID.constCast<PropertyExpression>().getVariableName();
}

// Returns the name of a secondary ART index on the property, or an empty string if none exists.
static std::string getSecondaryARTIndexName(const PropertyExpression& property, table_id_t tableID,
    main::ClientContext* context, const std::string& dbName) {
    if (property.isPrimaryKey(tableID) || !property.hasProperty(tableID)) {
        return {};
    }
    auto [cat, sm] = main::DatabaseManager::resolveTableStorage(*context, tableID, dbName);
    auto transaction = transaction::Transaction::Get(*context);
    auto tableEntry = cat->getTableCatalogEntry(transaction, tableID);
    // Bind-time property presence (hasProperty) can disagree with the catalog entry
    // resolved here (e.g. ANY-graph tables share IDs with main-catalog tables and
    // resolveTableStorage with an empty dbName prefers the main catalog). Never throw
    // from the optimizer on that mismatch; skip the rewrite and keep scan + filter.
    if (!tableEntry->containsProperty(property.getPropertyName())) {
        return {};
    }
    const auto propertyID = tableEntry->getPropertyID(property.getPropertyName());
    auto* table = sm->getTable(tableID)->ptrCast<NodeTable>();
    for (auto* indexEntry : cat->getIndexEntries(transaction, tableID)) {
        if (!indexEntry->containsPropertyID(propertyID) ||
            !StringUtils::caseInsensitiveEquals(indexEntry->getIndexType(),
                ArtPrimaryKeyIndex::getIndexType().typeName)) {
            continue;
        }
        auto index = table->getIndex(indexEntry->getIndexName());
        if (!index.has_value() || index.value()->isPrimary()) {
            continue;
        }
        return indexEntry->getIndexName();
    }
    return {};
}

static const TableStats* getStorageStats(const CardinalityEstimator* cardinalityEstimator,
    table_id_t tableID) {
    if (cardinalityEstimator == nullptr) {
        return nullptr;
    }
    auto* stats = cardinalityEstimator->getTableStats(tableID);
    return stats != nullptr && stats->storageStats.has_value() ? &*stats->storageStats : nullptr;
}

double FilterPushDownOptimizer::estimateScanCost(double numRows) const {
    const auto numThreads = std::max<uint64_t>(1, context->getClientConfig()->numThreads);
    return PlannerKnobs::SCAN_STARTUP_COST + numRows * PlannerKnobs::SCAN_ROW_COST / numThreads;
}

bool FilterPushDownOptimizer::isPrimaryKeyLookupCheaper(table_id_t tableID, uint64_t numKeys,
    bool isHashIndex) const {
    // Without statistics, keep preferring the index.
    auto* stats = getStorageStats(cardinalityEstimator, tableID);
    if (stats == nullptr) {
        return true;
    }
    const auto keyCost = isHashIndex ? PlannerKnobs::HASH_INDEX_KEY_LOOKUP_COST :
                                       PlannerKnobs::ART_INDEX_KEY_LOOKUP_COST;
    const auto indexCost = numKeys * (keyCost + PlannerKnobs::INDEX_ROW_FETCH_COST);
    return indexCost < estimateScanCost(static_cast<double>(stats->getTableCard()));
}

bool FilterPushDownOptimizer::isSecondaryARTLookupCheaper(table_id_t tableID,
    const std::string& dbName, uint64_t numKeys, const PropertyExpression& property) const {
    // Without statistics, keep preferring the index.
    auto* stats = getStorageStats(cardinalityEstimator, tableID);
    if (stats == nullptr) {
        return true;
    }
    auto transaction = transaction::Transaction::Get(*context);
    auto [cat, sm] = main::DatabaseManager::resolveTableStorage(*context, tableID, dbName);
    auto entry = cat->getTableCatalogEntry(transaction, tableID);
    if (!entry->containsProperty(property.getPropertyName())) {
        return true;
    }
    const auto columnID = entry->getColumnID(property.getPropertyName());
    if (columnID == INVALID_COLUMN_ID || columnID == ROW_IDX_COLUMN_ID) {
        return true;
    }
    const auto numDistinct = stats->getNumDistinctValues(columnID);
    if (numDistinct == 0) {
        return true;
    }
    const auto numRows = static_cast<double>(stats->getTableCard());
    const auto rowsPerKey = std::max(1.0, numRows / static_cast<double>(numDistinct));
    const auto numMatchedRows = std::min(numRows, numKeys * rowsPerKey);
    // The ART index lacks this transaction's inserts, so each lookup also checks all of the
    // transaction's uncommitted rows.
    double numUncommittedRows = 0;
    if (auto* localStorage = transaction->getLocalStorage()) {
        if (auto* localTable = localStorage->getLocalTable(tableID)) {
            numUncommittedRows = static_cast<double>(localTable->getNumTotalRows());
        }
    }
    const auto indexCost = numKeys * PlannerKnobs::ART_INDEX_KEY_LOOKUP_COST +
                           numMatchedRows * PlannerKnobs::INDEX_ROW_FETCH_COST +
                           numUncommittedRows * PlannerKnobs::UNCOMMITTED_ROW_MATCH_COST;
    return indexCost < estimateScanCost(numRows);
}

// Number of keys in a constant key list: a literal, a parameter, or a cast of either.
static uint64_t getNumKeys(const Expression& keys) {
    switch (keys.expressionType) {
    case ExpressionType::LITERAL: {
        auto value = keys.constCast<LiteralExpression>().getValue();
        return value.getDataType().getLogicalTypeID() == LogicalTypeID::LIST ?
                   value.getChildrenSize() :
                   1;
    }
    case ExpressionType::PARAMETER: {
        auto& parameter = keys.constCast<ParameterExpression>();
        // The index-or-scan choice depends on the list size, so the plan can't be reused for
        // other values.
        parameter.markBakedIntoPlan();
        auto value = parameter.getValue();
        return value.getDataType().getLogicalTypeID() == LogicalTypeID::LIST ?
                   value.getChildrenSize() :
                   1;
    }
    case ExpressionType::FUNCTION:
        return getNumKeys(*keys.getChild(0));
    default:
        return 1;
    }
}

static std::optional<std::pair<std::shared_ptr<Expression>, std::string>>
popSecondaryARTEqualityComparison(PredicateSet& predicateSet, const Expression& nodeID,
    table_id_t tableID, main::ClientContext* context, const std::string& dbName,
    const std::function<bool(const PropertyExpression&)>& isIndexCheaper) {
    for (auto i = 0u; i < predicateSet.equalityPredicates.size(); ++i) {
        auto predicate = predicateSet.equalityPredicates[i];
        auto lhs = predicate->getChild(0);
        auto rhs = predicate->getChild(1);
        if (!isNodeProperty(*lhs, nodeID) && isNodeProperty(*rhs, nodeID)) {
            std::swap(lhs, rhs);
        }
        if (!isNodeProperty(*lhs, nodeID) || !isConstantExpression(rhs)) {
            continue;
        }
        auto& property = lhs->constCast<PropertyExpression>();
        auto indexName = getSecondaryARTIndexName(property, tableID, context, dbName);
        if (indexName.empty() || !isIndexCheaper(property)) {
            continue;
        }
        predicateSet.equalityPredicates.erase(predicateSet.equalityPredicates.begin() + i);
        return std::make_pair(rhs, indexName);
    }
    return std::nullopt;
}

static bool isListContains(const Expression& expression) {
    if (expression.expressionType != ExpressionType::FUNCTION) {
        return false;
    }
    auto& name = expression.constCast<ScalarFunctionExpression>().getFunction().name;
    return name == function::ListContainsFunction::name || name == function::ListHasFunction::name;
}

// Pops `n.property IN <constant list>` for which canUseIndex(property, list) holds and returns the
// list. The list must already have the property's type, so its elements are usable as index keys.
static std::shared_ptr<Expression> popInListComparison(PredicateSet& predicateSet,
    const Expression& nodeID,
    const std::function<bool(const PropertyExpression&, const Expression&)>& canUseIndex) {
    auto& predicates = predicateSet.nonEqualityPredicates;
    for (auto i = 0u; i < predicates.size(); ++i) {
        if (!isListContains(*predicates[i])) {
            continue;
        }
        auto list = predicates[i]->getChild(0);
        auto element = predicates[i]->getChild(1);
        if (!isNodeProperty(*element, nodeID) || !isConstantExpression(list) ||
            list->getDataType().getLogicalTypeID() != LogicalTypeID::LIST ||
            ListType::getChildType(list->getDataType()) != element->getDataType() ||
            !canUseIndex(element->constCast<PropertyExpression>(), *list)) {
            continue;
        }
        predicates.erase(predicates.begin() + i);
        return list;
    }
    return nullptr;
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::visitScanNodeTableReplace(
    const std::shared_ptr<LogicalOperator>& op) {
    auto& scan = op->cast<LogicalScanNodeTable>();
    auto nodeID = scan.getNodeID();
    // Apply column predicates.
    if (context->getClientConfig()->enableZoneMap) {
        scan.setPropertyPredicates(
            getColumnPredicateSets(scan.getProperties(), predicateSet.getAllPredicates()));
    }
    // Apply index scan
    auto tableIDs = scan.getTableIDs();
    std::shared_ptr<Expression> primaryKeyEqualityComparison = nullptr;
    auto& dbMap = scan.getTableDBMap();
    auto getResolvedTable = [&](table_id_t id) -> storage::NodeTable* {
        auto dbIt = dbMap.find(id);
        auto dbName = dbIt != dbMap.end() ? dbIt->second : std::string{};
        auto [cat, sm] = main::DatabaseManager::resolveTableStorage(*context, id, dbName);
        return sm->getTable(id)->ptrCast<storage::NodeTable>();
    };
    std::string dbName;
    if (tableIDs.size() == 1) {
        primaryKeyEqualityComparison = predicateSet.popNodePKEqualityComparison(*nodeID);
        auto dbIt = dbMap.find(tableIDs[0]);
        dbName = dbIt != dbMap.end() ? dbIt->second : std::string{};
    }
    if (primaryKeyEqualityComparison != nullptr) { // Try rewrite index scan
        auto* table = getResolvedTable(tableIDs[0]);
        // Columnar-backed tables (icebug-disk, Arrow) have no usable primary key index;
        // keep the scan + filter instead of rewriting to an index lookup that misses.
        auto rhs = primaryKeyEqualityComparison->getChild(1);
        if (dynamic_cast<storage::ColumnarNodeTableBase*>(table) != nullptr) {
            predicateSet.addPredicate(primaryKeyEqualityComparison);
        } else if (table->tryGetPrimaryKeyIndex() != nullptr && isConstantExpression(rhs)) {
            auto extraInfo = std::make_unique<PrimaryKeyScanInfo>(rhs);
            scan.setScanType(LogicalScanNodeTableType::PRIMARY_KEY_SCAN);
            scan.setExtraInfo(std::move(extraInfo));
            scan.computeFlatSchema();
        } else {
            // Cannot rewrite and add predicate back.
            predicateSet.addPredicate(primaryKeyEqualityComparison);
        }
    } else if (tableIDs.size() == 1) {
        auto* table = getResolvedTable(tableIDs[0]);
        auto* pkIndex = table->tryGetPrimaryKeyIndex();
        std::shared_ptr<Expression> primaryKeyList = nullptr;
        if (pkIndex != nullptr && dynamic_cast<storage::ColumnarNodeTableBase*>(table) == nullptr) {
            const auto isHashIndex = pkIndex->getIndexInfo().indexType !=
                                     storage::ArtPrimaryKeyIndex::getIndexType().typeName;
            primaryKeyList = popInListComparison(predicateSet, *nodeID,
                [&](const PropertyExpression& property, const Expression& list) {
                    return property.isPrimaryKey(tableIDs[0]) &&
                           isPrimaryKeyLookupCheaper(tableIDs[0], getNumKeys(list), isHashIndex);
                });
        }
        if (primaryKeyList != nullptr) {
            scan.setScanType(LogicalScanNodeTableType::PRIMARY_KEY_SCAN);
            scan.setExtraInfo(PrimaryKeyScanInfo::createKeyList(std::move(primaryKeyList)));
            scan.computeFlatSchema();
        } else if (pkIndex != nullptr && pkIndex->getIndexInfo().indexType ==
                                             storage::ArtPrimaryKeyIndex::getIndexType().typeName) {
            auto primaryKeyRangeComparison = predicateSet.popNodePKRangeComparison(*nodeID);
            if (primaryKeyRangeComparison.hasBound()) {
                auto extraInfo = std::make_unique<PrimaryKeyScanInfo>(
                    primaryKeyRangeComparison.lowerBound, primaryKeyRangeComparison.lowerInclusive,
                    primaryKeyRangeComparison.upperBound, primaryKeyRangeComparison.upperInclusive);
                scan.setScanType(LogicalScanNodeTableType::PRIMARY_KEY_SCAN);
                scan.setExtraInfo(std::move(extraInfo));
                scan.computeFlatSchema();
            }
        }
    }
    if (scan.getScanType() == LogicalScanNodeTableType::SCAN && tableIDs.size() == 1) {
        auto secondaryIndexComparison = popSecondaryARTEqualityComparison(predicateSet, *nodeID,
            tableIDs[0], context, dbName, [&](const PropertyExpression& property) {
                return isSecondaryARTLookupCheaper(tableIDs[0], dbName, 1 /* numKeys */, property);
            });
        if (secondaryIndexComparison.has_value()) {
            scan.setScanType(LogicalScanNodeTableType::SECONDARY_INDEX_SCAN);
            scan.setExtraInfo(std::make_unique<SecondaryIndexScanInfo>(
                secondaryIndexComparison->second, secondaryIndexComparison->first));
            scan.computeFlatSchema();
        } else {
            std::string indexName;
            auto keyList = popInListComparison(predicateSet, *nodeID,
                [&](const PropertyExpression& property, const Expression& list) {
                    indexName = getSecondaryARTIndexName(property, tableIDs[0], context, dbName);
                    return !indexName.empty() && isSecondaryARTLookupCheaper(tableIDs[0], dbName,
                                                     getNumKeys(list), property);
                });
            if (keyList != nullptr) {
                scan.setScanType(LogicalScanNodeTableType::SECONDARY_INDEX_SCAN);
                scan.setExtraInfo(std::make_unique<SecondaryIndexScanInfo>(indexName,
                    std::move(keyList), true /* isKeyList */));
                scan.computeFlatSchema();
            }
        }
    }
    return finishPushDown(op);
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::visitTableFunctionCallReplace(
    const std::shared_ptr<LogicalOperator>& op) {
    auto& tableFunctionCall = op->cast<LogicalTableFunctionCall>();
    if (!tableFunctionCall.getTableFunc().supportsPushDownFunc()) {
        return finishPushDown(op);
    }
    std::vector<ColumnPredicateSet> columnPredicates;
    std::unordered_set<const Expression*> pushedPredicates;
    auto predicates = predicateSet.getAllPredicates();
    for (auto& column : tableFunctionCall.getBindData()->columns) {
        auto columnPredicateSet = ColumnPredicateSet();
        for (auto& predicate : predicates) {
            auto columnPredicate = ColumnPredicateUtil::tryConvert(*column, *predicate);
            if (columnPredicate == nullptr) {
                continue;
            }
            columnPredicateSet.addPredicate(std::move(columnPredicate));
            pushedPredicates.insert(predicate.get());
        }
        columnPredicates.push_back(std::move(columnPredicateSet));
    }
    tableFunctionCall.setColumnPredicates(std::move(columnPredicates));
    auto remainingPredicates = PredicateSet();
    for (auto& predicate : predicates) {
        if (!pushedPredicates.contains(predicate.get())) {
            remainingPredicates.addPredicate(predicate);
        }
    }
    predicateSet = std::move(remainingPredicates);
    return finishPushDown(op);
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::visitExtendReplace(
    const std::shared_ptr<LogicalOperator>& op) {
    if (op->ptrCast<BaseLogicalExtend>()->isRecursive() ||
        !context->getClientConfig()->enableZoneMap) {
        return visitChildren(op);
    }
    auto& extend = op->cast<LogicalExtend>();
    // Apply column predicates.
    auto columnPredicates =
        getColumnPredicateSets(extend.getProperties(), predicateSet.getAllPredicates());
    extend.setPropertyPredicates(std::move(columnPredicates));
    return visitChildren(op);
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::finishPushDown(
    std::shared_ptr<LogicalOperator> op) {
    if (predicateSet.isEmpty()) {
        return op;
    }
    auto predicates = predicateSet.getAllPredicates();
    auto root = appendFiltersStatsAware(std::move(predicates), op);
    predicateSet.clear();
    return root;
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::appendFilters(
    const expression_vector& predicates, std::shared_ptr<LogicalOperator> child) {
    if (predicates.empty()) {
        return child;
    }
    auto root = child;
    for (auto& p : predicates) {
        root = appendFilter(p, root);
    }
    return root;
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::appendFiltersStatsAware(
    expression_vector predicates, std::shared_ptr<LogicalOperator> child) {
    if (predicates.empty() || cardinalityEstimator == nullptr) {
        return appendFilters(predicates, std::move(child));
    }
    auto root = child;
    while (!predicates.empty()) {
        auto bestIdx = 0u;
        auto bestCardinality = cardinalityEstimator->estimateFilter(*root, *predicates[bestIdx]);
        for (auto i = 1u; i < predicates.size(); ++i) {
            const auto cardinality = cardinalityEstimator->estimateFilter(*root, *predicates[i]);
            if (cardinality < bestCardinality) {
                bestCardinality = cardinality;
                bestIdx = i;
            }
        }
        auto predicate = predicates[bestIdx];
        predicates.erase(predicates.begin() + bestIdx);
        root = appendFilter(std::move(predicate), root);
        root->setCardinality(bestCardinality);
    }
    return root;
}

std::shared_ptr<LogicalOperator> FilterPushDownOptimizer::appendFilter(
    std::shared_ptr<Expression> predicate, std::shared_ptr<LogicalOperator> child) {
    auto printInfo = std::make_unique<OPPrintInfo>();
    auto filter = std::make_shared<LogicalFilter>(std::move(predicate), std::move(child));
    filter->computeFlatSchema();
    return filter;
}

void PredicateSet::addPredicate(std::shared_ptr<Expression> predicate) {
    if (predicate->expressionType == ExpressionType::EQUALS) {
        equalityPredicates.push_back(std::move(predicate));
    } else {
        nonEqualityPredicates.push_back(std::move(predicate));
    }
}

static bool isNodePrimaryKey(const Expression& expression, const Expression& nodeID) {
    if (expression.expressionType != ExpressionType::PROPERTY) {
        // not property
        return false;
    }
    auto& property = expression.constCast<PropertyExpression>();
    if (property.getVariableName() != nodeID.constCast<PropertyExpression>().getVariableName()) {
        // not property for node
        return false;
    }
    return property.isPrimaryKey();
}

std::shared_ptr<Expression> PredicateSet::popNodePKEqualityComparison(const Expression& nodeID) {
    // We pop when the first primary key equality comparison is found.
    auto resultPredicateIdx = INVALID_IDX;
    for (auto i = 0u; i < equalityPredicates.size(); ++i) {
        auto predicate = equalityPredicates[i];
        if (isNodePrimaryKey(*predicate->getChild(0), nodeID)) {
            resultPredicateIdx = i;
            break;
        } else if (isNodePrimaryKey(*predicate->getChild(1), nodeID)) {
            // Normalize primary key to LHS.
            auto leftChild = predicate->getChild(0);
            auto rightChild = predicate->getChild(1);
            predicate->setChild(1, leftChild);
            predicate->setChild(0, rightChild);
            resultPredicateIdx = i;
            break;
        }
    }
    if (resultPredicateIdx != INVALID_IDX) {
        auto result = equalityPredicates[resultPredicateIdx];
        equalityPredicates.erase(equalityPredicates.begin() + resultPredicateIdx);
        return result;
    }
    return nullptr;
}

PrimaryKeyRangePredicate PredicateSet::popNodePKRangeComparison(const Expression& nodeID) {
    PrimaryKeyRangePredicate result;
    auto lowerPredicateIdx = INVALID_IDX;
    auto upperPredicateIdx = INVALID_IDX;
    for (auto i = 0u; i < nonEqualityPredicates.size(); ++i) {
        auto predicate = nonEqualityPredicates[i];
        if (!ExpressionTypeUtil::isComparison(predicate->expressionType) ||
            predicate->expressionType == ExpressionType::NOT_EQUALS) {
            continue;
        }
        auto comparisonType = predicate->expressionType;
        std::shared_ptr<Expression> bound;
        if (isNodePrimaryKey(*predicate->getChild(0), nodeID)) {
            bound = predicate->getChild(1);
        } else if (isNodePrimaryKey(*predicate->getChild(1), nodeID)) {
            bound = predicate->getChild(0);
            comparisonType = ExpressionTypeUtil::reverseComparisonDirection(comparisonType);
        } else {
            continue;
        }
        if (!isConstantExpression(bound)) {
            continue;
        }
        switch (comparisonType) {
        case ExpressionType::GREATER_THAN:
            if (lowerPredicateIdx != INVALID_IDX) {
                return {};
            }
            result.lowerBound = bound;
            result.lowerInclusive = false;
            lowerPredicateIdx = i;
            break;
        case ExpressionType::GREATER_THAN_EQUALS:
            if (lowerPredicateIdx != INVALID_IDX) {
                return {};
            }
            result.lowerBound = bound;
            result.lowerInclusive = true;
            lowerPredicateIdx = i;
            break;
        case ExpressionType::LESS_THAN:
            if (upperPredicateIdx != INVALID_IDX) {
                return {};
            }
            result.upperBound = bound;
            result.upperInclusive = false;
            upperPredicateIdx = i;
            break;
        case ExpressionType::LESS_THAN_EQUALS:
            if (upperPredicateIdx != INVALID_IDX) {
                return {};
            }
            result.upperBound = bound;
            result.upperInclusive = true;
            upperPredicateIdx = i;
            break;
        default:
            break;
        }
    }
    std::array<idx_t, 2> predicateIndices{lowerPredicateIdx, upperPredicateIdx};
    std::sort(predicateIndices.begin(), predicateIndices.end(), std::greater<>());
    for (auto predicateIdx : predicateIndices) {
        if (predicateIdx != INVALID_IDX) {
            nonEqualityPredicates.erase(nonEqualityPredicates.begin() + predicateIdx);
        }
    }
    return result;
}

expression_vector PredicateSet::getAllPredicates() {
    expression_vector result;
    result.insert(result.end(), equalityPredicates.begin(), equalityPredicates.end());
    result.insert(result.end(), nonEqualityPredicates.begin(), nonEqualityPredicates.end());
    return result;
}

} // namespace optimizer
} // namespace lbug
