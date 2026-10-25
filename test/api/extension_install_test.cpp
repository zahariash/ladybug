#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include "extension/extension.h"
#include "gtest/gtest.h"
#include "main/connection.h"
#include "main/database.h"
#include "test_helper/test_helper.h"
#include <format>

using namespace lbug::main;
namespace fs = std::filesystem;

namespace lbug {
namespace testing {

#ifndef __WASM__ // extensions are not available in WASM
static void writeFile(const fs::path& path, const std::string& content) {
    std::ofstream file(path, std::ios::binary);
    file << content;
}

static std::string readFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Packagers place extension files ahead of time and run INSTALL without network access.
TEST(ExtensionInstallTest, InstallKeepsInstalledFileWithoutDownload) {
    auto database = std::make_unique<Database>(":memory:", SystemConfig());
    auto conn = std::make_unique<Connection>(database.get());
    auto homeDir = TestHelper::getTempDir("ExtensionInstall");
    ASSERT_TRUE(conn->query(std::format("CALL home_directory='{}'", homeDir))->isSuccess());
    auto libPath = fs::path(
        extension::ExtensionUtils::getLocalPathForExtensionLib(conn->getClientContext(), "json"));
    fs::create_directories(libPath.parent_path());
    const std::string content = "placed by a packager";
    writeFile(libPath, content);
    auto modified = fs::last_write_time(libPath);

    // Nothing listens on port 1, so a download from this repo fails.
    const std::string unreachableRepo = "http://127.0.0.1:1/";
    for (const auto& query :
        {std::string("INSTALL json"), std::format("INSTALL json FROM '{}'", unreachableRepo)}) {
        auto result = conn->query(query);
        ASSERT_TRUE(result->isSuccess()) << query << ": " << result->getErrorMessage();
        EXPECT_EQ(result->getNext()->getValue(0)->getValue<std::string>(),
            "Extension: json is already installed.\nTo update it, you can run: UPDATE json.");
    }
    EXPECT_EQ(readFile(libPath), content);
    EXPECT_EQ(fs::last_write_time(libPath), modified);

    auto result = conn->query(std::format("FORCE INSTALL json FROM '{}'", unreachableRepo));
    ASSERT_FALSE(result->isSuccess());
    EXPECT_NE(result->getErrorMessage().find("Failed to download extension"), std::string::npos)
        << result->getErrorMessage();
}
#endif

} // namespace testing
} // namespace lbug
