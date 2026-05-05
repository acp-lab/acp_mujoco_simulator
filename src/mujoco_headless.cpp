#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unistd.h>

namespace {

constexpr const char *kPluginDirName = "mujoco_plugin";
void SignalHandler(int) { _exit(0); }

template <typename T>
void DeclareAndReadParam(const rclcpp::Node::SharedPtr &node,
                         const std::string &name, T &value, const char *fmt) {
  node->declare_parameter<T>(name, value);
  if (!node->get_parameter(name, value)) {
    RCLCPP_ERROR(node->get_logger(), "failed to read parameter: %s",
                 name.c_str());
    return;
  }

  if constexpr (std::is_same_v<T, std::string>) {
    RCLCPP_INFO(node->get_logger(), "%s: %s", name.c_str(), value.c_str());
  } else if constexpr (std::is_same_v<T, bool>) {
    RCLCPP_INFO(node->get_logger(), "%s: %s", name.c_str(),
                value ? "true" : "false");
  } else {
    RCLCPP_INFO(node->get_logger(), (name + std::string(": ") + fmt).c_str(),
                value);
  }
}

std::string GetExecutableDir() {
  std::error_code ec;
  const auto exe_path = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) {
    std::cerr << "Failed to resolve executable path: " << ec.message() << '\n';
    return "";
  }
  return exe_path.parent_path().string();
}

void ScanPluginLibraries(const std::filesystem::path &plugin_dir) {
  const int builtin_plugins = mjp_pluginCount();
  if (builtin_plugins) {
    std::cout << "Built-in plugins:\n";
    for (int i = 0; i < builtin_plugins; ++i) {
      std::cout << "    " << mjp_getPluginAtSlot(i)->name << '\n';
    }
  }

  std::filesystem::path resolved_plugin_dir = plugin_dir;
  if (resolved_plugin_dir.empty()) {
    const std::string executable_dir = GetExecutableDir();
    if (executable_dir.empty()) {
      std::cerr << "Failed to resolve plugin directory\n";
      return;
    }
    resolved_plugin_dir =
        std::filesystem::path(executable_dir) / kPluginDirName;
  }

  mj_loadAllPluginLibraries(
      resolved_plugin_dir.c_str(),
      +[](const char *filename, int first, int count) {
        std::cout << "Plugins registered by library '" << filename << "':\n";
        for (int i = first; i < first + count; ++i) {
          std::cout << "    " << mjp_getPluginAtSlot(i)->name << '\n';
        }
      });
}

mjModel *LoadModel(const char *filename) {
  char error[1024] = "Could not load model";

  const std::string filename_str(filename);
  if (filename_str.size() >= 4 &&
      filename_str.substr(filename_str.size() - 4) == ".mjb") {
    mjModel *model = mj_loadModel(filename, nullptr);
    if (!model) {
      std::cerr << error << '\n';
    }
    return model;
  }

  mjModel *model = mj_loadXML(filename, nullptr, error, sizeof(error));
  if (!model) {
    std::cerr << error << '\n';
  }
  return model;
}

} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("mujoco_headless");

  std::string model_path;
  std::string mujoco_plugin_dir;
  DeclareAndReadParam(node, "model_path", model_path, "%s");
  DeclareAndReadParam(node, "mujoco_plugin_dir", mujoco_plugin_dir, "%s");

  const auto non_ros_args = rclcpp::remove_ros_arguments(argc, argv);
  if (non_ros_args.size() == 2) {
    model_path = non_ros_args[1];
  }

  if (model_path.empty()) {
    std::cerr << "Usage: mujoco_headless <model.xml|model.mjb>\n"
              << "   or: mujoco_headless --ros-args -p "
                 "model_path:=<model.xml|model.mjb>\n";
    rclcpp::shutdown();
    return 1;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  std::cout << "MuJoCo version " << mj_versionString() << '\n';
  if (mjVERSION_HEADER != mj_version()) {
    std::cerr << "Headers and library have different versions\n";
    rclcpp::shutdown();
    return 1;
  }

  ScanPluginLibraries(mujoco_plugin_dir);

  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      LoadModel(model_path.c_str()), mj_deleteModel);
  if (!model) {
    rclcpp::shutdown();
    return 1;
  }

  std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
      mj_makeData(model.get()), mj_deleteData);
  if (!data) {
    std::cerr << "Failed to allocate mjData\n";
    rclcpp::shutdown();
    return 1;
  }

  mj_forward(model.get(), data.get());

  using clock = std::chrono::steady_clock;
  auto next_tick = clock::now();
  const auto step_duration = std::chrono::duration<double>(model->opt.timestep);

  while (true) {
    mj_step(model.get(), data.get());
    next_tick += std::chrono::duration_cast<clock::duration>(step_duration);
    std::this_thread::sleep_until(next_tick);
  }

  return 0;
}
