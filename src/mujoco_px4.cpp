#include <common/mavlink.h>
#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
constexpr double kExchangeDt = 0.004;
constexpr double kOmegaMax = 3000.0;
constexpr double kKf = 2.063751543951938e-6;
constexpr double kLatitude = 47.397742;
constexpr double kLongitude = 8.545594;
constexpr double kAltitude = 488.0;
constexpr double kEarthRadius = 6378137.0;
constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

class Socket {
 public:
  explicit Socket(int fd = -1) : fd_(fd) {}
  ~Socket() { if (fd_ >= 0) ::close(fd_); }
  Socket(const Socket &) = delete;
  Socket &operator=(const Socket &) = delete;
  int get() const { return fd_; }
  void reset(int fd) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = fd;
  }
 private:
  int fd_;
};

// One TCP stream per simulated vehicle. Binding loopback keeps the first
// implementation local; this executable is a simulator, not a flight driver.
class SimulatorLink {
 public:
  explicit SimulatorLink(int port) {
    server_.reset(::socket(AF_INET, SOCK_STREAM, 0));
    require(server_.get() >= 0, "Cannot create simulator socket");
    int enabled = 1;
    setsockopt(server_.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(::bind(server_.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0,
            "Cannot bind simulator port: " + std::string(std::strerror(errno)));
    require(::listen(server_.get(), 1) == 0, "Cannot listen for PX4");
  }

  bool accept() {
    while (rclcpp::ok()) {
      pollfd fd{server_.get(), POLLIN, 0};
      const int ready = ::poll(&fd, 1, 100);
      if (ready < 0 && errno == EINTR) continue;
      require(ready >= 0, "Simulator accept poll failed");
      if (ready > 0) {
        client_.reset(::accept(server_.get(), nullptr, nullptr));
        require(client_.get() >= 0, "PX4 accept failed");
        int enabled = 1;
        setsockopt(client_.get(), IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
        timeval timeout{2, 0};
        setsockopt(client_.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        return true;
      }
    }
    return false;
  }

  void send(const mavlink_message_t &message) {
    std::array<uint8_t, MAVLINK_MAX_PACKET_LEN> buffer{};
    const size_t length = mavlink_msg_to_send_buffer(buffer.data(), &message);
    size_t offset = 0;
    while (offset < length && rclcpp::ok()) {
      const auto n = ::send(client_.get(), buffer.data() + offset, length - offset, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) continue;
      require(n > 0, "PX4 disconnected or simulator send timed out");
      offset += static_cast<size_t>(n);
    }
  }

  bool receive_actuators(mavlink_hil_actuator_controls_t &controls, int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    do {
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - Clock::now()).count();
      pollfd fd{client_.get(), POLLIN, 0};
      const int ready = ::poll(&fd, 1, static_cast<int>(std::clamp<int64_t>(left, 0, 100)));
      if (ready < 0 && errno == EINTR) continue;
      require(ready >= 0, "Simulator receive poll failed");
      if (ready > 0) {
        require(!(fd.revents & (POLLERR | POLLNVAL)), "PX4 disconnected or simulator socket failed");
        std::array<uint8_t, 4096> buffer{};
        const auto n = ::recv(client_.get(), buffer.data(), buffer.size(), MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        require(n > 0, "PX4 disconnected");
        bool found = false;
        for (ssize_t i = 0; i < n; ++i) {
          mavlink_message_t message{};
          mavlink_status_t status{};
          if (mavlink_parse_char(MAVLINK_COMM_0, buffer[i], &message, &status) &&
              message.msgid == MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS) {
            mavlink_msg_hil_actuator_controls_decode(&message, &controls);
            found = true;
          }
        }
        if (found) return true;
      }
    } while (rclcpp::ok() && Clock::now() < deadline);
    return false;
  }
 private:
  Socket server_, client_;
};

struct Vehicle {
  int body;
  int accel;
  int gyro;
  std::array<int, 4> rotors;
};

Vehicle resolve_vehicle(const mjModel *m, const std::string &prefix) {
  auto id = [&](mjtObj kind, const std::string &name) {
    const int result = mj_name2id(m, kind, name.c_str());
    require(result >= 0, "Missing model element: " + name);
    return result;
  };
  require(m->nu == 4, "PX4 scene must contain exactly four rotor actuators and no AcroMode controller");
  Vehicle v{};
  v.body = id(mjOBJ_BODY, prefix);
  v.accel = id(mjOBJ_SENSOR, prefix + "_accel");
  v.gyro = id(mjOBJ_SENSOR, prefix + "_gyro");
  require(m->sensor_type[v.accel] == mjSENS_ACCELEROMETER &&
          m->sensor_type[v.gyro] == mjSENS_GYRO, "Expected native accelerometer and gyro sensors");
  for (int i = 0; i < 4; ++i) {
    v.rotors[i] = id(mjOBJ_ACTUATOR, prefix + "_rotor" + std::to_string(i + 1));
    const int actuator = v.rotors[i];
    require(m->actuator_plugin[actuator] >= 0, "RotorMotor plugin required");
    require(std::abs(m->actuator_ctrlrange[2 * actuator + 1] - kOmegaMax) < 1e-6,
            "Rotor speed limit differs from the PX4 airframe");
    const char *kf = mj_getPluginConfig(m, m->actuator_plugin[actuator], "kf");
    require(kf && std::abs(std::stod(kf) - kKf) < 1e-12,
            "Rotor thrust coefficient differs from the PX4 airframe");
  }
  return v;
}

uint64_t sim_usec(const mjData *d) {
  return std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(d->time * 1e6)));
}

void send_sensors(SimulatorLink &link, const mjModel *m, const mjData *d,
                  const Vehicle &v, uint64_t tick) {
  const auto *a = d->sensordata + m->sensor_adr[v.accel];
  const auto *g = d->sensordata + m->sensor_adr[v.gyro];
  mavlink_hil_sensor_t imu{};
  imu.time_usec = sim_usec(d);
  // MuJoCo accelerometers report specific force already: do not subtract
  // gravity again. FLU -> FRD rotates both sensor vectors by pi about body X.
  imu.xacc = a[0]; imu.yacc = -a[1]; imu.zacc = -a[2];
  imu.xgyro = g[0]; imu.ygyro = -g[1]; imu.zgyro = -g[2];
  imu.fields_updated = 0x3f;
  const auto *position = d->xpos + 3 * v.body;
  if (tick % 5 == 0) {
    static std::mt19937 noise_rng(42);
    static std::normal_distribution<double> mag_noise_gauss(0.0, 0.0005);
    static std::normal_distribution<double> baro_noise_hpa(0.0, 0.01);
    // Ideal fixed magnetic field near the Zurich GPS origin, in ENU gauss.
    // Rotate the world field into body FLU, then body FRD.
    const mjtNum mag_world[3]{0.01, 0.21, -0.43};
    mjtNum mag_body[3]{};
    mju_mulMatTVec(mag_body, d->xmat + 9 * v.body, mag_world, 3, 3);
    imu.xmag = mag_body[0] + mag_noise_gauss(noise_rng);
    imu.ymag = -mag_body[1] + mag_noise_gauss(noise_rng);
    imu.zmag = -mag_body[2] + mag_noise_gauss(noise_rng);
    const double altitude = kAltitude + position[2];
    // A stationary, perfectly constant barometer trips PX4's stuck-sensor
    // detector. Model 1 Pa RMS pressure noise with a reproducible seed.
    imu.abs_pressure = 1013.25 * std::pow(1.0 - altitude / 44330.0, 5.255) +
                       baro_noise_hpa(noise_rng);
    imu.pressure_alt = altitude;
    imu.temperature = 15.0 - 0.0065 * altitude;
    imu.fields_updated |= 0x1bc0;  // mag, absolute pressure, pressure altitude, temperature
  }
  mavlink_message_t message{};
  mavlink_msg_hil_sensor_encode(1, MAV_COMP_ID_IMU, &message, &imu);
  link.send(message);

  if (tick % 25 == 0) {
    mjtNum velocity[6]{};
    mj_objectVelocity(m, d, mjOBJ_XBODY, v.body, velocity, 0);
    mavlink_hil_gps_t gps{};
    gps.time_usec = sim_usec(d);
    gps.lat = std::llround((kLatitude + position[1] / kEarthRadius * 180 / kPi) * 1e7);
    gps.lon = std::llround((kLongitude + position[0] /
        (kEarthRadius * std::cos(kLatitude * kPi / 180)) * 180 / kPi) * 1e7);
    gps.alt = std::llround((kAltitude + position[2]) * 1000);
    gps.eph = 30; gps.epv = 50;
    gps.vn = std::lround(velocity[4] * 100);
    gps.ve = std::lround(velocity[3] * 100);
    gps.vd = std::lround(-velocity[5] * 100);
    gps.vel = std::lround(std::hypot(velocity[3], velocity[4]) * 100);
    const double course = std::atan2(velocity[3], velocity[4]) * 180 / kPi;
    gps.cog = std::lround(std::fmod(course + 360, 360) * 100);
    gps.fix_type = 3;
    gps.satellites_visible = 12;
    mavlink_msg_hil_gps_encode(1, MAV_COMP_ID_IMU, &message, &gps);
    link.send(message);
  }
}

void apply_controls(mjData *d, const Vehicle &v, const mavlink_hil_actuator_controls_t &controls) {
  const bool armed = controls.mode & MAV_MODE_FLAG_SAFETY_ARMED;
  for (int i = 0; i < 4; ++i) {
    const float u = controls.controls[i];
    require(std::isfinite(u), "PX4 returned a nonfinite motor command");
    require(u >= -1e-6f && u <= 1.000001f, "Expected non-reversible normalized motor outputs [0,1]");
    // THR_MDL_FAC=1 in the supplied PX4 configuration performs the square-root
    // thrust linearization. The simulator output is consequently normalized speed.
    d->ctrl[v.rotors[i]] = armed ? kOmegaMax * std::clamp<double>(u, 0.0, 1.0) : 0.0;
  }
}
}  // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    auto node = std::make_shared<rclcpp::Node>("mujoco_px4");
    const auto model_path = node->declare_parameter<std::string>("model_path", "");
    const auto plugins = node->declare_parameter<std::string>("mujoco_plugin_dir", "");
    const auto prefix = node->declare_parameter<std::string>("quad_name", "quadrotor");
    const int port = node->declare_parameter<int>("simulator_port", 4560);
    const double duration = node->declare_parameter<double>("duration_s", 0.0);
    const auto csv_path = node->declare_parameter<std::string>("csv_path", "");
    require(!model_path.empty() && !plugins.empty(), "Set model_path and mujoco_plugin_dir");
    require(port > 1023 && port <= 65535, "Invalid simulator TCP port");
    require(std::isfinite(duration) && duration >= 0, "duration_s must be finite and nonnegative");
    require(mjVERSION_HEADER == mj_version(), "MuJoCo headers/library version mismatch");
    mj_loadAllPluginLibraries(plugins.c_str(), nullptr);
    char error[1024]{};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error)), mj_deleteModel);
    require(model != nullptr, "Cannot load model: " + std::string(error));
    const int substeps = std::lround(kExchangeDt / model->opt.timestep);
    require(substeps >= 1 && std::abs(substeps * model->opt.timestep - kExchangeDt) < 1e-10,
            "Physics timestep must divide the 4 ms sensor exchange interval exactly");
    const Vehicle vehicle = resolve_vehicle(model.get(), prefix);
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
    require(data != nullptr, "Cannot allocate MuJoCo data");
    std::ofstream csv;
    if (!csv_path.empty()) {
      csv.open(csv_path);
      require(csv.is_open(), "Cannot open CSV output");
      csv << "time_s,x_enu,y_enu,z_enu,p_flu,q_flu,r_flu,armed,u1,u2,u3,u4\n";
    }
    SimulatorLink link(port);
    RCLCPP_INFO(node->get_logger(), "Waiting for PX4 on TCP 127.0.0.1:%d", port);
    if (link.accept()) {
      RCLCPP_INFO(node->get_logger(), "PX4 connected; sensors at 250 Hz, physics at %.0f Hz",
                  1.0 / model->opt.timestep);
      bool synchronized = false;
      uint64_t tick = 0;
      auto next_tick = Clock::now();
      const auto startup_deadline = next_tick + std::chrono::seconds(30);
      mavlink_hil_actuator_controls_t controls{};
      while (rclcpp::ok() && (duration == 0 || data->time < duration)) {
        // Keep networking outside MuJoCo callbacks: RK4 invokes them several
        // times per step. Refresh sensors at the actual exchange timestamp.
        mj_forward(model.get(), data.get());
        send_sensors(link, model.get(), data.get(), vehicle, tick);
        const bool received = link.receive_actuators(controls, synchronized ? 2000 : 1);
        if (!rclcpp::ok()) break;
        if (received) {
          require(controls.flags & 1, "PX4 must be built with lockstep enabled");
          if (!synchronized) {
            synchronized = true;
            RCLCPP_INFO(node->get_logger(), "First motor output received; lockstep active");
          }
          apply_controls(data.get(), vehicle, controls);
        } else {
          require(!synchronized, "PX4 actuator timeout: simulation stopped");
          require(Clock::now() < startup_deadline, "No actuator outputs during PX4 startup");
        }
        if (csv.is_open() && tick % 5 == 0) {
          const auto *p = data->xpos + 3 * vehicle.body;
          const auto *g = data->sensordata + model->sensor_adr[vehicle.gyro];
          csv << data->time << ',' << p[0] << ',' << p[1] << ',' << p[2] << ','
              << g[0] << ',' << g[1] << ',' << g[2] << ','
              << bool(controls.mode & MAV_MODE_FLAG_SAFETY_ARMED);
          for (int i = 0; i < 4; ++i) csv << ',' << controls.controls[i];
          csv << '\n';
        }
        for (int i = 0; i < substeps; ++i) mj_step(model.get(), data.get());
        ++tick;
        next_tick += std::chrono::microseconds(4000);
        // Never accelerate to catch up after a stall.
        if (Clock::now() > next_tick + std::chrono::milliseconds(100)) next_tick = Clock::now();
        std::this_thread::sleep_until(next_tick);
      }
      RCLCPP_INFO(node->get_logger(), "Stopped at %.3f simulated seconds", data->time);
    }
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("mujoco_px4"), "%s", error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
