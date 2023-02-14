#include <tams_biotac/subtract_temperature.h>

#include <fstream>
#include <yaml-cpp/yaml.h>

SubtractTemperature::SubtractTemperature() { readCalibrationYaml(); }

SubtractTemperature::SubtractTemperature(std::string input_topic,
                                         std::string output_topic) {
  subscriber_ = node_handle_.subscribe(input_topic, 1,
                                       &SubtractTemperature::callback, this);
  publisher_ =
      node_handle_.advertise<sr_robot_msgs::BiotacAll>(output_topic, 1);
  readCalibrationYaml();
}

sr_robot_msgs::BiotacAll
SubtractTemperature::subtract_temperature(const sr_robot_msgs::BiotacAll msg) {
  sr_robot_msgs::BiotacAll result = msg;

  if (!no_calibration_file_) {
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 19; j++)
        result.tactiles[i].electrodes[j] -=
            result.tactiles[i].tdc * calibration_values_[i][j];
    }
  }
  return result;
}

void SubtractTemperature::callback(
    const sr_robot_msgs::BiotacAll::ConstPtr &msg) {
  publisher_.publish(subtract_temperature(*msg));
}

void SubtractTemperature::readCalibrationYaml() {
  std::ifstream f("biotac_info/temperature_calibration.yaml");
  if (!f.good()) {
    ROS_WARN("No temperature calibration file found");
    no_calibration_file_ = true;
    return;
  }
  no_calibration_file_ = false;

  YAML::Node node = YAML::LoadFile("biotac_info/temperature_calibration.yaml");

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 19; j++) {
      calibration_values_[i][j] =
          node["temperature_calibration"][i][j].as<float>();
    }
  }
}
