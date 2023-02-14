#include <tams_biotac/tactile_calibrated.h>

#include <yaml-cpp/yaml.h>

TactileCalibrated::TactileCalibrated() {
  ros::NodeHandle private_node_handle("~");
  std::string yaml_path;
  private_node_handle.param<std::string>("calibration_file", yaml_path,
                                         "biotac_info/calibration.yaml");
  parseYaml(yaml_path);
}

TactileCalibrated::TactileCalibrated(std::string input_topic,
                                     std::string output_topic) {
  subscriber_ = node_handle_.subscribe(input_topic, 1,
                                       &TactileCalibrated::callback, this);
  publisher_ = node_handle_.advertise<tams_biotac::BiotacAll>(output_topic, 1);

  ros::NodeHandle private_node_handle("~");
  std::string yaml_path;
  private_node_handle.param<std::string>("calibration_file", yaml_path,
                                         "biotac_info/calibration.yaml");

  parseYaml(yaml_path);
}

void TactileCalibrated::parseYaml(std::string path) {
  YAML::Node doc;
  doc = YAML::LoadFile(path);

  for (int i = 0; i < doc.size(); i++) {
    calibration_values_[i].frame_id = doc[i]["frame_id"].as<std::string>();
    for (int j = 0; j < 19; j++) {
      calibration_values_[i].c[j] = doc[i][j + 1][0].as<float>();
      calibration_values_[i].p[j] = doc[i][j + 1][1].as<float>();
    }
  }
}

void TactileCalibrated::callback(
    const sr_robot_msgs::BiotacAll::ConstPtr &msg) {
  tams_biotac::BiotacAll result;

  for (int i = 0; i < 5; i++) {
    result.tactiles[i].header = msg->header;
    result.tactiles[i].header.frame_id = calibration_values_[i].frame_id;
    result.tactiles[i].pac0 = msg->tactiles[i].pac0;
    result.tactiles[i].pac1 = msg->tactiles[i].pac1;
    result.tactiles[i].tac = msg->tactiles[i].tac;
    result.tactiles[i].tdc = msg->tactiles[i].tdc;
    result.tactiles[i].pdc = msg->tactiles[i].pdc;

    for (int j = 0; j < 19; j++) {
      result.tactiles[i].electrodes[j] =
          ((msg->tactiles[i].electrodes[j]) * calibration_values_[i].p[j]);// +
//          calibration_values_[i].c[j];
    }
  }

  publisher_.publish(result);
}

// CalibrationValues[] TactileCalibrated::getCalibrationValues() {
//  return calibration_values_;
//}
