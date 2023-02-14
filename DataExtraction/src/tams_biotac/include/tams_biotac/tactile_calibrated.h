#ifndef TAMS_BIOTAC_TACTILE_CALIBRATED
#define TAMS_BIOTAC_TACTILE_CALIBRATED

#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>
#include <tams_biotac/BiotacAll.h>

struct CalibrationValues {
  std::string frame_id;
  float p[19];
  float c[19];
};

class TactileCalibrated {
public:
  TactileCalibrated();
  TactileCalibrated(std::string input_topic, std::string output_topic);
  ~TactileCalibrated(){};

  // CalibrationValues[5] getCalibrationValues();

private:
  ros::NodeHandle node_handle_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;

  CalibrationValues calibration_values_[5];

  void parseYaml(std::string path);
  void callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg);
};

#endif
