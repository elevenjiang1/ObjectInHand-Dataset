#ifndef TAMS_BIOTAC_SUBTRACT_TEMPERATURE
#define TAMS_BIOTAC_SUBTRACT_TEMPERATURE

#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>

class SubtractTemperature {
public:
  SubtractTemperature();
  SubtractTemperature(std::string input_topic, std::string output_topic);
  ~SubtractTemperature(){};

  sr_robot_msgs::BiotacAll
  subtract_temperature(const sr_robot_msgs::BiotacAll msg);

private:
  ros::NodeHandle node_handle_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;

  bool no_calibration_file_;
  float calibration_values_[5][19];

  void callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg);
  void readCalibrationYaml();
};

#endif
