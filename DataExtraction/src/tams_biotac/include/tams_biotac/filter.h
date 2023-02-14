#ifndef TAMS_BIOTAC_FILTER
#define TAMS_BIOTAC_FILTER

#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>

class Filter {
public:
  Filter(float alph = 0.5);
  Filter(std::string input_topic, std::string output_topic, float alpha = 0.5);
  ~Filter();

  sr_robot_msgs::BiotacAll filter(const sr_robot_msgs::BiotacAll sensor_data);

private:
  ros::NodeHandle node_handle_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;

  float alpha_;

  sr_robot_msgs::BiotacAll *filtered_msg_;

  void callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg);

  // Exponentially Weighted Moving Average
  short eWMA(short new_value, short old_value);
};

#endif
