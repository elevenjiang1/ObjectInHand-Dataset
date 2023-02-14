#ifndef TAMS_BIOTAC_VISUALIZE_ELECTRODES
#define TAMS_BIOTAC_VISUALIZE_ELECTRODES

#include <ros/ros.h>

#include <sr_robot_msgs/BiotacAll.h>
#include <visualization_msgs/MarkerArray.h>

class VisualizeElectrodes {
public:
  VisualizeElectrodes();
  VisualizeElectrodes(std::string input_topic);
  ~VisualizeElectrodes(){};

  void visualize(const sr_robot_msgs::BiotacAll& msg);

private:
  ros::NodeHandle node_handle_;
  ros::Subscriber subscriber_;
  ros::Publisher marker_publisher_;
  ros::WallTime last_publish_;

  visualization_msgs::MarkerArray marker_array_;

  void callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg);

  // set position and size of markers
  void initializeMarkerArray();
};

#endif
