#ifndef TAMS_BIOTAC_NORMALIZATION
#define TAMS_BIOTAC_NORMALIZATION

#include <deque>
#include <ros/ros.h>

#include <sr_hardware_interface/tactile_sensors.hpp>
#include <sr_robot_msgs/BiotacAll.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Empty.h>

#define SKIP 25
#define PDC_TH 5
#define ELEC_TH 15

class Normalization {
public:
  Normalization();
  Normalization(std::string input_topic, std::string output_topic);
  ~Normalization(){};

  sr_robot_msgs::BiotacAll normalize(sr_robot_msgs::BiotacAll msg);
  sr_robot_msgs::Biotac normalize(sr_robot_msgs::Biotac msg, int i = 0);
  tactiles::BiotacData normalize(tactiles::BiotacData msg, int i = 0);
  void resetOffset();

private:
  ros::NodeHandle node_handle_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::ServiceServer service_;

  tactiles::BiotacData offset_[5];
  std::deque<int> pdc_history_[5];
  int skip_count_[5];
  bool initialized_[5];
  bool periodic_normalization_;

  void init();
  void callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg);
  void resetCallback(const std_msgs::Empty::ConstPtr &msg);
  bool serviceCallback(std_srvs::Empty::Request &request,
                       std_srvs::Empty::Response &response);

  tactiles::BiotacData substractOffset(tactiles::BiotacData msg,
                                       tactiles::BiotacData offset);

  tactiles::BiotacData srToTactile(sr_robot_msgs::Biotac msg);
  sr_robot_msgs::Biotac tactileToSr(tactiles::BiotacData msg);
};

#endif
