#ifndef TAMS_BIOTAC_FORCE
#define TAMS_BIOTAC_FORCE

#include <ros/ros.h>
#include <sr_robot_msgs/BiotacAll.h>
#include <tams_biotac/Contact.h>

const float x_surface_electrode[19] = {
    0.001368,  -0.002700, -0.006200, -0.008000, -0.010500, -0.013400, 0.006288,
    0.004324,  0.004324,  0.002011,  0.001368,  -0.002700, -0.006200, -0.008000,
    -0.010500, -0.013400, -0.002800, -0.009800, -0.013600};

const float y_surface_electrode[19] = {
    -0.006690, -0.004840, -0.004840, -0.006829, -0.004840, -0.006829, 0.000000,
    -0.002782, 0.002782,  0.000000,  0.006690,  0.004840,  0.004840,  0.006829,
    0.004840,  0.006829,  0.000000,  0.000000,  0.000000};

const float z_surface_electrode[19] = {
    -0.001538, -0.005057, -0.005057, -0.001538, -0.005057, -0.001538, -0.003076,
    -0.004750, -0.004750, -0.006705, -0.001538, -0.005057, -0.005057, -0.001538,
    -0.005057, -0.001538, -0.007000, -0.007000, -0.007000};

class Force {
public:
  Force();
  Force(std::string input_topic);
  ~Force(){};

  float getForce(int pdc, float x, float y, float z);

private:
  ros::NodeHandle node_handle_;
  ros::Subscriber contact_subscriber_;
  ros::Subscriber tactile_subscriber_;
  ros::Publisher publisher_;

  float calibration_values_[19];
  int current_pdc_;

  void loadCalibrationValues();
  void contactCallback(const tams_biotac::Contact::ConstPtr &msg);
  void tactileCallback(const sr_robot_msgs::BiotacAll::ConstPtr &msg);
};

#endif
