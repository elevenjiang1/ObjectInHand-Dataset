#ifndef TAMS_BIOTAC_CONTACTS
#define TAMS_BIOTAC_CONTACTS

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <sr_robot_msgs/BiotacAll.h>
#include <tams_biotac/BiotacAll.h>
#include <tams_biotac/ContactArray.h>
#include <tams_biotac/force.h>

#define N_EXP 2.0

// coordinates of the electrodes on the Biotacs in mm
const float x_electrode[19] = {0.993,   -2.700, -6.200, -8.000, -10.500,
                               -13.400, 4.763,  3.031,  3.031,  1.299,
                               0.993,   -2.700, -6.200, -8.000, -10.500,
                               -13.400, -2.800, -9.800, -13.600};
const float y_electrode[19] = {
    -4.855, -3.513, -3.513, -4.956, -3.513, -4.956, 0.000, -1.950, 1.950, 0.000,
    4.855,  3.513,  3.513,  4.956,  3.513,  4.956,  0.000, 0.000,  0.000};
const float z_electrode[19] = {-1.116, -3.670, -3.670, -1.116, -3.670,
                               -1.116, -2.330, -3.330, -3.330, -4.330,
                               -1.116, -3.670, -3.670, -1.116, -3.670,
                               -1.116, -5.080, -5.080, -5.080};

const std::string finger[5] = {"rh_ff_biotac_link", "rh_mf_biotac_link",
                               "rh_rf_biotac_link", "rh_lf_biotac_link",
                               "rh_th_biotac_link"};

class Contacts {
public:
  Contacts();
  Contacts(std::string input_topic, std::string output_topic);
  ~Contacts(){};

  tams_biotac::ContactArray getContacts(sr_robot_msgs::BiotacAll msg);
  tams_biotac::ContactArray getContacts(tams_biotac::BiotacAll msg);

  // The frame_id is not part of the sr_robot_msgs::Biotac msg
  tams_biotac::Contact getContact(sr_robot_msgs::Biotac msg);
  tams_biotac::Contact getContact(tams_biotac::Biotac msg);

  static void mapPointToSurface(geometry_msgs::Point &point);

private:
  ros::NodeHandle node_handle_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::Publisher marker_publisher_;
  bool cal_;
  Force force_;

  void srCallback(const sr_robot_msgs::BiotacAll::ConstPtr &msg);
  void tamsCallback(const tams_biotac::BiotacAll::ConstPtr &msg);

  tams_biotac::ContactArray current_contacts_;

  void visualizeContacts();
};

#endif
