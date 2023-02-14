#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <sr_robot_msgs/BiotacAll.h>
#include <yaml-cpp/yaml.h>

ros::Publisher pub;
ros::Publisher pub2;
double cal[5][19];

void callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg){
  std_msgs::Float64MultiArray result;
  for (int i=0; i<5; i++) {
    result.data.push_back((4025 / log((155183-(46555*((float)msg->tactiles[i].tdc/4095)))/((float)msg->tactiles[i].tdc/4095))) - 273.15);
  }
  for (int i=0; i<5; i++) {
    result.data.push_back(-41.07/log((155183-(46555*((float)msg->tactiles[i].tac/4095)))/((float)msg->tactiles[i].tac/4095)));
  }
  // publish temperature in Celsius
  pub.publish(result);
  sr_robot_msgs::BiotacAll result2 = *msg;

  // publish temperature independant electrode values
  for (int i=0; i<5; i++) {
    for (int j=0; j<19; j++){
      result2.tactiles[i].electrodes[j] = result2.tactiles[i].electrodes[j] - result2.tactiles[i].tdc * cal[i][j];
    }
  }
  pub2.publish(result2);

}

int main(int argc, char **argv) {
  ros::init(argc, argv, "Temperature");
  ros::NodeHandle nh;


  ros::NodeHandle private_node_handle("~");

  XmlRpc::XmlRpcValue values;
  private_node_handle.getParam("temperature_calibration_values", values);

  for (int i=0; i<5; i++){
    for (int j=0; j<19; j++) {
      cal[i][j] = values[i][j];
    }
  }

  ros::Subscriber sub = nh.subscribe("/rh/tactile", 10, callback);
  pub = nh.advertise<std_msgs::Float64MultiArray>("/rh/temperature", 10);
  pub2 = nh.advertise<sr_robot_msgs::BiotacAll>("/rh/tactile_temperature", 10);

  ros::spin();
  return 0;
}

