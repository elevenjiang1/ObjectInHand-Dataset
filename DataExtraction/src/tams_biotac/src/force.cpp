#include <tams_biotac/force.h>

#include <fstream>
#include <yaml-cpp/yaml.h>

Force::Force() { loadCalibrationValues(); }

Force::Force(std::string input_topic) {
  contact_subscriber_ =
      node_handle_.subscribe(input_topic, 1, &Force::contactCallback, this);
  tactile_subscriber_ = node_handle_.subscribe("/rh/tactile_normalized", 1,
                                               &Force::tactileCallback, this);

  publisher_ =
      node_handle_.advertise<tams_biotac::Contact>("/rh/force_contact", 1);

  loadCalibrationValues();
}

void Force::loadCalibrationValues() {
  std::ifstream f("biotac_info/force_calibration.yaml");
  if (!f.good()) {
    ROS_WARN("No calibration file found");
    for (int i = 0; i < 19; i++)
      calibration_values_[i] = 1;
    return;
  }

  YAML::Node node = YAML::LoadFile("biotac_info/force_calibration.yaml");

  for (int i = 0; i < 19; i++)
    calibration_values_[i] = node["pdc_calibration"][i].as<float>();
}

float Force::getForce(int pdc, float x, float y, float z) {
  std::vector<std::pair<float, int> > distance;

  for (int i = 0; i < 19; i++) {
    float dist =
        hypot(hypot(x - x_surface_electrode[i], y - y_surface_electrode[i]),
              z - z_surface_electrode[i]);
    distance.push_back(std::make_pair(dist, i));
  }

  std::sort(distance.begin(), distance.end());

  float sum_distance =
      distance[0].first + distance[1].first + distance[2].first;
  float div1 = sum_distance / distance[0].first;
  float div2 = sum_distance / distance[1].first;
  float div3 = sum_distance / distance[2].first;

  float sum2 = div1 + div2 + div3;

  float imp_1 = div1 / sum2;
  float imp_2 = div2 / sum2;
  float imp_3 = div3 / sum2;

  float force = (imp_1 * calibration_values_[distance[0].second] * pdc) +
                (imp_2 * calibration_values_[distance[1].second] * pdc) +
                (imp_3 * calibration_values_[distance[2].second] * pdc);

  if (force < 0)
    return 0;
  return force * -1;
}

void Force::contactCallback(const tams_biotac::Contact::ConstPtr &msg) {
  tams_biotac::Contact contact = *msg;
  contact.wrench.force.x = getForce(current_pdc_, contact.position.x, contact.position.y,
                           contact.position.z);
  publisher_.publish(contact);
}

void Force::tactileCallback(const sr_robot_msgs::BiotacAll::ConstPtr &msg) {
  current_pdc_ = msg->tactiles[0].pdc;
}
