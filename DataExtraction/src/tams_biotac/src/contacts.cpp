#include <tams_biotac/contacts.h>

#include <visualization_msgs/MarkerArray.h>

Contacts::Contacts() {}

Contacts::Contacts(std::string input_topic, std::string output_topic) {
  if (input_topic == "rh/tactile_normalized") {
    cal_ = false;
    subscriber_ =
        node_handle_.subscribe(input_topic, 1, &Contacts::srCallback, this);
    publisher_ =
        node_handle_.advertise<tams_biotac::ContactArray>(output_topic, 1);
    marker_publisher_ =
        node_handle_.advertise<visualization_msgs::MarkerArray>("contacts_marker", 10);
  } else {
    cal_ = true;
    subscriber_ =
        node_handle_.subscribe(input_topic, 1, &Contacts::tamsCallback, this);
    publisher_ =
        node_handle_.advertise<tams_biotac::ContactArray>(output_topic, 1);
    marker_publisher_ =
        node_handle_.advertise<visualization_msgs::MarkerArray>("calibrated_contacts_marker", 10);
  }
}

tams_biotac::ContactArray Contacts::getContacts(sr_robot_msgs::BiotacAll msg) {
  tams_biotac::ContactArray result;

  for (int i = 0; i < msg.tactiles.size(); i++) {
    if ((msg.tactiles[i].pdc) > 10) {
      tams_biotac::Contact contact = getContact(msg.tactiles[i]);
      // if force < 0, no electrode with negative impedance exist
      //if (contact.force < 0)
      //  continue;
      contact.header = msg.header;
      contact.header.frame_id = finger[i];
      mapPointToSurface(contact.position);
      result.contacts.push_back(contact);
    }
  }
  return current_contacts_ = result;
}

tams_biotac::ContactArray Contacts::getContacts(tams_biotac::BiotacAll msg) {
  tams_biotac::ContactArray result;

  for (int i = 0; i < msg.tactiles.size(); i++) {
    if ((msg.tactiles[i].pdc) > 10) {
      tams_biotac::Contact contact = getContact(msg.tactiles[i]);
      // if force < 0, no electrode with negative impedance exist
      if (contact.wrench.force.x < 0)
        continue;
      contact.header = msg.tactiles[i].header;
      mapPointToSurface(contact.position);
      result.contacts.push_back(contact);
    }
  }
  return current_contacts_ = result;
}

tams_biotac::Contact Contacts::getContact(sr_robot_msgs::Biotac msg) {
  tams_biotac::Biotac result;
  result.pac0 = msg.pac0;
  result.pac1 = msg.pac1;
  result.pdc = msg.pdc;
  result.tac = msg.tac;
  result.tdc = msg.tdc;
  for (int i = 0; i < 19; i++)
    result.electrodes[i] = msg.electrodes[i];

  return getContact(result);
}

tams_biotac::Contact Contacts::getContact(tams_biotac::Biotac msg) {
  tams_biotac::Contact result;
  geometry_msgs::Point num;
  float den = 0;
  result.wrench.force.x = 0;

  for (int i = 0; i < 19; i++) {
    if (msg.electrodes[i] < 0) {
      int n = 1;
      if (!cal_)
        n = N_EXP;
      num.x += (pow(fabs(msg.electrodes[i]), n) * x_electrode[i]);
      num.y += (pow(fabs(msg.electrodes[i]), n) * y_electrode[i]);
      num.z += (pow(fabs(msg.electrodes[i]), n) * z_electrode[i]);
      den += pow(fabs(msg.electrodes[i]), n);
    }
  }

  // if den is not positive no contact point could be computed
  if (den > 0){
    // in meter
    result.position.x = (num.x / den) * 0.001;
    result.position.y = (num.y / den) * 0.001;
    result.position.z = (num.z / den) * 0.001;
    result.wrench.force.x = force_.getForce(msg.pdc, result.position.x, result.position.y, result.position.z);
  }

  return result;
}

void Contacts::mapPointToSurface(geometry_msgs::Point &point) {
  // otherwise we divide through 0
  if (point.x == 0 && point.y == 0 && point.z == 0)
    return;

  // distinguish the cylindrical part from the spherical part of the Biotac
  geometry_msgs::Point point_surface;
  float r = 0.007; // radius of the Biotac surface

  if (point.x > 0.0) {
    point_surface.x =
        (point.x * r) /
        (sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2)));
    point_surface.y =
        (point.y * r) /
        (sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2)));
    point_surface.z =
        (point.z * r) /
        (sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2)));
  } else {
    point_surface.x = point.x;
    point_surface.y = (r * point.y) / (sqrt(pow(point.y, 2) + pow(point.z, 2)));
    point_surface.z = (r * point.z) / (sqrt(pow(point.y, 2) + pow(point.z, 2)));
  }
  point = point_surface;
}

void Contacts::srCallback(const sr_robot_msgs::BiotacAll::ConstPtr &msg) {
  tams_biotac::ContactArray result = getContacts(*msg);
  if (result.contacts.size() > 0) {
    publisher_.publish(result);
    visualizeContacts();
  }
}

void Contacts::tamsCallback(const tams_biotac::BiotacAll::ConstPtr &msg) {
  publisher_.publish(getContacts(*msg));
  visualizeContacts();
}

void Contacts::visualizeContacts() {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker sphere;

  sphere.type = visualization_msgs::Marker::SPHERE;
  if (cal_)
    sphere.color.b = 1.0f;
  else
    sphere.color.g = 1.0f;
  sphere.color.a = 1.0;
  sphere.pose.orientation.w = 1.0;
  sphere.action = visualization_msgs::Marker::ADD;
  sphere.lifetime = ros::Duration(0.1);

  for (int i = 0; i < current_contacts_.contacts.size(); i++) {
    sphere.ns = current_contacts_.contacts[i].header.frame_id;
    sphere.header = current_contacts_.contacts[i].header;
    sphere.scale.x = 0.002; // + 0.000004 * current_contacts_.contacts[i].force;
    sphere.scale.y = 0.002; // + 0.000004 * current_contacts_.contacts[i].force;
    sphere.scale.z = 0.002; // + 0.000004 * current_contacts_.contacts[i].force;
    sphere.pose.position.x = current_contacts_.contacts[i].position.x;
    sphere.pose.position.y = current_contacts_.contacts[i].position.y;
    sphere.pose.position.z = current_contacts_.contacts[i].position.z;
    marker_array.markers.push_back(sphere);
  }
  marker_publisher_.publish(marker_array);
}
