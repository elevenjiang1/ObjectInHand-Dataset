#include <tams_biotac/visualize_electrodes.h>

#include <tams_biotac/contacts.h>

const unsigned int MAX_UPDATE_RATE_HZ{ 25 };

VisualizeElectrodes::VisualizeElectrodes() { initializeMarkerArray(); }

VisualizeElectrodes::VisualizeElectrodes(std::string input_topic) {
  subscriber_ = node_handle_.subscribe(input_topic, 1,
                                       &VisualizeElectrodes::callback, this);
  marker_publisher_ = node_handle_.advertise<visualization_msgs::MarkerArray>(
      "electrode_marker", 1);

  initializeMarkerArray();
}

void VisualizeElectrodes::visualize(const sr_robot_msgs::BiotacAll& msg) {
  float intensity = 0.0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 19; j++) {
      intensity = msg.tactiles[i].electrodes[j] * 0.01;

      if (intensity > 0.0) {
        if (intensity > 1.0)
          intensity = 1.0;
        marker_array_.markers[i * 19 + j].color.r = 1.0 - intensity;
        marker_array_.markers[i * 19 + j].color.g = 1.0 - intensity;
        marker_array_.markers[i * 19 + j].color.b = 1.0;
      } else {
        if (intensity < -1.0)
          intensity = -1.0;
        marker_array_.markers[i * 19 + j].color.r = 1.0;
        marker_array_.markers[i * 19 + j].color.g = 1.0 + intensity;
        marker_array_.markers[i * 19 + j].color.b = 1.0 + intensity;
      }
    }
  }
  marker_publisher_.publish(marker_array_);
}

void VisualizeElectrodes::callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg) {
  ros::WallTime now{ ros::WallTime::now() };
  if(now - last_publish_ > ros::WallDuration(1.0/MAX_UPDATE_RATE_HZ)){
    visualize(*msg);
    last_publish_ = now;
  }
}

void VisualizeElectrodes::initializeMarkerArray() {
  visualization_msgs::Marker electrode;

  electrode.type = visualization_msgs::Marker::SPHERE;
  electrode.color.r = 1.0;
  electrode.color.g = 1.0;
  electrode.color.b = 1.0;
  electrode.color.a = 1.0;
  electrode.scale.x = 0.0015;
  electrode.scale.y = 0.0015;
  electrode.scale.z = 0.0015;
  electrode.pose.orientation.w = 1.0;
  electrode.action = visualization_msgs::Marker::ADD;
  electrode.frame_locked = true;

  // distinguish the cylindrical part from the spherical part of the Biotac
  for (int i = 0; i < 5; i++) {
    electrode.ns = finger[i];
    electrode.header.frame_id = finger[i];
    for (int j = 0; j < 19; j++) {
      electrode.id = i * 19 + j;

      float r = 0.007; // radius of the Biotac surface in mm

      if (x_electrode[j] > 0.0) {
        electrode.pose.position.x =
            x_electrode[j] *
            (r / sqrt(pow(x_electrode[j], 2) + pow(y_electrode[j], 2) +
                      pow(z_electrode[j], 2)));
        electrode.pose.position.y =
            y_electrode[j] *
            (r / sqrt(pow(x_electrode[j], 2) + pow(y_electrode[j], 2) +
                      pow(z_electrode[j], 2)));
        electrode.pose.position.z =
            z_electrode[j] *
            (r / sqrt(pow(x_electrode[j], 2) + pow(y_electrode[j], 2) +
                      pow(z_electrode[j], 2)));
      } else {
        electrode.pose.position.x = 0.001 * x_electrode[j];
        electrode.pose.position.y =
            r * y_electrode[j] /
            (sqrt(pow(y_electrode[j], 2) + pow(z_electrode[j], 2)));
        electrode.pose.position.z =
            r * z_electrode[j] /
            (sqrt(pow(y_electrode[j], 2) + pow(z_electrode[j], 2)));
      }
      marker_array_.markers.push_back(electrode);
    }
  }
}
