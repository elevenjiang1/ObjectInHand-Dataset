#include <tams_biotac/normalization.h>

Normalization::Normalization() {
  subscriber_ = node_handle_.subscribe("/reset_biotac_normalization", 1, &Normalization::resetCallback, this);
  init();
}

Normalization::Normalization(std::string input_topic,
                             std::string output_topic) {
  subscriber_ =
      node_handle_.subscribe(input_topic, 1, &Normalization::callback, this);
  publisher_ =
      node_handle_.advertise<sr_robot_msgs::BiotacAll>(output_topic, 1);
  init();
}

void Normalization::init() {
  ros::NodeHandle pnh("~");
  pnh.param<bool>("periodic_normalization", periodic_normalization_, "true");

  service_ = node_handle_.advertiseService(
      "reset_biotac_normalization", &Normalization::serviceCallback, this);

  for (int i=0; i<5; i++) {
    skip_count_[i] = 0;
    initialized_[i] = false;
  }
}

void Normalization::callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg) {
  publisher_.publish(normalize(*msg));
}

void Normalization::resetCallback(const std_msgs::Empty::ConstPtr &msg) {
  resetOffset();
}

bool Normalization::serviceCallback(std_srvs::Empty::Request &request,
                                    std_srvs::Empty::Response &response) {
  resetOffset();
  return true;
}

sr_robot_msgs::BiotacAll
Normalization::normalize(sr_robot_msgs::BiotacAll msg) {
  sr_robot_msgs::BiotacAll result;
  result.header = msg.header;
  for (int i = 0; i < 5; i++)
    result.tactiles[i] = normalize(msg.tactiles[i], i);

  return result;
}

sr_robot_msgs::Biotac Normalization::normalize(sr_robot_msgs::Biotac msg,
                                               int i) {
  return tactileToSr(normalize(srToTactile(msg), i));
}

tactiles::BiotacData Normalization::normalize(tactiles::BiotacData msg, int i) {
  if (!initialized_[i]) {
    offset_[i] = msg;
    initialized_[i] = true;
  }

  tactiles::BiotacData normalized_msg = substractOffset(msg, offset_[i]);

  skip_count_[i]++;
  if (periodic_normalization_ && skip_count_[i] >= SKIP) {
    skip_count_[i] = 0;
    // Create history of 5 pdc values over a time of 2,5 seconds
    pdc_history_[i].push_back(normalized_msg.pdc);
    if (pdc_history_[i].size() > 5) {
      pdc_history_[i].pop_front();

      float pdc_sum = 0;
      for (std::deque<int>::iterator it = pdc_history_[i].begin();
           it != pdc_history_[i].end(); ++it) {
        pdc_sum += *it;
      }
      // If the average pdc minus the old offset is lower then the threshold,
      // determine the finger could be in a rest state. Check electrodes to
      // validate.
      if (((pdc_sum / pdc_history_[i].size())) < PDC_TH) {
        short min, max;
        min = max = normalized_msg.electrodes[0];

        for (int j = 1; j < 19; j++) {
          min = std::min(min, normalized_msg.electrodes[j]);
          max = std::max(max, normalized_msg.electrodes[j]);
        }

        // Reset offset if rest state is detected
        if ((abs(min - max)) < ELEC_TH) {
          offset_[i] = msg;
          pdc_history_[i].clear();
        }
      }
    }
  }
  return normalized_msg;
}

void Normalization::resetOffset() {
  for (int i=0; i<5; i++) {
    skip_count_[i] = 0;
    initialized_[i] = false;
  }
}

tactiles::BiotacData
Normalization::substractOffset(tactiles::BiotacData msg,
                               tactiles::BiotacData offset) {
  msg.pac0 -= offset.pac0;
  msg.pac1 -= offset.pac1;
  msg.pdc -= offset.pdc;
  for (int j = 0; j < 19; j++)
    msg.electrodes[j] -= offset.electrodes[j];
  return msg;
}

tactiles::BiotacData Normalization::srToTactile(sr_robot_msgs::Biotac msg) {
  tactiles::BiotacData result;
  result.pac0 = msg.pac0;
  result.pac1 = msg.pac1;
  result.pdc = msg.pdc;
  result.tac = msg.tac;
  result.tdc = msg.tdc;
  for (int i = 0; i < 19; i++)
    result.electrodes.push_back(msg.electrodes[i]);
  return result;
}

sr_robot_msgs::Biotac Normalization::tactileToSr(tactiles::BiotacData msg) {
  sr_robot_msgs::Biotac result;
  result.pac0 = msg.pac0;
  result.pac1 = msg.pac1;
  result.pdc = msg.pdc;
  result.tac = msg.tac;
  result.tdc = msg.tdc;
  result.electrodes.resize(19);
  for (int i = 0; i < 19; i++)
    result.electrodes[i] = msg.electrodes[i];
  return result;
}
