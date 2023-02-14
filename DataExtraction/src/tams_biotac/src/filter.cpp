#include <tams_biotac/filter.h>

Filter::Filter(float alpha) {
  alpha_ = alpha;
  filtered_msg_ = NULL;
}

Filter::Filter(std::string input_topic, std::string output_topic, float alpha) {
  subscriber_ = node_handle_.subscribe(input_topic, 1, &Filter::callback, this);
  publisher_ =
      node_handle_.advertise<sr_robot_msgs::BiotacAll>(output_topic, 1);

  alpha_ = alpha;
  filtered_msg_ = NULL;
}

Filter::~Filter() { delete filtered_msg_; }

void Filter::callback(const sr_robot_msgs::BiotacAll::ConstPtr &msg) {
  publisher_.publish(filter(*msg));
}

sr_robot_msgs::BiotacAll Filter::filter(const sr_robot_msgs::BiotacAll msg) {
  if (filtered_msg_ == NULL) {
    filtered_msg_ = new sr_robot_msgs::BiotacAll(msg);
    return *filtered_msg_;
  }

  filtered_msg_->header = msg.header;

  for (int i = 0; i < 5; i++) {
    filtered_msg_->tactiles[i].pac0 =
        eWMA(msg.tactiles[i].pac0, filtered_msg_->tactiles[i].pac0);
    filtered_msg_->tactiles[i].pac1 =
        eWMA(msg.tactiles[i].pac1, filtered_msg_->tactiles[i].pac1);
    filtered_msg_->tactiles[i].pdc =
        eWMA(msg.tactiles[i].pdc, filtered_msg_->tactiles[i].pdc);
    filtered_msg_->tactiles[i].tac =
        eWMA(msg.tactiles[i].tac, filtered_msg_->tactiles[i].tac);
    filtered_msg_->tactiles[i].tdc =
        eWMA(msg.tactiles[i].tdc, filtered_msg_->tactiles[i].tdc);
    for (int j = 0; j < 19; j++) {
      filtered_msg_->tactiles[i].electrodes[j] =
          eWMA(msg.tactiles[i].electrodes[j],
               filtered_msg_->tactiles[i].electrodes[j]);
    }
  }

  return *filtered_msg_;
}

short Filter::eWMA(short new_value, short old_value) {
  return (short)(alpha_ * new_value + (1 - alpha_) * old_value);
}
