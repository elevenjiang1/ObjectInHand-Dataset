#include <tams_biotac/contacts.h>
#include <tams_biotac/tactile_calibrated.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "TamsBiotacCalibrated");

  TactileCalibrated tactile_calibrated("rh/tactile_normalized",
                                       "rh/tactile_calibrated");
  Contacts calibrated_contacts("rh/tactile_calibrated",
                               "rh/calibrated_contacts");
  ros::spin();
  return 0;
}

