#include <tams_biotac/subtract_temperature.h>
#include <tams_biotac/filter.h>
#include <tams_biotac/normalization.h>
#include <tams_biotac/visualize_electrodes.h>
#include <tams_biotac/contacts.h>
//#include <tams_biotac/force.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "TamsBiotac");

  SubtractTemperature subtract_temperature("rh/tactile", "rh/tactile_subtract_temperature");
  Filter filter("rh/tactile_subtract_temperature", "rh/tactile_filtered", 0.5);
  Normalization normalization("rh/tactile_filtered", "rh/tactile_normalized");
  VisualizeElectrodes visualize_electrodes("rh/tactile_normalized");
  Contacts contacts("rh/tactile_normalized", "rh/contacts");
//  Force force("ft_contact");

  ros::spin();
  return 0;
}
