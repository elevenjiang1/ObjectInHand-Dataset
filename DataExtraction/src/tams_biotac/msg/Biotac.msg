std_msgs/Header header
float64 pac0
float64 pac1
float64 pdc
float64 tac
float64 tdc
float64[19] electrodes


## TODO change this message:
## Raw BioTac data
# sr_robot_msgs/Biotac raw

## Processed Biotac data (compensated temperature, compensated pressure, filtered, biased)
## like above?
# float64 pdc
# float64 tdc
# float64[19] electrodes

## Temperature in degrees
# float64 temperature

## Contact information
# tams_biotac/Contact contact
