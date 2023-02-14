# tams_biotac

-----


This package is used to preprocess the raw BioTac sensor data.

The data run through different stages:
* Subtract temperature: The temperature dependency of the electrodes is compensated
* Filter: Noise filtering
* Normalization: pdc, pac and electrode values will be shifted to 0 in an idle state

-----

__Usage__

To start the pipeline the following launch file can be used:  
```roslaunch tams_biotac biotac.launch```

The started node will describe to the /rh/tactile topic.