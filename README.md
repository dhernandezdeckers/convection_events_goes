# convection_events_goes
Python codes to identify and analyze convective events based 
on GOES-13 or GOES-16 IR data.

# How to use
The code is divided in three parts. First, GOES data must be 
processed to create a gridded brightness temperature field
(read_GOES_data_runscript.py). Second, convective events are
identified in this common grid (find_convective_events_runscript.py),
and third, these events are processed and analyzed
(process_convective_events.py).

# Acknowledgements
The use of this code should acknowledge the following paper,
where further details of this methodology can be found:

Hernandez-Deckers D., 2022: Features of atmospheric deep 
convection in Northwestern South America obtained from 
infrared satellite data, Quart. J. Roy. Meteor. Soc., 148(742), 
338-350. https://doi.org/10.1002/qj.4208
