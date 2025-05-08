# tissue_image_analysis
Created 04/02/2023 by Natasha Cowley.
Updated 14/02/2025

## Setup and Running:

For this itteration of the code we need traces with spider legs (vertices at edge of trace need 3 edges coming out of them to be picked up by the algorithm). Traces should also be vertically inverted.

Files should be named according to the established naming convention e.g.

'20231019_1_IP_GFPCAAX-CheHis_uu_0p5_SP_fr001_trace'

1. Experiment date
2. Experiement number on that day
3. Initials
4. What cells are injected or treated with seperated by a dash
5. stretch axis (u,b) stretch type(u,f,s,c) 
6. stretch amount
7. projection type
8. frame frome tif
9. trace



Cell traces should be placed in the input file. For each experiment a configuration file should be filled out, with different timepoints on seperate lines. This is a .csv file with the following fields:


 - **Edges_Name**: the name of the trace file without extension.
 - **t_min**: Timepoint in experiement in minutes.
 - **Pixel_Size**: pixel size from original image (find in Fiji or log, likely to be 1024 or 512)
 - **Micron_Size**: micron size from original image (find in Fiji or log)

If running a flipper trace add and addional column for the lifetime image (this should aslo be stored in the input directory, and invereted so it has the same orientation as the trace file)

 - **Flipper_file**: the name of the masked lifetimefile without extension

You will also need to set flipper==True in the input directory

The fields are case sensitive so do not change the names of the columns.

An example config file can be found in the Input folder.

If you are running this for a stretched sample which is not at 0s time, you will need to make sure 
that there is a pref area file calculated from the t=0s trace in the input directory. The name of
which should start with the date of the experiment. If you have experiments from the same date, be
sure to only have the pref_area file from the desired experiment in the Input folder.

