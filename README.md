# tissue_image_analysis
Created 04/02/2023 by Natasha Cowley.

## Setup and Running:

For this itteration of the code we need traces with spider legs and dots (nuclei).

Cell traces and 'nuclei' files should be placed in the input file. For each trace a 
configuration file should be filled out. This is a .csv file with the following fields:

Exp_date: The date the experiment was run in yyyymmdd format
Exp_ID: an identifier of your chosing for the experient, makesure this begins with the date in yyyymmdd format. sugest including time point and stretch type.
Trace_Type: 0=Manual, 1=Cellpose (this is here for future development)
Nuclei_Exist: 0= no, 1=yes, (also here for future code development, currently nuclei are necessary)
Edges_Name: the name of the trace file without extension.
Nuclei_Name: the name of the nuclei file without extension.
Stretch_Type: 0=unstretched, 1= fast stretch, 2= incremental stretch
t_sec: Timepoint in experiement in minutes.
Pixel_Size: pixel size from original image (find in Fiji, likely to be 1024 or 512)
Micron_Size: micron size from original image (find in Fiji)


The fields are case sensitive so do not change the names of the columns.

An example config file can be found in the Input folder.

If you are running this for a stretched sample which is not at 0s time, you will need to make sure 
that there is a pref area file calculated from the t=0s trace in the input directory. The name of
which should start with the date of the experiment. If you have experiments from the same date, be
sure to only have the pref_area file from the desired experiment in the Input folder.

## Partially running code

If you have the trace data files:
  - UniqueTissueEdges.npy
  - uniqueTissueVerts.npy
  - centroids.xlsx
  - edgesOnCells.xlsx

You can run matrix_generation.py to generate the cell matrices. 
You will need to supply the path to the files in the user input section at the top of the script. 

If you have the matrix files:
  - Matrix_A.txt
  - Matrix_B.txt
  - Matrix_C.txt
  - Matrix_R.txt

You can run data_generation.py to generate data and plots.
You will need to supply the path to the files in the user input section at the top of the script. 
