# tissue_image_analysis
Created 04/02/2023 by Natasha Cowley.

This is the cleanned up analysis code currently only for manual traces. Development of the image analysis algorithm to work with cellpose traces is ongoing.

## Setup and Running:

For this itteration of the code we need traces with spider legs and dots (nuclei).

Cell traces and 'nuclei' files should be placed in the input file. For each trace a configuration file should be filled out. This is a .csv file with the following fields:


 - **Edges_Name**: the name of the trace file without extension.
 - **t_min**: Timepoint in experiement in minutes.
 - **Pixel_Size**: pixel size from original image (find in Fiji or log, likely to be 1024 or 512)
 - **Micron_Size**: micron size from original image (find in Fiji or log)


The fields are case sensitive so do not change the names of the columns.

An example config file can be found in the Input folder.

If you are running this for a stretched sample which is not at 0s time, you will need to make sure 
that there is a pref area file calculated from the t=0s trace in the input directory. The name of
which should start with the date of the experiment. If you have experiments from the same date, be
sure to only have the pref_area file from the desired experiment in the Input folder.

<!-- ## Partially running code

If you have the trace data files:
  - UniqueTissueEdges.npy
  - uniqueTissueVerts.npy
  - centroids.xlsx
  - edgesOnCells.xlsx

You can run _matrix_generation.py_ to generate the cell matrices. 
You will need to supply the path to the files in the user input section at the top of the script. 

If you have the matrix files:
  - Matrix_A.txt
  - Matrix_B.txt
  - Matrix_C.txt
  - Matrix_R.txt

You can run _data_generation.py_ to generate data and plots.
You will need to supply the path to the files in the user input section at the top of the script.  -->
