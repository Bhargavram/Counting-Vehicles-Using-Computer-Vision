# Counting-Vehicles-Using-Computer-Vision
Application to count number of vehicles in a given video of traffic using computer vision.

• Background subtraction: Subtracting the background layer (static layer without any moving objects) from the current frame. MOG algorithm will be used for background subtraction. 
• Filtering the images: Noise from the resultant image will be removed by filtering It. The filters are used: Threshold, Erode, Dilate, Opening, Closing. 
• Object detection by contours: Some filtering by height, width and add centroid.
• Building processing pipeline 
