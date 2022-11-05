

# Resize all images in a directory to half the size.
#
# Save on a new file with the same name but with "small_" prefix
# on high quality jpeg format.
#
# If the script is in /images/ and the files are in /images/2012-1-1-pics
# call with: python resize.py 2012-1-1-pics

import image
import os
import sys

directory = "Img"

for file_name in os.listdir(directory):
	if file_name.endswith(".png"):
		print("Processing %s" % file_name)
		image = image.open(os.path.join(directory, file_name))

		x,y = image.size
		new_dimensions = (64,64) #dimension set here
		output = image.resize(new_dimensions, image.ANTIALIAS)

		output_file_name = os.path.join(directory, "small_" + file_name)
		output.save(output_file_name, "JPEG", quality = 95)

print("All done")
