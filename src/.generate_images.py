import os
import shutil

dir_input = "ttf/"
dir_output = "img/"

# Define your font name here
# font_filename = "Impact.ttf"
# font_filename = "Arial Unicode.ttf"
# font_filename = "UnGungseo.ttf"
font_filename = "Songti.ttc"

F = fontforge.open(dir_input + font_filename)

# if not os.path.exists(dir_output):
# 	os.mkdir(dir_output)

# dir_this_font = dir_output + os.path.splitext(font_filename)[0]

# if os.path.exists(dir_this_font):
# 	shutil.rmtree(dir_this_font)

# os.mkdir(dir_this_font)

# for name in F:
#     filename = name + ".bmp"
#     F[name].export(dir_this_font + "/" + filename)
