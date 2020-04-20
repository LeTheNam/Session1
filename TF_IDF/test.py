import os
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
f = open('./DS_LAB/Session_1/TF_IDF/stop_words.txt')
