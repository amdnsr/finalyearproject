import os
import shutil
import sys


def clearFolderContents(directory):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    except OSError:
        print("Error in deleting the contents of the directory: " + directory)


if __name__ == '__main__':
    no_of_arguments = len(sys.argv)
    if no_of_arguments == 3:
        command_name = sys.argv[1]
        if command_name == "clear":
            folder_name = sys.argv[2]
            clearFolderContents(folder_name)
