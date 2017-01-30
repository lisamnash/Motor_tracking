import os

def dump_pickled_data(output_dir, filename, data):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    of = open(output_dir + '/' + filename + '.pickle', 'wb')
    pickle.dump(data, of, pickle.HIGHEST_PROTOCOL)
    of.close()


def find_files_by_extension(root_dir, ext, tot=False):
    filenames = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                if tot == False:
                    filenames.append(file)
                else:
                    filenames.append(root + '/' + file)
    return filenames
