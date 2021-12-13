def read_labels_from_file(filename):
    with open(filename,'rb') as f: #use gzip to open the file in read binary mode
        magic = f.read(4) # magic number is the first 4 bytes

        # the same as above but with labels
        nolab = f.read(4)
        nolab = int.from_bytes(nolab,'big')
        print("Num of labels is:", nolab)
        # for looping through labels
        labels = [f.read(1) for i in range(nolab)]
        labels = [int.from_bytes(label, 'big') for label in labels]
    return labels

def read_images_from_file(filename):
    with open(filename,'rb') as f:
        magic = f.read(4)

        # Number of images in next 4 bytes
        noimg = f.read(4)
        noimg = int.from_bytes(noimg,'big')
        print("Number of images is:", noimg)

        # Number of rows in next 4 bytes
        norow = f.read(4)
        norow = int.from_bytes(norow,'big')
        print("Number of rows is:", norow)
        
        # Number of columns in next 4 bytes
        nocol = f.read(4)
        nocol = int.from_bytes(nocol,'big')
        print("Number of cols is:", nocol)

        images = [] # create array
        #for loop
        for i in range(noimg):
            rows = []
            for r in range(norow):
                cols = []
                for c in range(nocol):
                    cols.append(int.from_bytes(f.read(1), 'big')) # append the current byte for every column
                rows.append(cols) # append columns array for every row
            images.append(rows) # append rows for every image
    return images