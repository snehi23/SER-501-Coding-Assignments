import numpy
from pylab import *
from skimage import img_as_float


# DUAL GRADIENT ENERGY CALCULATION OF AN IMAGE
def dual_gradient_energy(img):
    img = img_as_float(img)
    #  r = W, c = H
    r, c = img.shape[:2]
    gradient_matrix = numpy.zeros((r, c))

    for y in range(0, c):
        for x in range(0, r):
            if x == 0 and y == 0:
                rx = img[x + 1, y, 0] - img[r - 1, y, 0]
                gx = img[x + 1, y, 1] - img[r - 1, y, 1]
                bx = img[x + 1, y, 2] - img[r - 1, y, 2]

                ry = img[x, c - 1, 0] - img[x, y + 1, 0]
                gy = img[x, c - 1, 1] - img[x, y + 1, 1]
                by = img[x, c - 1, 2] - img[x, y + 1, 2]

            elif x == r - 1 and y == c - 1:
                rx = img[x - 1, y, 0] - img[0, y, 0]
                gx = img[x - 1, y, 1] - img[0, y, 1]
                bx = img[x - 1, y, 2] - img[0, y, 2]

                ry = img[x, 0, 0] - img[x, y - 1, 0]
                gy = img[x, 0, 1] - img[x, y - 1, 1]
                by = img[x, 0, 2] - img[x, y - 1, 2]

            elif x == 0 and y == c - 1:
                rx = img[x + 1, y, 0] - img[r - 1, y, 0]
                gx = img[x + 1, y, 1] - img[r - 1, y, 1]
                bx = img[x + 1, y, 2] - img[r - 1, y, 2]

                ry = img[x, 0, 0] - img[x, y - 1, 0]
                gy = img[x, 0, 1] - img[x, y - 1, 1]
                by = img[x, 0, 2] - img[x, y - 1, 2]

            elif x == r - 1 and y == 0:
                rx = img[x - 1, y, 0] - img[0, y, 0]
                gx = img[x - 1, y, 1] - img[0, y, 1]
                bx = img[x - 1, y, 2] - img[0, y, 2]

                ry = img[x, c - 1, 0] - img[x, y + 1, 0]
                gy = img[x, c - 1, 1] - img[x, y + 1, 1]
                by = img[x, c - 1, 2] - img[x, y + 1, 2]

            elif x == 0:
                rx = img[r - 1, y, 0] - img[x + 1, y, 0]
                gx = img[r - 1, y, 1] - img[x + 1, y, 1]
                bx = img[r - 1, y, 2] - img[x + 1, y, 2]

                ry = img[x, y - 1, 0] - img[x, y + 1, 0]
                gy = img[x, y - 1, 1] - img[x, y + 1, 1]
                by = img[x, y - 1, 2] - img[x, y + 1, 2]

            elif y == 0:
                rx = img[x - 1, y, 0] - img[x + 1, y, 0]
                gx = img[x - 1, y, 1] - img[x + 1, y, 1]
                bx = img[x - 1, y, 2] - img[x + 1, y, 2]

                ry = img[x, c - 1, 0] - img[x, y + 1, 0]
                gy = img[x, c - 1, 1] - img[x, y + 1, 1]
                by = img[x, c - 1, 2] - img[x, y + 1, 2]

            elif x == r - 1:
                rx = img[x - 1, y, 0] - img[0, y, 0]
                gx = img[x - 1, y, 1] - img[0, y, 1]
                bx = img[x - 1, y, 2] - img[0, y, 2]

                ry = img[x, y - 1, 0] - img[x, y + 1, 0]
                gy = img[x, y - 1, 1] - img[x, y + 1, 1]
                by = img[x, y - 1, 2] - img[x, y + 1, 2]

            elif y == c - 1:
                rx = img[x - 1, y, 0] - img[x + 1, y, 0]
                gx = img[x - 1, y, 1] - img[x + 1, y, 1]
                bx = img[x - 1, y, 2] - img[x + 1, y, 2]

                ry = img[x, 0, 0] - img[x, y - 1, 0]
                gy = img[x, 0, 1] - img[x, y - 1, 1]
                by = img[x, 0, 2] - img[x, y - 1, 2]
            else:
                rx = img[x - 1, y, 0] - img[x + 1, y, 0]
                gx = img[x - 1, y, 1] - img[x + 1, y, 1]
                bx = img[x - 1, y, 2] - img[x + 1, y, 2]

                ry = img[x, y - 1, 0] - img[x, y + 1, 0]
                gy = img[x, y - 1, 1] - img[x, y + 1, 1]
                by = img[x, y - 1, 2] - img[x, y + 1, 2]
            delta_x = rx**2 + gx**2 + bx**2
            delta_y = ry**2 + gy**2 + by**2
            gradient_matrix[x, y] = delta_x + delta_y

    return gradient_matrix


# HORIZONTAL SEAM PATH
def find_seam(img):
    img = img_as_float(img)
    r, c = img.shape[:2]
    gradient_matrix = numpy.zeros((r, c))
    seam_matrix = numpy.zeros((r, c))

    gradient_matrix = dual_gradient_energy(img)

    # INITIALIZE TOP ROW AS GRADIENT VALUE
    for y in range(0, c - 1):
        seam_matrix[0, y] = gradient_matrix[0, y]

    # COST MATRIX OF ENERGY OF IMAGE
    for y in range(0, c):
        for x in range(1, r):

            if y == 0:
                seam_matrix[x, y] = gradient_matrix[x, y] + min(seam_matrix[x - 1, y], seam_matrix[x - 1, y + 1])
            elif y == c - 1:
                seam_matrix[x, y] = gradient_matrix[x, y] + min(seam_matrix[x - 1, y], seam_matrix[x - 1, y - 1])
            else:
                seam_matrix[x, y] = gradient_matrix[x, y] + min(seam_matrix[x - 1, y],
                                                                seam_matrix[x - 1, y - 1], seam_matrix[x - 1, y + 1])

    last_row = [0] * c
    seam_path = []

    # SEPARATE OUT LAST ROW OF COST MATRIX
    for y in range(0, c):
        last_row[y] = seam_matrix[r - 1, y]

    # CALCULATE POSITION OF MIN VALUE OF BOTTOM OF COST MATRIX
    seam_position = (r - 1, last_row.index(min(last_row)))

    seam_path.append(seam_position[1])

    # BACKTRACK TO RE-CREATE SEAM PATH
    for b in range(1, r):
        x, y = seam_position
        if y == 0:
            p = seam_matrix[x - 1, y]
            q = seam_matrix[x - 1, y + 1]
            if p < q:
                seam_position = (x - 1, y)
            else:
                seam_position = (x - 1, y + 1)
        elif y == c - 1:
            p = seam_matrix[x - 1, y]
            q = seam_matrix[x - 1, y - 1]
            if p < q:
                seam_position = (x - 1, y)
            else:
                seam_position = (x - 1, y - 1)
        else:
            p = seam_matrix[x - 1, y]
            q = seam_matrix[x - 1, y - 1]
            r = seam_matrix[x - 1, y + 1]

            if p < q and p < r:
                seam_position = (x - 1, y)

            elif q < r and q < p:
                seam_position = (x - 1, y - 1)

            elif r < p and r < q:
                seam_position = (x - 1, y + 1)

            else:
                seam_position = (x - 1, y + 1)
        seam_path.append(seam_position[1])

    return seam_path


# MARK SEAM PIXELS TO RED TO VISUALISE ON IMAGE
def plot_seam(img, seam):
    r, c = img.shape[:2]

    for x in range(0, r):
        img[x, seam[x]] = [1.0, 0.0, 0.0]

    figure()
    gray()
    imshow(img)
    title("SEAM PLOT ON IMAGE")
    show()


# REMOVAL OF VERTICAL SEAM FROM IMAGE
def remove_seam(img, seam):
    r, c = img.shape[:2]
    present = False
    new_image = numpy.zeros((r, c - 1, 3))

    for x in range(0, r):
        for y in range(0, c):
                if y != seam[x] and present is False:
                    new_image[x, y] = img[x, y]
                elif y == seam[x]:
                    present = True
                else:
                    new_image[x, y - 1] = img[x, y]
        present = False
    return new_image


# VIEW IMAGE
def show_image(img):
    figure()
    gray()
    imshow(img)
    show()


# ORIGINAL VS COMPRESSED IMAGE VISUALISATION
def show_compression(org_img, seam_count):

    org_r, org_c = org_img.shape[:2]
    img = org_img.copy()
    all_seam_plot = org_img.copy()
    all_seams = []
    for i in range(0, seam_count):
        seam_path = find_seam(img)
        img = remove_seam(img, seam_path)
        compression_rate = round(i * 100.0 / org_c, 2)
        print str(compression_rate)+"% Compressed "
        all_seams.append(seam_path)

    for s in all_seams:
        for x in range(0, org_r):
            all_seam_plot[x, s[x]] = [1.0, 0.0, 0.0]

    figure()
    gray()
    subplot(2, 2, 1)
    imshow(all_seam_plot)
    title("ALL " + str(seam_count) + " SEAMS PLOT")
    subplot(2, 2, 3)
    imshow(org_img)
    title("ORIGINAL")
    subplot(2, 2, 4)
    imshow(img)
    title(str(compression_rate) + "% COMPRESSED")
    show()


def main():

    image_name = "TEST2.png"
    seam_count = 150

    # ENERGY GRADIENT OF IMAGE (W BY H MATRIX)
    org_img = imread(image_name)
    gradient_matrix = dual_gradient_energy(org_img)
    show_image(gradient_matrix)

    # FIRST VERTICAL SEAM IN IMAGE (AN ARRAY OF H INTEGERS)
    org_img = imread(image_name)
    seam_path = find_seam(org_img)

    # PLOT FIRST VERTICAL SEAM IN IMAGE
    org_img = imread(image_name)
    seam_path = find_seam(org_img)
    plot_seam(org_img, seam_path)

    # REMOVE FIRST VERTICAL SEAM IN IMAGE
    org_img = imread(image_name)
    seam_path = find_seam(org_img)
    new_image = remove_seam(org_img, seam_path)

    # ORIGINAL IMAGE VS COMPRESSED IMAGE BY SEAM COUNT
    org_img = imread(image_name)
    show_compression(org_img, seam_count)

if __name__ == '__main__':
    main()