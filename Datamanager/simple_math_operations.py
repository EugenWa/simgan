import numpy as np

# faktor for line 2 is returned
def intersect_lines(line1, line2):

    A = np.array([[line1[1][0], -line2[1][0]], [line1[1][1], -line2[1][1]]])
    if np.linalg.det(A) < 0.0 or np.linalg.det(A) > 0.0:
        B = np.array([line2[0][0]-line1[0][0], line2[0][1]-line1[0][1]])
        x = np.dot(np.linalg.inv(A), B)
        return x
    else:
        A = np.array([[line1[1][0], -line2[1][0]], [line1[1][2], -line2[1][2]]])
        if np.linalg.det(A) < 0 or np.linalg.det(A) > 0:
            B = np.array([line2[0][0] - line1[0][0], line2[0][2] - line1[0][2]])
            x = np.dot(np.linalg.inv(A), B)
            return x
        else:
            A = np.array([[line1[1][1], -line2[1][1]], [line1[1][2], -line2[1][2]]])
            if np.linalg.det(A) < 0 or np.linalg.det(A) > 0:
                B = np.array([line2[0][1] - line1[0][1], line2[0][2] - line1[0][2]])
                x = np.dot(np.linalg.inv(A), B)
                return x

    if line1[1][1] is not 0:
        ratio = line1[1][0]/line1[1][1]
        bias = line1[0][0]-(line2[0][0]*ratio)
        r2 = line2[1][0]-(line2[1][1]*ratio)
        if r2 is not 0:
            return bias/r2
    else:
        return 0

def intersect_plane_linelambda(gerade, ebene):  # ebene besteht aus d und normiertem normalenvektor
    un = np.dot(gerade[2], ebene[1])
    print("UN: ", un)
    gn = np.dot(gerade[0], ebene[1])
    if un == 0:
        if ebene[0] - gn is 0:
            return "all"
        else:
            return None
    else:
        return (ebene[0]-gn)/un

def intersect_plane_linepoint(gerade, ebene):  # ebene besteht aus d und normiertem normalenvektor
    un = np.dot(gerade[2], ebene[1])
    gn = np.dot(gerade[0], ebene[1])
    if un == 0:
        if ebene[0] - gn is 0:
            return "all"
        else:
            return None
    else:
        return ((ebene[0] - gn)/un)*gerade[2] + gerade[0]


# --- For Data ---
def flip_labels(ylabels):
    flipped_labels = np.zeros(ylabels.shape)
    for i in range(ylabels.shape[0]):
        if ylabels[i] < 1:
            flipped_labels[i] = 1
        else:
            flipped_labels[i] = 0
    return flipped_labels

def prepare_lable_shape(labels, new_item_shape=(4, 4, 1)):
    'split labels'
    ones_l = []

    for i in range(labels.shape[0]):
        if labels[i] > 0:
            ones_l.append(i)

    #ones_Matrix = np.ones((len(ones_l), ) + new_item_shape)
    new_shaped_labels = np.zeros((labels.shape[0],) + new_item_shape)

    for i in range(len(ones_l)):
        new_shaped_labels[ones_l[i]] = np.ones(new_item_shape)

    return new_shaped_labels

