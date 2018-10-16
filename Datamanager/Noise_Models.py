import numpy as np
from Data_sources.Triangles import *

class Ellipse:
    def __init__(self, dimensions=[15, 5], rotation_angel=0, centerpoint=[15, 20]):
        self.dimension = dimensions
        self.rot_angel = np.deg2rad(rotation_angel)
        self.centerpoint = np.array(centerpoint)

        self.rotation_Matrix = np.array([[np.cos(-rotation_angel), -np.sin(-rotation_angel)],[np.sin(-rotation_angel), np.cos(-rotation_angel)]])

    def rotate_point(self, vec):
        vec_R = np.array([0,0])
        vec_R[0] = np.cos(-self.rot_angel)*vec[0] - np.sin(-self.rot_angel)*vec[1]
        vec_R[1] = np.cos(-self.rot_angel)*vec[1] + np.sin(-self.rot_angel)*vec[0]
        return vec_R
        return np.dot(self.rotation_Matrix, vec)




class Noise_mod:
    def __init__(self):
        self.stencil_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.stencil_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, - 1]])

    def find_region_ofinterest(self, img):
        minmax_I = []
        minmax_J = []
        for i in range(img.shape[0]):
            if abs(255 - np.mean(img[i, :])) > 0.5:
                minmax_I.append(i)
        for i in range(img.shape[1]):
            if abs(255 - np.mean(img[:, i])) > 0.5:
                minmax_J.append(i)
        bound_x = [minmax_I[0], minmax_I[len(minmax_I) - 1]]
        bound_y = [minmax_J[0], minmax_J[len(minmax_J) - 1]]
        return bound_x, bound_y

    '''
    mode 0 -> uniform inside the square
    mode 1 -> gaussian with mu @ center
    '''
    def draw_ellipse_close_to_region_of_interest(self, img, mode=1):
        bound_x, bound_y = self.find_region_ofinterest(img)
        dif = min(bound_x[1] - bound_x[0], (bound_y[1] - bound_y[0])) * 0.5
        if mode < 1:
            e_center = [np.random.uniform(bound_x[0], bound_x[1]), np.random.uniform(bound_y[0], bound_y[1])]
            dim = np.random.uniform(int(dif * 0.2), int(dif * 0.4), (2,))
        else:
            e_center = [(bound_x[1] + bound_x[0])/2, (bound_y[1] + bound_y[0])/2]
            c1 = np.random.normal(e_center[0], dif*0.25, 1)
            c2 = np.random.normal(e_center[1], dif * 0.25, 1)
            e_center = [c1, c2]
            dim = np.random.uniform(int(dif * 0.2), int(dif * 0.4), (2,))

        rot_angel = np.random.randint(0, 180)
        elpse = Ellipse(dim, rot_angel, e_center)

        return self.draw_ellipse(img, elpse)

    '''
    mode 0 -> random lines
    mode 1 -> lines are orthogonal to image borders
    '''
    def draw_lines(self, img, amount, mode=0, normally_Distributed=True, fixed_Size=False, const_linecolor=True):
        bx, by = self.find_region_ofinterest(img)
        if fixed_Size:
            l_width = 2
        else:
            l_width = np.random.randint(1, 4)
        for lines in range(amount):
            if mode == 0:
                if normally_Distributed:
                    p0 = np.random.normal((bx[0] + bx[1]) / 2, 0.5*min(bx[1]-bx[0], by[1]-by[0]), 1)
                    p1 = np.random.normal((by[0] + by[1]) / 2, 0.5*min(bx[1]-bx[0], by[1]-by[0]), 1)
                    sides = np.random.randint(0, 4, (2,))
                    sides[1] = (sides[0] + 2)%4     # oposite side
                    print(sides)
                else:
                    p0 = np.random.randint(0, img.shape[0])
                    p1 = np.random.randint(0, img.shape[1])
                    sides = np.random.randint(0, 4, (2,))
                    while sides[0] == sides[1]:
                        sides[1] = np.random.randint(0, 4)
                point_line = [p0, p1]

                points = []

                for i in range(len(sides)):
                    pi = np.zeros(2)
                    if sides[i] == 0:
                        pi = np.array([0, point_line[i]])
                    elif sides[i] == 1:
                        pi = np.array([point_line[i], img.shape[1]-1])
                    elif sides[i] == 2:
                        pi = np.array([ img.shape[1]-1, point_line[i]])
                    else:
                        pi = np.array([point_line[i], 0])
                    points.append(pi)
            elif mode == 1:
                if normally_Distributed:
                    p0 = np.random.normal((bx[0] + bx[1]) / 2, 0.5*max(bx[1] - bx[0], by[1] - by[0]), 1)
                else:
                    p0 = np.random.randint(0, img.shape[0])
                side = np.random.randint(0, 2)
                points = []
                if side < 1:
                    points.append(np.array([0, p0]))
                    points.append(np.array([img.shape[0]-1, p0]))
                else:
                    points.append(np.array([p0, 0]))
                    points.append(np.array([p0, img.shape[1] - 1]))

            if const_linecolor:
                line_color = 255
            else:
                line_color = np.random.randint(100, 255)

            img = self.draw_line(img, points[0].astype(np.float64), points[1].astype(np.float64), l_width, line_color)
        return img

    def draw_line(self, img, start_l, end_l, linewidth = 3, linecolor=255):
        orientation = end_l-start_l
        len_or = np.linalg.norm(orientation)
        orientation = orientation/len_or
        current_p = start_l-orientation*10
        for i in range(int(len_or + 11)+10):
            for dx in range(linewidth):
                for dy in range(linewidth):
                    point1 = [int(current_p[0] + dx-linewidth/2), int(current_p[1] + dy-linewidth/2)]
                    if point1[0] < 0 or point1[0] >= img.shape[0]:
                        continue
                    if point1[1] < 0 or point1[1] >= img.shape[1]:
                        continue
                    img[point1[0], point1[1]] = linecolor
            current_p += orientation
        return img






    def draw_ellipse(self, img, ellipse):
        length_square = 2*int(max(ellipse.dimension[0]+1, ellipse.dimension[1]+1))
        a_sqr = np.square(ellipse.dimension[0])
        b_sqr = np.square(ellipse.dimension[1])
        for i in range(length_square):
            for j in range(length_square):
                dx = i - length_square/2
                dy = j - length_square/2
                pix = np.array([dx, dy])
                if (ellipse.centerpoint[0] + dx < 0) or (ellipse.centerpoint[0] + dx >= img.shape[0]):
                    continue
                if (ellipse.centerpoint[1] + dy < 0) or (ellipse.centerpoint[1] + dy >= img.shape[1]):
                    continue
                # otherwise paint pixel
                # rotate pixel, to check it its in + scale for norm.form
                pix = np.square(ellipse.rotate_point(pix))
                if (pix[0]/a_sqr + pix[1]/b_sqr) < 1:
                    img[int(ellipse.centerpoint[0] + dx), int(ellipse.centerpoint[1] + dy)]  = 255
        return img


    def mean_filter(self, img):
        fs = 2
        filtershape = 2*fs
        padding = [fs, fs]
        img_shape = img.shape
        img = self.zero_padding(img, padding)

        img_s = np.zeros(img_shape)

        for i in range(img_s.shape[0]):
            for j in range(img_s.shape[1]):
                ij_mean_list = []
                for ix in range(filtershape + 1):
                    for jx in range(filtershape + 1):
                        ij_mean_list.append(img[padding[0] + i + (ix - fs), padding[1] + j + (jx - fs)])

                img_s[i,j] = np.mean(ij_mean_list)

        return img_s

    def sobel_filter(self, img):
        # padding with zeros
        shape_smpl = img.shape
        padding = [1, 1]
        img = self.zero_padding(img, padding)

        '''
            for arrangeing the noise in a better manner
            
        '''
        integrated_image = 0
        counter = 0
        img_hp = np.zeros(shape_smpl)
        #filter
        for i in range(shape_smpl[0]):
            for j in range(shape_smpl[1]):
                tx = 0
                ty = 0
                # !!! this might make no sense, but works hence the stencil is 3x3 !!!
                for st_i in np.linspace(-1, 1, self.stencil_X.shape[0]):
                    for st_j in np.linspace(-1, 1, self.stencil_X.shape[1]):
                        tx += img[i + int(st_i) + padding[0], j+int(st_j) +padding[1]]*self.stencil_X[int(st_i), int(st_j)]
                        ty += img[i + int(st_i) + padding[0], j + int(st_j)+padding[1]] * self.stencil_Y[int(st_i), int(st_j)]

                img_hp[i, j] = np.sqrt(np.square(tx) + np.square(ty))
                integrated_image += abs(img_hp[i, j])
                counter += 1
        return img_hp, (integrated_image/counter)

    def add_noise_static(self, img, img_hp, max_bounds, avrg, treshhold=False, abs_F=False):
        # return img + img_hp

        # make noise calculation
        lower_borderline = max_bounds
        max_deviation = np.max(img_hp)

        bounds = float(lower_borderline) / max_deviation

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if treshhold:
                    if abs(img_hp[i, j]) > (avrg*1.28):
                        if abs_F:
                            img[i, j] += abs(img_hp[i, j] * bounds)
                        else:
                            img[i, j] += img_hp[i, j] * bounds
                else:
                    if img_hp[i, j] is not 0:
                        if abs_F:
                            img[i, j] += abs(img_hp[i, j] * bounds)
                        else:
                            img[i, j] += img_hp[i, j] * bounds

                if img[i, j] > 255:
                    img[i, j] = 255
                elif img[i, j] < 0:
                    img[i, j] = 0
        return img

    def add_noise_std(self, img, img_hp, max_bounds):
        #return img + img_hp

        # make noise calculation
        #lower_borderline = np.min(img)
        #if lower_borderline < 25:
        lower_borderline = max_bounds
        max_deviation = np.max(img_hp)

        bounds = float(lower_borderline)/max_deviation


        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_hp[i, j] is not 0:
                    cs = np.random.randint(0, 2)
                    if cs < 1:
                        img[i, j] += (img_hp[i, j]*np.random.uniform(bounds - bounds*0.25, bounds))
                    else:
                        img[i, j] -= (img_hp[i, j] * np.random.uniform(bounds - bounds*0.25, bounds))
                    if img[i, j] > 255:
                        img[i, j] = 255
                    elif img[i,j] < 0:
                        img[i, j] = 0
        return img

    def add_noise_std_with_trahshold(self, img, img_hp, avrg, max_bounds):
        #return img + img_hp

        lower_borderline = max_bounds
        max_deviation = np.max(img_hp)
        bounds = float(lower_borderline) / max_deviation

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if abs(img_hp[i, j]) > (avrg*1.28):
                    cs = np.random.randint(0, 2)
                    if cs < 1:
                        img[i, j] += (img_hp[i, j]*np.random.uniform(bounds - bounds*0.25, bounds))
                    else:
                        img[i, j] -= (img_hp[i, j] * np.random.uniform(bounds - bounds*0.25, bounds))
                    if img[i, j] > 255:
                        img[i, j] = 255
                    elif img[i,j] < 0:
                        img[i, j] = 0
        return img

    def input_noise_edge(self, img, fixed_dist):
        img_hp, avrg = self.sobel_filter(img)

        if fixed_dist:
            return self.add_noise_std(img, img_hp, 15)
        else:
            return self.add_noise_std_with_trahshold(img, img_hp, avrg, 15)


    def image_line_degradation(self, img):
        bound_x,bound_y =  self.find_region_ofinterest(img)

        max_diff = max(abs(bound_x[1]-bound_x[0]), abs(bound_y[1] - bound_y[0]))
        staff = 2 * abs(bound_x[1]-bound_x[0]) + 2 * abs(bound_y[1] - bound_y[0])
        firstpoint = np.random.uniform(0, staff)

        '''
        secondpoint = np.random.uniform(0, staff)
        while abs(secondpoint-firstpoint) < max_diff * 1.1:
            secondpoint = np.random.uniform(0, staff)
        '''

        safety_margin = 1.1
        L_1 = firstpoint - (max_diff*safety_margin)
        if L_1 < 0 or L_1 < staff*0.01:                     # overlap
            staff_max = staff-L_1
            staff_min = firstpoint + max_diff

            secondpoint = np.random.uniform(staff_min ,staff_max)
        else:                           # NO overlap
            staff2_max = staff
            staff2_min = firstpoint + max_diff
            staff1_max = L_1
            staff1_min = 0

            staff1_lenght = staff1_max-staff1_min
            staff2_lenght = staff2_max-staff2_min

            likelyhood_staff1 = (staff1_lenght/(staff1_lenght+staff2_lenght))
            if np.random.randint(0, 100) < int(likelyhood_staff1*100):
                # point in staff 1
                secondpoint = np.random.uniform(staff1_min, staff1_max)
            else:
                secondpoint = np.random.uniform(staff2_min, staff2_max)
        # create random line through this rectangle

        dx = (bound_x[1] - bound_x[0])
        dy = (bound_y[1] - bound_y[0])
        q1 = dx
        q2 = q1 + dy
        q3 = q2 + q1

        x_i = bound_x[0]
        y_j = bound_y[0]
        if firstpoint < q1:
            x_i += int(firstpoint)
        elif firstpoint < q2:
            x_i += q1
            y_j += (int(firstpoint) - q1)
        elif firstpoint < q3:
            y_j += dy
            x_i += (dx - (
                        int(firstpoint) - q2))
        else:
            y_j += (dy - (
                        int(firstpoint) - q3))

        pointA = [x_i, y_j]

        x_i = bound_x[0]
        y_j = bound_y[0]
        if secondpoint < q1:
            x_i += int(secondpoint)
        elif secondpoint < q2:
            x_i += q1
            y_j += (int(secondpoint) - q1)
        elif secondpoint < q3:
            y_j += dy
            x_i += (dx - (
                    int(secondpoint) - q2))
        else:
            y_j += (dy - (
                    int(secondpoint) - q3))
        pointB = [x_i, y_j]

        linethciness_in_perc = 0.09
        linethinkness = int(min(abs(bound_x[1]-bound_x[0]), abs(bound_y[1] - bound_y[0])) * linethciness_in_perc)
        if linethinkness < 1:
            linethinkness = 1

        line_direction = np.array([pointB[0]-pointA[0], pointB[1]-pointA[1]])
        line_len = np.sqrt(np.sum(np.square(line_direction)))
        line_direction = line_direction/line_len    # normiert

        current_point = pointA
        for i in range(int(line_len+1)):
            # drawline
            for i_t in range(linethinkness):
                if int(current_point[0]) + i_t < img.shape[0] and 0 < int(current_point[0]) - i_t:
                    img[int(current_point[0]) + i_t, int(current_point[1])] = 255
                    img[int(current_point[0]) - i_t, int(current_point[1])] = 255
            for j_t in range(linethinkness):
                if int(current_point[1]) + j_t < img.shape[1] and 0 < int(current_point[1]) - j_t:
                    img[int(current_point[0]), int(current_point[1]) + j_t] = 255
                    img[int(current_point[0]), int(current_point[1]) - j_t] = 255
            current_point[0] += line_direction[0]
            current_point[1] += line_direction[1]

        return img




    def deliation(self, img):
        shape_smpl = img.shape
        padding = [1, 1]
        imgp = self.zero_padding(img, padding)
        for i in range(shape_smpl[0]):
            for j in range(shape_smpl[1]):
                x = i + padding[0]
                y = j + padding[1]
                img[i, j] = max(imgp[x -1, y-1], imgp[x, y-1], imgp[x +1, y-1],
                                imgp[x -1, y], imgp[x, y], imgp[x +1, y],
                                imgp[x -1, y+1], imgp[x , y+1], imgp[x +1, y+1])
        return img

    def mirrow_padding(self, img, pads):
        pass

    def zero_padding(self, img, pads):
        img_p = np.full((img.shape[0] + 2*pads[0], img.shape[1] + 2*pads[1]), 255)
        img_p[pads[0]:-pads[0], pads[1]:-pads[1]] = img
        return img_p