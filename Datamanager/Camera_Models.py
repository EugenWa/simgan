import numpy as np
from Data_sources.Noise_Models import *
from Data_sources.simple_math_operations import *
from Data_sources.Triangles import *

class Camera:
    def __init__(self, coor = np.array([0, 0, -1.5]), norm = np.array([0, 0, 1]), imagedim = [64, 64], imgframe = [[-2, 2, 0], [2, -2, 0]] ,d = None):    # kp was D sein soll -> evtl. scanline verschiebungsrate
        self.coor = coor                # Coordinaten der Kamera
        self.norm = norm                # Normale der Ebene, aka kamera richtung
        self.imagedim = imagedim        # bildgröße
        self.imgframe = imgframe        # ebene durch die raycasts laufen
        self.D = d

    def raycast_render(self, triangle, image):
        farestpoint = triangle.vertex_tensor[np.argmax(np.sum(np.square(triangle.vertex_tensor - self.coor), axis=1))]

        #create image
        # create raycasts
        incrementx = abs(self.imgframe[0][0] - self.imgframe[1][0]) / self.imagedim[0]
        incrementy = abs(self.imgframe[0][1] - self.imgframe[1][1]) / self.imagedim[1]

        dmax = 40 #np.sqrt(np.sum(np.square(farestpoint)))
        x = self.imgframe[0][0]
        y = self.imgframe[0][1]
        for i in range(self.imagedim[1]):
            #print("NEW Y")
            x = self.imgframe[0][0]
            for j in range(self.imagedim[0]):
                raystart = np.array([x, y, 0])
                #print(x, y)
                x += incrementx

                ray = [self.coor, raystart, raystart-self.coor]
                intspoint = intersect_plane_linepoint(ray, triangle.ebene)
                #checken wo der point liegt
                dist = 100000
                if intspoint is None:
                    #background
                    image[i,j] = 255
                elif intspoint is "all":
                    intspoint = raystart
                    if triangle.check_loc(intspoint):
                        dist = np.sqrt(np.sum(np.square(intspoint-self.coor)))
                else:
                    if triangle.check_loc(intspoint):
                        dist = np.sqrt(np.sum(np.square(intspoint-self.coor)))

                if dist >= dmax:
                    # background
                    image[i, j] = image[i, j]
                else:
                    if image[i, j] > ((dist / dmax) * 255):
                        image[i, j] = int((dist / dmax) * 255)



            y -= incrementy
        return image

    def in_depth_render(self, triangle, image):
        farestpoint = triangle.vertex_tensor[np.argmax(np.sum(np.square(triangle.vertex_tensor - self.coor), axis=1))]

        # create image
        # create raycasts
        incrementx = abs(self.imgframe[0][0] - self.imgframe[1][0]) / self.imagedim[0]
        incrementy = abs(self.imgframe[0][1] - self.imgframe[1][1]) / self.imagedim[1]


        dmax = 12.0  # np.sqrt(np.sum(np.square(farestpoint)))
        x = self.imgframe[0][0]
        y = self.imgframe[0][1]
        for i in range(self.imagedim[1]):
            x = self.imgframe[0][0]
            for j in range(self.imagedim[0]):
                ray_2 = np.array([x, y, 0])
                ray_1 = np.array([x, y, -1])

                x += incrementx

                ray = [ray_1, ray_2, ray_2 - ray_1]
                intspoint = intersect_plane_linepoint(ray, triangle.ebene)
                # checken wo der point liegt
                dist = 100000
                if intspoint is None:
                    # background
                    image[i, j] = 255
                elif intspoint is "all":
                    intspoint = ray_2
                    if triangle.check_loc(intspoint):
                        dist = intspoint[2] - ray_1[2]
                else:
                    if triangle.check_loc(intspoint):
                        dist = intspoint[2] - ray_1[2]

                if dist >= dmax:
                    # background
                    image[i, j] = image[i, j]
                else:
                    if image[i, j] > ((dist / dmax) * 255):
                        image[i, j] = int((dist / dmax) * 255)

            y -= incrementy

        return image

    def triangle_intersectionpoints(self, triangle, current_d_ebene):       # aktuelle verschiebung der scnanebene, bem: d von camera ursprung nicht KOS ursprung
        # schnittpunkt mit allen geraden berechnen
        intersec_points = []
        for idx in range(3):
            gerade = [triangle.vertex_tensor[idx%3], triangle.vertex_tensor[(idx+1)%3], triangle.geraden_tensor[idx]]
            intersec = intersect_plane_linelambda(gerade,[current_d_ebene, self.norm])
            if intersec is None:
                # parallel
                pass
            elif intersec is "all":
                pass
            else:
                # check if the point is between the bounds
                if intersec < 0 or intersec > 1:                # da der geraden richtungsvektor die länge entsprechend der distanz zwischen den punkten besitzt
                    pass
                else:
                    intersec_points.append(gerade[0] + intersec*gerade[2])
        return intersec_points


