import numpy as np
from Data_sources.simple_math_operations import *



class Triangle:
    def __init__(self, vert1, vert2, vert3):
        self.vertex_tensor = np.zeros((3, 3))
        self.vertex_tensor[0] = vert1
        self.vertex_tensor[1] = vert2
        self.vertex_tensor[2] = vert3

        self.geraden_tensor = np.zeros((3, 3))
        self.geraden_tensor[0] = vert2 - vert1
        self.geraden_tensor[1] = vert3 - vert2
        self.geraden_tensor[2] = vert1 - vert3

        # ebeneneinteilung

        #ebenengleichungs
        self.norm = np.cross(self.geraden_tensor[0], self.geraden_tensor[1])    # NORMIEREN
        # norm vom ursprung wegzeigen lassen
        if np.dot((vert1), self.norm) < 0:
            self.norm = self.norm*(-1)
        self.norm = self.norm * 1/np.sqrt(np.sum(np.square(self.norm)))
        self.ebene = [np.dot(self.vertex_tensor[0], self.norm), self.norm]#[np.sqrt(np.sum(np.square(self.vertex_tensor[0]))), self.norm]
        #print("G1: ", self.geraden_tensor[0])
        #print("G2: ", self.geraden_tensor[1])
        #print("Norm: ", self.norm)
        # eingehende seiten
        self.g1n = np.cross(self.geraden_tensor[0], self.norm)
        self.g2n = np.cross(self.geraden_tensor[1], self.norm)
        self.g3n = np.cross(self.geraden_tensor[2], self.norm)

        # in die richtige richung drehen
        if np.dot(self.g1n, self.geraden_tensor[1]) < 0:
            self.g1n = self.g1n * (-1)
        if np.dot(self.g2n, self.geraden_tensor[2]) < 0:
            self.g2n = self.g2n * (-1)
        if np.dot(self.g3n, self.geraden_tensor[0]) < 0:
            self.g3n = self.g3n * (-1)

        # schnittpunkt berechnen
        tmp = intersect_lines([vert3, self.g1n], [vert1, self.geraden_tensor[0]])
        self.s1 = vert1 + tmp[1]*self.geraden_tensor[0]

        tmp = intersect_lines([vert1, self.g2n], [vert2, self.geraden_tensor[1]])
        self.s2 = vert2 + tmp[1] * self.geraden_tensor[1]

        tmp = intersect_lines([vert2, self.g3n], [vert3, self.geraden_tensor[2]])
        self.s3 = vert3 + tmp[1] * self.geraden_tensor[2]


    def fill(self):
        pass

    def check_loc(self, point):
        # in von g1
        if np.dot(point-self.s1, self.g1n) < 0:
            return False
        else:
            if np.dot(point-self.s2, self.g2n) < 0:
                return False
            else:
                if np.dot(point-self.s3, self.g3n) < 0:
                    return False
        return True

