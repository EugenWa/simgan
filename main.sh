#!/bin/sh
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py ImgBasic1 0 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py ImgBasic2 0 0.0002 0.5 30 100


CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py ImgOneRes1 1 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py ImgOneRes2 1 0.0002 0.5 30 100


CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py ImgTwoRes1 2 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py ImgTwoRes2 2 0.0002 0.5 30 100


CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py UpscaleBasic1 3 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py UpscaleBasic2 3 0.0002 0.5 30 100


CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py MtPl1 4 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py MtPl2 4 0.0002 0.5 30 100

CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py firstgen1 5 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py firstgen2 5 0.0002 0.5 30 100


pytCUDA_VISIBLE_DEVICES="0" ipython3hon3 supervised_training_of_generators.py refinergen1 6 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py refinergen2 6 0.0002 0.5 30 100


CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py fullres1 7 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py fullres1 7 0.0002 0.5 30 100

CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py fullresnobias 8 0.001 0.5 30 100
CUDA_VISIBLE_DEVICES="0" ipython3 supervised_training_of_generators.py fullresnobias 8 0.0002 0.5 30 100

