import cv2 as cv
import h5py
import os
import shutil
import glob
import numpy as np
import time 
from PIL import Image
from pathlib import Path

from edgetpu.classification.engine import ClassificationEngine

from threading import Thread
import logging

logger = logging.getLogger('reachy.flyers')

class Embeddings(object):
    def __init__(self, facenet_path, embeddings_data_path):
        if os.path.exists(embeddings_data_path):
            shutil.rmtree(embeddings_data_path)
            os.mkdir(embeddings_data_path)
        else:
            os.mkdir(embeddings_data_path)
        
        self.embeddings_data_path = embeddings_data_path
        self.facenet_engine = ClassificationEngine(facenet_path)
                
    def get_embedding(self, face):
        face = self.resize_face(face)
        emb = self.facenet_engine.classify_with_input_tensor(face, top_k=130, threshold=-0.1)
        emb.sort(key=lambda emb: emb[0])
        return emb

    def get_id_from_embedding(self, emb_to_id, threshold):
        try:
            names_list = np.load(self.embeddings_data_path + '/names_list.npy')
            emb_list = np.load(self.embeddings_data_path + '/emb_list.npy')
            
        except FileNotFoundError:
            return 'Unknown', 0

        comp_arr = []
        occurence_dic = np.load(self.embeddings_data_path + '/occ_dic.npy', allow_pickle='True').item()
        nb_classes = int(np.shape(emb_list)[1] / 5)

        for i in range(nb_classes):
            diff_list = np.linalg.norm(emb_to_id-emb_list[:, 5*i:5*(i+1)], axis=0)
            name = names_list[np.argmin(diff_list) + 5*i].split('.')[0]
            comp_arr.append([name, np.mean(diff_list)])

        if min([comp_val[1] for comp_val in comp_arr]) < threshold:
            name = comp_arr[np.argmin([comp_val[1] for comp_val in comp_arr])][0]
            occurence_dic[name] += 1
            np.save(self.embeddings_data_path + '/occ_dic.npy', occurence_dic)
            return name, nb_classes

        return 'Unknown', nb_classes

    def add_someone(self, name, ssd_engine, stack_im):
        try:
            names_list = np.ndarray.tolist(np.load(self.embeddings_data_path + '/names_list.npy'))
            emb_list = np.ndarray.tolist(np.load(self.embeddings_data_path + '/emb_list.npy'))
            occurence_dic = np.load(self.embeddings_data_path + '/occ_dic.npy', allow_pickle='True').item()
            first = False
        except FileNotFoundError:
            first = True
            logger.info('########## Failed to get embedding files')
            names_list, emb_list, occurence_dic = np.array([]), np.array([]), {}

        emb_list_prov = np.array([])

        # Metrics
        occurence_dic[name] = 0

        for i in range(5):
            pil_img = Image.fromarray(stack_im[i])
            face_coord, size = [], []

            candidates = ssd_engine.detect_with_image(pil_img, relative_coord=False)
            if not candidates:
                logger.info('Nobody on one of the stacked images')
                return

            for candidate in candidates:
                [[x1, y1], [x2, y2]] = candidate.bounding_box
                face_coord.append([int(y1), int(y2), int(x1), int(x2)])
                size.append((y2-y1)*(x2-x1))

            y1, y2, x1, x2 = face_coord[np.argmax(size)][:]
            names_list = np.append(names_list, str(name) + '.' + str(i))

            emb_list_prov = np.append(emb_list_prov, np.array([data[1] for data in self.get_embedding(stack_im[i][y1:y2, x1:x2])]))

        emb_list_prov = emb_list_prov.reshape(5, 128).T

        if first:
            emb_list = emb_list_prov
        else:
            emb_list = np.hstack((emb_list, emb_list_prov))

        np.save(self.embeddings_data_path + '/names_list.npy', names_list)
        np.save(self.embeddings_data_path + '/emb_list.npy', emb_list)
        np.save(self.embeddings_data_path + '/occ_dic.npy', occurence_dic)
        

    def resize_face(self, face):
        face = cv.resize(face, (160, 160), interpolation=cv.INTER_LINEAR)
        return np.asarray(face).flatten()