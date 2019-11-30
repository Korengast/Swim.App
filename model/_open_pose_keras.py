import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
import scipy
import math
import pandas as pd
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment


class Open_Pose_Keras:

    def __init__(self, load_coco_weights=True):
        self.__load_coco_weights = load_coco_weights
        self.__build()

        self.__model = None

    def train(self, config, train_client, val_client, iterations_per_epoch, validation_steps, metrics_id,
              last_epoch, use_client_gen, callbacks_list):

        base_lr = 2e-5
        momentum = 0.9
        weight_decay = 5e-4
        lr_policy = "step"
        gamma = 0.333
        stepsize = 121746 * 17  # in original code each epoch is 121746 and step change is on 17th epoch
        max_iter = 200

        for epoch in range(last_epoch, max_iter):
            train_di = train_client.gen()

            # train for one iteration
            self.__model.fit_generator(train_di,
                                       steps_per_epoch=iterations_per_epoch,
                                       epochs=epoch + 1,
                                       callbacks=callbacks_list,
                                       use_multiprocessing=False,
                                       # TODO: if you set True touching generator from 2 threads will stuck the program
                                       initial_epoch=epoch
                                       )

            self.__validate(config, val_client, validation_steps, metrics_id, epoch + 1)

    def predict(self, input_img):
        return self.__model.predict(input_img)

    def __build(self):

        input_shape = (None, None, 3)
        stages = 6
        np_branch1 = 38
        np_branch2 = 19

        weights_path = "model/keras_model_weights.h5"
        img_input = Input(shape=input_shape)
        img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

        stage0_out = self.__vgg_block(img_normalized)

        stage1_branch1_out = self.__stage1_block(stage0_out, np_branch1, 1)
        stage1_branch2_out = self.__stage1_block(stage0_out, np_branch2, 2)
        x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        for sn in range(2, stages + 1):
            stageT_branch1_out = self.__stageT_block(x, np_branch1, sn, 1)
            stageT_branch2_out = self.__stageT_block(x, np_branch2, sn, 2)
            if (sn < stages):
                x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

        self.__model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
        if self.__load_coco_weights:
            self.__model.load_weights(weights_path)

    def __relu(self, x):
        return Activation('relu')(x)

    def __conv(self, x, nf, ks, name):
        x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
        return x1

    def __pooling(self, x, ks, st, name):
        x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
        return x

    def __vgg_block(self, x):
        # Block 1
        x = self.__conv(x, 64, 3, "conv1_1")
        x = self.__relu(x)
        x = self.__conv(x, 64, 3, "conv1_2")
        x = self.__relu(x)
        x = self.__pooling(x, 2, 2, "pool1_1")

        # Block 2
        x = self.__conv(x, 128, 3, "conv2_1")
        x = self.__relu(x)
        x = self.__conv(x, 128, 3, "conv2_2")
        x = self.__relu(x)
        x = self.__pooling(x, 2, 2, "pool2_1")

        # Block 3
        x = self.__conv(x, 256, 3, "conv3_1")
        x = self.__relu(x)
        x = self.__conv(x, 256, 3, "conv3_2")
        x = self.__relu(x)
        x = self.__conv(x, 256, 3, "conv3_3")
        x = self.__relu(x)
        x = self.__conv(x, 256, 3, "conv3_4")
        x = self.__relu(x)
        x = self.__pooling(x, 2, 2, "pool3_1")

        # Block 4
        x = self.__conv(x, 512, 3, "conv4_1")
        x = self.__relu(x)
        x = self.__conv(x, 512, 3, "conv4_2")
        x = self.__relu(x)

        # Additional non vgg layers
        x = self.__conv(x, 256, 3, "conv4_3_CPM")
        x = self.__relu(x)
        x = self.__conv(x, 128, 3, "conv4_4_CPM")
        x = self.__relu(x)

        return x

    def __stage1_block(self, x, num_p, branch):
        # Block 1
        x = self.__conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
        x = self.__relu(x)
        x = self.__conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
        x = self.__relu(x)
        x = self.__conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
        x = self.__relu(x)
        x = self.__conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
        x = self.__relu(x)
        x = self.__conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)

        return x

    def __stageT_block(self, x, num_p, stage, branch):
        # Block 1
        x = self.__conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
        x = self.__relu(x)
        x = self.__conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
        x = self.__relu(x)
        x = self.__conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
        x = self.__relu(x)
        x = self.__conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
        x = self.__relu(x)
        x = self.__conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
        x = self.__relu(x)
        x = self.__conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
        x = self.__relu(x)
        x = self.__conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))

        return x

    def __validate(self, config, val_client, validation_steps, metrics_id, epoch):

        val_di = val_client.gen()
        from keras.utils import GeneratorEnqueuer

        val_thre = GeneratorEnqueuer(val_di)
        val_thre.start()

        model_metrics = []
        inhouse_metrics = []

        for i in range(validation_steps):

            X, GT = next(val_thre.get())

            Y = self.__model.predict(X)

            model_losses = [(np.sum((gt - y) ** 2) / gt.shape[0] / 2) for gt, y in zip(GT, Y)]
            mm = sum(model_losses)

            if config.paf_layers > 0 and config.heat_layers > 0:
                GTL6 = np.concatenate([GT[-2], GT[-1]], axis=3)
                YL6 = np.concatenate([Y[-2], Y[-1]], axis=3)
                mm6l1 = model_losses[-2]
                mm6l2 = model_losses[-1]
            elif config.paf_layers == 0 and config.heat_layers > 0:
                GTL6 = GT[-1]
                YL6 = Y[-1]
                mm6l1 = None
                mm6l2 = model_losses[-1]
            else:
                assert False, "Wtf or not implemented"

            m = self.__calc_batch_metrics(i, GTL6, YL6, range(config.heat_start, config.bkg_start))
            inhouse_metrics += [m]

            model_metrics += [(i, mm, mm6l1, mm6l2, m["MAE"].sum() / GTL6.shape[0], m["RMSE"].sum() / GTL6.shape[0],
                               m["DIST"].mean())]
            print("Validating[BATCH: %d] LOSS: %0.4f, S6L1: %0.4f, S6L2: %0.4f, MAE: %0.4f, RMSE: %0.4f, DIST: %0.2f" %
                  model_metrics[-1])

        inhouse_metrics = pd.concat(inhouse_metrics)
        inhouse_metrics['epoch'] = epoch
        inhouse_metrics.to_csv("logs/val_scores.%s.%04d.txt" % (metrics_id, epoch), sep="\t")

        model_metrics = pd.DataFrame(model_metrics,
                                     columns=("batch", "loss", "stage6l1", "stage6l2", "mae", "rmse", "dist"))
        model_metrics['epoch'] = epoch
        del model_metrics['batch']
        model_metrics = model_metrics.groupby('epoch').mean()
        with open('%s.val.tsv' % metrics_id, 'a') as f:
            model_metrics.to_csv(f, header=(epoch == 1), sep="\t", float_format='%.4f')

        val_thre.stop()

    def __calc_batch_metrics(self, batch_no, gt, Y, heatmap_layers):

        MAE = Y - gt
        MAE = np.abs(MAE)
        MAE = np.mean(MAE, axis=(1, 2))

        RMSE = (Y - gt) ** 2
        RMSE = np.mean(RMSE, axis=(1, 2))
        RMSE = np.sqrt(RMSE)

        gt_parts = np.full((gt.shape[0], gt.shape[3]), np.nan)
        y_parts = np.full((gt.shape[0], gt.shape[3]), np.nan)
        y_dist = np.full((gt.shape[0], gt.shape[3]), np.nan)

        for n in range(gt.shape[0]):
            for l in heatmap_layers:
                y_peaks = self.__find_peaks(Y[n, :, :, l])
                y_parts[n, l] = len(y_peaks)
                gt_peaks = self.__find_peaks(gt[n, :, :, l])
                gt_parts[n, l] = len(gt_peaks)
                y_dist[n, l] = self.__assign_peaks(y_peaks, gt_peaks)

        batch_index = np.full(fill_value=batch_no, shape=MAE.shape)
        item_index, layer_index = np.mgrid[0:MAE.shape[0], 0:MAE.shape[1]]

        metrics = pd.DataFrame({'batch': batch_index.ravel(),
                                'item': item_index.ravel(),
                                'layer': layer_index.ravel(),
                                'MAE': MAE.ravel(),
                                'RMSE': RMSE.ravel(),
                                'GT_PARTS': gt_parts.ravel(),
                                'Y_PARTS': y_parts.ravel(),
                                'DIST': y_dist.ravel()
                                },
                               columns=('batch', 'item', 'layer', 'MAE', 'RMSE', 'GT_PARTS', 'Y_PARTS', 'DIST')
                               )

        return metrics

    def __find_peaks(self, layer, thre1=0.01):
        map_ori = cv2.resize(layer, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        map = gaussian_filter(map_ori, sigma=3)
        peaks_binary = (map == maximum_filter(map, 3)) & (map > thre1)

        if np.count_nonzero(peaks_binary) > 50:
            return []  # safety valve from N^2 in next stages

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

        return peaks_with_score

    def __assign_peaks(self, layer_y, layer_gt):

        if len(layer_y) == 0 and len(layer_gt) == 0:
            return np.nan

        if len(layer_y) == 0 or len(layer_gt) == 0:
            return 400

        d = np.array(layer_y)
        t = np.array(layer_gt)

        dx = np.subtract.outer(d[:, 0], t[:, 0])
        dy = np.subtract.outer(d[:, 1], t[:, 1])
        distance = np.sqrt(dx ** 2 + dy ** 2)
        # print(distance)

        y, gt = linear_sum_assignment(distance)
        # print(np.array(list(zip(y,gt))))

        dist = [distance[foo] for foo in zip(y, gt)]  # TODO: use numpy
        # print(dist)

        dist += [400] * (len(layer_y) - len(y))
        dist += [400] * (len(layer_gt) - len(gt))

        dist = np.mean(dist)

        return dist