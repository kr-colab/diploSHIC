from keras.utils import Sequence
import numpy as np
import gc


class DADiploSHICDataLoader(Sequence):
  def __init__(self, X_src, X_tgt, Y_pred, batch_size):
    self.tgt_data = X_src
    self.src_data = X_tgt
    self.y_pred = Y_pred

    self.batch_size = batch_size

    src_size = self.src_bgtm.shape[0]
    tgt_size = self.tar_bgtm.shape[0]

    self.no_batch = int(np.floor(np.minimum(src_size, tgt_size) / self.batch_size)) # model sees training sample at most once per epoch
    self.src_pred_idx = np.arange(src_size)
    self.src_discr_idx = np.arange(src_size)
    self.tgt_discr_idx = np.arange(tgt_size)

    np.random.shuffle(self.src_pred_idx)
    np.random.shuffle(self.src_discr_idx)
    np.random.shuffle(self.tgt_discr_idx)

  def __len__(self):
    return self.no_batch

  def on_epoch_end(self):
    np.random.shuffle(self.src_pred_idx)
    np.random.shuffle(self.src_discr_idx)
    np.random.shuffle(self.tgt_discr_idx)
    gc.collect()

  def __getitem__(self, idx):
    pred_batch_idx = self.src_pred_idx[idx*self.batch_size:(idx+1)*self.batch_size]
    discrSrc_batch_idx = self.src_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]
    discrTgt_batch_idx = self.tgt_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]

    batch_X = np.concatenate((self.src_data[pred_batch_idx],
                          self.src_data[discrSrc_batch_idx],
                          self.tgt_data[discrTgt_batch_idx]))

    batch_Y_pred = np.concatenate((self.y_pred[pred_batch_idx],
                                   -1*np.ones(len(discrSrc_batch_idx)),
                                   -1*np.ones(len(discrTgt_batch_idx))))

    batch_Y_discr = np.concatenate((-1*np.ones(len(pred_batch_idx)),
                                    np.zeros(len(discrSrc_batch_idx)),
                                    np.ones(len(discrTgt_batch_idx))))

    assert batch_X.shape[0] == self.batch_size*2, batch_X.shape[0]
    assert batch_Y_pred.shape == batch_Y_discr.shape, (batch_Y_pred, batch_Y_discr)

    return batch_X, {"predictor":batch_Y_pred, "discriminator":batch_Y_discr}