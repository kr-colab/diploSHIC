from keras.utils import Sequence
import numpy as np
import gc


def load_fvecs_from_directory(directory, n_subwin=11):
  hard = np.loadtxt(directory + "hard.fvec", skiprows=1)
  nDims = int(hard.shape[1] / n_subwin)
  h1 = np.reshape(hard, (hard.shape[0], nDims, n_subwin))
  neut = np.loadtxt(directory + "neut.fvec", skiprows=1)
  n1 = np.reshape(neut, (neut.shape[0], nDims, n_subwin))
  soft = np.loadtxt(directory + "soft.fvec", skiprows=1)
  s1 = np.reshape(soft, (soft.shape[0], nDims, n_subwin))
  lsoft = np.loadtxt(directory + "linkedSoft.fvec", skiprows=1)
  ls1 = np.reshape(lsoft, (lsoft.shape[0], nDims, n_subwin))
  lhard = np.loadtxt(directory + "linkedHard.fvec", skiprows=1)
  lh1 = np.reshape(lhard, (lhard.shape[0], nDims, n_subwin))
  both = np.concatenate((h1, n1, s1, ls1, lh1))
  y = np.concatenate((np.repeat(0, len(h1)),
                      np.repeat(1, len(n1)),
                      np.repeat(2, len(s1)),
                      np.repeat(3, len(ls1)),
                      np.repeat(4, len(lh1)),))
  return both.reshape(both.shape[0], nDims, n_subwin, 1), y


def load_empirical_fvecs_from_directory(directory, n_subwin=11):
  nDims =  int(emp.shape[1] / n_subwin)
  emp = np.loadtxt(directory + "empirical.fvec", skiprows=1)
  emp = np.reshape(emp, (emp.shape[0], nDims, n_subwin))
  return emp.reshape(emp, emp.shape[0], nDims, n_subwin, 1)


class DADiploSHICDataLoader(Sequence):
  def __init__(self, X_src, X_tgt, Y_pred, batch_size):
    self.tgt_data = X_tgt
    self.src_data = X_src
    self.y_pred = Y_pred
    
    self.batch_size = batch_size
    src_size = self.src_data.shape[0]
    tgt_size = self.tgt_data.shape[0]

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
                                     -1*np.ones((len(discrSrc_batch_idx), self.y_pred.shape[1])),
                                     -1*np.ones((len(discrTgt_batch_idx), self.y_pred.shape[1]))))
    batch_Y_discr = np.concatenate((-1*np.ones(len(pred_batch_idx)),
                                    np.zeros(len(discrSrc_batch_idx)),
                                    np.ones(len(discrTgt_batch_idx))))
    assert batch_X.shape[0] == self.batch_size*2, (batch_X.shape, self.batch_size*2)
    assert batch_Y_pred.shape[0] == batch_Y_discr.shape[0], (batch_Y_pred.shape, batch_Y_discr.shape)
    return batch_X, {"predictor":batch_Y_pred, "discriminator":batch_Y_discr}
  
  
class DiploSHICDataLoader(Sequence):
  def __init__(self, X_src, Y_pred, batch_size):
    self.data = X_src
    self.y_pred = Y_pred
    self.batch_size = batch_size
    size = self.data.shape[0]
    self.no_batch = int(np.floor(size/ self.batch_size))
    self.pred_idx = np.arange(size)
    np.random.shuffle(self.pred_idx)

  def __len__(self):
    return self.no_batch

  def on_epoch_end(self):
    np.random.shuffle(self.pred_idx)
    gc.collect()

  def __getitem__(self, idx):
    pred_batch_idx = self.pred_idx[idx*self.batch_size:(idx+1)*self.batch_size]
    batch_X = self.data[pred_batch_idx]
    batch_Y_pred = self.y_pred[pred_batch_idx]
    return batch_X, batch_Y_pred
