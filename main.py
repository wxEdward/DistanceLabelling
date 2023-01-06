import generator
import model
import pickle
import torch
from fastbook import *
from fastai.collab import *
from fastai.tabular.all import *
from fastai.losses import *
import argparse
import os
import torch.nn as nn

def get_small_emb_sz(dls_df):
    emb = get_emb_sz(dls_df)
    return [(emb[0][0],1)]

def MRELoss(inp, targ) -> Tensor:
  inp = torch.flatten(inp).float()
  targ = torch.flatten(targ).float()
  nom = torch.nn.functional.l1_loss(inp, targ)
  noo = torch.Tensor([0]).repeat(targ.size(0)).to(device)
  denom = torch.nn.functional.l1_loss(noo, targ)
  loss = (nom/denom).mean()
  return loss

def CombineLoss(inp, targ) -> Tensor:
    mse_loss = MSELossFlat()
    loss_1 = mse_loss(inp,targ)

    inp = torch.flatten(inp).float()
    targ = torch.flatten(targ).float()
    nom = torch.nn.functional.l1_loss(inp, targ)
    noo = torch.Tensor([0]).repeat(targ.size(0)).to(device)
    denom = torch.nn.functional.l1_loss(noo, targ)
    loss_2 = (nom/denom).mean()

    return alpha*loss_1 + (1-alpha)*loss_2

parser = argparse.ArgumentParser()
parser.add_argument('-l',"--lgN", type=int, help="Log N")
parser.add_argument('-e', "--epochs", type= int, help="Number of epochs")
#parser.add_argument('-f', "--loss", type = str, help="Loss Function")
parser.add_argument('-a', "--alpha", type = float, help="Alpha for losses")

args = parser.parse_args()

lg_N = args.lgN
num_epochs = args.epochs
alpha = args.alpha
loss_function = CombineLoss
"""
if args.loss == "MRE":
    loss_function = MRELoss
if args.loss == "MSE":
    loss_function = MSELossFlat()
if args.loss == "BOTH":
"""

savepath = 'save/'+str(args.alpha)+'_'+str(args.lgN)+'_'+str(args.epochs)+'/'
"""
os.mkdir(savepath)
dataset, dataset_2, dataset_3, node_id = generator.generate_dataset(lg_N)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = pd.read_csv(savepath+'dataset.csv')
data  = np.array(dataset)
df = pd.DataFrame(data[:,1:], columns = ['src','dst','label'])
"""
df.to_csv(savepath+'dataset.csv')
with open(savepath+'dataset_2.pkl', "wb") as outfile:
    pickle.dump(dataset_2, outfile)
with open(savepath+"node_id.pkl", "wb") as outfile:
    pickle.dump(node_id, outfile)
"""
dataset_2 = pd.read_pickle(savepath+'dataset_2.pkl')
node_id = pd.read_pickle(savepath+'node_id.pkl')

dls_df = CollabDataLoaders.from_df(df, bs=64)
embs = get_small_emb_sz(dls_df)
trainer = model.CollabNN(*embs, y_range=(0,embs[0][0]))
#learn = Learner(dls_df, trainer, loss_func = loss_function, path=savepath)

learn = Learner(dls_df, trainer, loss_func = loss_function, path=savepath)
learn.load('model')
learn.fit_one_cycle(n_epoch = num_epochs, lr_max=5e-3, wd=0.01, cbs=SaveModelCallback(with_opt=True))

torch.save(trainer,savepath+'model.pth')
with open(savepath+"predictions.pkl", "wb") as outfile:
    pickle.dump(model.CollabNN.predictions, outfile)
with open(savepath+"embeddings.pkl", "wb") as outfile:
    pickle.dump(model.CollabNN.embeddings, outfile)
