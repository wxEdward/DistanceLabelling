from fastai.collab import *
from fastai.tabular.all import *

class CollabNN(Module):
    predictions = {}
    embeddings = {}
    # x = {}
    def __init__(self, src_sz, y_range=(0,5.5), n_act=200):
        self.node_factors = Embedding(*src_sz)
        #self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(src_sz[1]+src_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        self.prediction = None

    def forward(self, x):
        x_copy = torch.clone(x)
        #embs = int(2**8 *self.node_factors(x[:,0]))/float(2**8),int(2**8 *self.node_factors(x[:,1]))/float(2**8)
        embs = self.node_factors(x[:,0]), self.node_factors(x[:,1])
        #CollabNN.embs_2 = embs
        x = self.layers(torch.cat(embs, dim=1))
        self.prediction = sigmoid_range(x, *self.y_range)
        # CollabNN.predictions[x_copy] = self.prediction, embs
        for i in range(len(x)):
          pair = x_copy[i]
          CollabNN.predictions[(int(pair[0]),int(pair[1]))] = float(self.prediction[i])
          CollabNN.embeddings[int(pair[0])] = float(embs[0][i])
          CollabNN.embeddings[int(pair[1])] = float(embs[1][i])
          # CollabNN.x[(int(pair[0]),int(pair[1]))] = x_copy[i]
        return self.prediction

    def get_predictions(self):
      return self.prediction

    def quantize_inference(self, x, n_bits):
        embs = (self.node_factors(x[:,0]/max_value)*(2**n_bits)).round()/(2**n_bits) * max_value, (self.node_factors(x[:,1]/max_value)*(2**n_bits)).round()/(2**n_bits) * max_value
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)
