import torch
import random
from torch import nn
from torch.autograd import Variable
import torch
import random
from torch import nn
from torch.autograd import Variable


## Write cnn modules for gex modalities
class gexCNN(nn.Module):
    """customized  module"""
    #argument index is the poisition for each choromosome
    def __init__(self, kernel_size):
        super(gexCNN, self).__init__()

        # Conv layer
        self.in_channels = 1 
        self.out_channels = config.N_CHANNELS
        self.kernel_size = kernel_size   
        self.stride = 50 # TO CHANGE 
        self.padding = 25 # TO CHANGE
        self.pool_size = 2
        self.pool_stride = 1
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels = self.in_channels, 
                      out_channels = self.out_channels, 
                      kernel_size = self.kernel_size,
                      stride = self.stride,
                      padding = self.padding),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = self.pool_size,
                         stride = self.pool_stride)
        )

        # # FC layer
        # self.conv_out_features = int((config.N_GENES + 2*self.padding - self.kernel_size) / self.stride + 1)
        # self.fc_in_features = int((self.conv_out_features - self.pool_size) / self.pool_stride + 1) * self.out_channels
        # self.fc_out_feature = 300
        # self.fc = nn.Linear(in_features = self.fc_in_features, out_features = self.fc_out_feature) 

    def forward(self, x):
        r"""  
        Generate GEX embeddings
        
        Parameters
        ----------
        x
            Pre-processed GEX data (batch_size x 1 x N_GENES)
        
        Returns
        -------
        gex_embed
            GEX embeddings of a batch (batch_size x seq_len x dim_size)
        """
        gex_embed = self.convs(x.float())
        # gex_embed = torch.flatten(gex_embed, 1)
        # gex_embed = self.fc(gex_embed)
        return gex_embed.transpose(1,2).to(config.DEVICE)

# Write cnn modules for atac modalities
class atacCNN(nn.Module):
    #argument index is the poisition for each choromosome
    def __init__(self, index, kernel_size_1, kernel_size_2):
        super(atacCNN, self).__init__()
        self.index = index
        
        # Conv layer
        self.in_channels_1 = 1 
        self.out_channels_1 = int(config.N_CHANNELS / 2)
        self.kernel_size_1 = kernel_size_1
        self.stride_1 = 10 # TO CHANGE 
        self.padding_1 = 5 # TO CHANGE

        self.in_channels_2 = int(config.N_CHANNELS / 2)
        self.out_channels_2 = config.N_CHANNELS 
        self.kernel_size_2 = kernel_size_2
        self.stride_2 = 5 # TO CHANGE 
        self.padding_2 = 3 # TO CHANGE

        self.convs = nn.Sequential(
            nn.Conv1d(in_channels = self.in_channels_1, 
                      out_channels = self.out_channels_1, 
                      kernel_size = self.kernel_size_1,
                      stride = self.stride_1,
                      padding = self.padding_1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 5, stride = 2),

            nn.Conv1d(in_channels = self.in_channels_2, 
                      out_channels = self.out_channels_2, 
                      kernel_size = self.kernel_size_2,
                      stride = self.stride_2,
                      padding = self.padding_2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 1)             
        )



    def forward(self, x):
        r"""  
        Generate ATAC embeddings
        
        Parameters
        ----------
        x
            Pre-processed ATAC data (batch_size x 1 x N_PEAKS)
        
        Returns
        -------
        atac_embed
            ATAC embeddings of a batch (batch_size x seq_len x dim_size)
        """
        atac_embed = []
        for chr in self.index.keys(): 
            idx = self.index[chr]
            x_chr = x[:,:,idx]
            x_chr = self.convs(x_chr.float())
            atac_embed.append(x_chr)
        atac_embed = torch.cat(atac_embed, dim = 2)
        return atac_embed.transpose(1,2).to(config.DEVICE)



class MultimodalAttention(nn.Module):
    def __init__(self):
        super(MultimodalAttention, self).__init__()
        self.nhead_gex = 1
        self.nhead_atac = 4
        self.nhead_multi = 4
        self.nlayer_gex = 1
        self.nlayer_atac = 1
        self.nlayer_multi = 1

        self.encoder_layer_gex = nn.TransformerEncoderLayer(d_model = config.N_CHANNELS, nhead = self.nhead_gex)
        self.transformer_encoder_gex = nn.TransformerEncoder(self.encoder_layer_gex, num_layers = self.nlayer_gex)
        # self.linear_gex_0 = nn.LazyLinear(out_features = 1)
        self.linear1_gex_0 = nn.LazyLinear(out_features = 10)
        self.linear2_gex_0 = nn.LazyLinear(out_features = 300)


        self.encoder_layer_atac = nn.TransformerEncoderLayer(d_model = config.N_CHANNELS, nhead = self.nhead_atac)
        self.transformer_encoder_atac = nn.TransformerEncoder(self.encoder_layer_atac, num_layers = self.nlayer_atac)
        # self.linear_atac_0 = nn.LazyLinear(out_features = 1)
        self.linear1_atac_0 = nn.LazyLinear(out_features = 30)
        self.linear2_atac_0 = nn.LazyLinear(out_features = 300)

        self.encoder_layer_multi = nn.TransformerEncoderLayer(d_model = config.N_CHANNELS, nhead = self.nhead_multi)
        self.transformer_encoder_multi = nn.TransformerEncoder(self.encoder_layer_multi, num_layers = self.nlayer_multi)
        # self.linear_gex_1 = nn.LazyLinear(out_features = 1)
        # self.linear_atac_1 = nn.LazyLinear(out_features = 1)
        self.linear1_gex_1 = nn.LazyLinear(out_features = 10)
        self.linear2_gex_1 = nn.LazyLinear(out_features = 300)
        self.linear1_atac_1 = nn.LazyLinear(out_features = 30)
        self.linear2_atac_1 = nn.LazyLinear(out_features = 300)
    

    def forward(self, gex_embed, atac_embed):
      r"""  
      Incorporate two self-attention and one cross-attention module

      Parameters
      ----------
      gex_embed
          GEX embeddings of a batch (batch_size x seq_len_gex x dim_size)
      atac_embed
          ATAC embeddings of a batch (batch_size x seq_len_atac x dim_size)

      Returns
      -------
      ## TO FILL
      """
      seq_len_gex = gex_embed.size()[1]
      seq_len_atac = atac_embed.size()[1]

      gex_context = self.transformer_encoder_gex(gex_embed)
      atac_context = self.transformer_encoder_atac(atac_embed)

      # Average self-attention fragment representation
      # gex_out_0 = self.linear_gex_0(gex_context.permute(0,2,1)).squeeze(2)
      # atac_out_0 = self.linear_atac_0(atac_context.permute(0,2,1)).squeeze(2)
      gex_out_0  = self.linear1_gex_0(gex_context.permute(0,2,1)).squeeze(2)
      # print(gex_out_0.size())
      gex_out_0  = self.linear2_gex_0(gex_out_0.flatten(start_dim=1))
      # print(gex_out_0.size())
      atac_out_0 = self.linear1_atac_0(atac_context.permute(0,2,1)).squeeze(2)
      # print(atac_out_0.size())
      atac_out_0 = self.linear2_atac_0(atac_out_0.flatten(start_dim=1))
      # print(atac_out_0.size())

      multi_embed = torch.cat((gex_context, atac_context), dim = 1)
      multi_context = self.transformer_encoder_multi(multi_embed)
      
      multi_context_gex = multi_context[:, :seq_len_gex, :]
      multi_context_atac = multi_context[:, seq_len_gex:, :]
      
      # # Average cross-attention fragment representation
      # gex_out_1 = multi_context_gex.mean(dim = 1)
      # atac_out_1 = multi_context_atac.mean(dim = 1)
      gex_out_1  = self.linear1_gex_1(multi_context_gex.permute(0,2,1)).squeeze(2)
      gex_out_1  = self.linear2_gex_1(gex_out_1.flatten(start_dim=1))
      atac_out_1 = self.linear1_atac_1(multi_context_atac.permute(0,2,1)).squeeze(2)
      atac_out_1 = self.linear2_atac_1(atac_out_1.flatten(start_dim=1))

      return gex_out_0.to(config.DEVICE), gex_out_1.to(config.DEVICE), atac_out_0.to(config.DEVICE), atac_out_1.to(config.DEVICE)


class Encoder(nn.Module):
    def __init__(self, kernel_size_gex, kernel_size_atac_1, kernel_size_atac_2, index):
        super(Encoder, self).__init__()

        self.kernel_size_gex = kernel_size_gex
        self.kernel_size_atac_1 = kernel_size_atac_1
        self.kernel_size_atac_2 = kernel_size_atac_2
        self.index = index

        self.gex_cnn = gexCNN(kernel_size = self.kernel_size_gex)
        self.atac_cnn = atacCNN(kernel_size_1 = self.kernel_size_atac_1, kernel_size_2 = self.kernel_size_atac_2, index = self.index)
        self.multi_attention = MultimodalAttention()

        
    def forward(self, x_gex, x_atac):

        gex_embed = self.gex_cnn(x_gex)
        atac_embed = self.atac_cnn(x_atac)
        gex_out_0, gex_out_1, atac_out_0, atac_out_1 = self.multi_attention(gex_embed, atac_embed)

        return gex_out_0, gex_out_1, atac_out_0, atac_out_1


from numpy.lib.shape_base import row_stack
class bidirectTripletLoss(nn.Module):
    r"""
    
    Output bidirectional triplet loss for two pairs of gex and atac
    ----------
    gex_0_mat: Matrix of GEX embeddings from self-attention (batch_size x embedding_size_0)

    Returns
    -------
    loss
    """
    def __init__(self, alpha, margin):
        super(bidirectTripletLoss, self).__init__()

        self.alpha = alpha
        self.margin = margin
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def similarityScore(self, gex_out_0, gex_out_1, atac_out_0, atac_out_1):
        r"""
        Output similarity scores for two pairs of gex and atac
        ----------


        Returns
        -------
        score: batch_size * batch_size
        similarity score between two modalities
        """ 

        # print(gex_mat.size())
        # print(atac_mat.size())

        # gex_mat0, gex_mat1 = torch.split(gex_mat, config.N_CHANNELS, dim = 1)
        # atac_mat0, atac_mat1 = torch.split(atac_mat,config.N_CHANNELS, dim = 1)

        gex_out_0 = nn.functional.normalize(gex_out_0, dim = 1)
        gex_out_1 = nn.functional.normalize(gex_out_1, dim = 1)
        atac_out_0 = nn.functional.normalize(atac_out_0, dim = 1)
        atac_out_1 = nn.functional.normalize(atac_out_1, dim = 1)
        
        score = torch.mm(gex_out_0, atac_out_0.transpose(0,1)) + self.alpha * torch.mm(gex_out_1, atac_out_1.transpose(0,1))
        return  score.to(config.DEVICE)

    # def triplet(self, score_mat):

    #     batch_size = score_mat.size()[0]       
    #     true_score = torch.diagonal(score_mat)
    #     # print("true_score:\n", true_score)
    #     # print("true_score dimension:\n", true_score.size())
        
    #     reduced_score_mat = score_mat - torch.diag(true_score) # set the diagnoal score to be zero
        
    #     neg_index_1 = torch.argmax(reduced_score_mat, dim = 1) # indices of hard negatives for GEX
    #     neg_index_2 = torch.argmax(reduced_score_mat, dim = 0) # indices of hard negatives for ATAC
    #     # print("neg_index_1:\n", neg_index_1); print("neg_index_2:\n", neg_index_2)
    #     # print("neg_index_1 dimension:\n", neg_index_1.size()); print("neg_index_2 dimension:\n", neg_index_2.size())
    #     neg_1 = score_mat[[range(batch_size), neg_index_1]] # hard negatives for GEX 
    #     neg_2 = score_mat[[neg_index_2, range(batch_size)]] # hard negatives for ATAC
    #     # print("neg_1:\n", neg_1); print("neg_2:\n", neg_2)
        
    #     loss_1 = torch.max(self.margin - true_score + neg_1, torch.zeros(1, batch_size).to(config.DEVICE))
    #     loss_2 = torch.max(self.margin - true_score + neg_2, torch.zeros(1, batch_size).to(config.DEVICE))

    #     return torch.mean(loss_1 + loss_2)
    def idxSemiHardRow(self, score_mat):

        batch_size = score_mat.size()[0]       
        true_score = torch.diagonal(score_mat)
        
        reduced_score_mat = score_mat - torch.diag(true_score) # set the diagnoal score to be zero
        idx_hardneg = torch.argmax(reduced_score_mat, dim = 1) # indices of hard negatives

        # Sample from negatives with similarity score following positive - margin < nagative < positive
        row_scaled = score_mat - true_score.view(-1, 1) # each row scaled by diagonal values
        row_logical = torch.logical_and(row_scaled < 0, row_scaled > - self.margin) # need to sample from Trues
        iidx_sample_from = [[j for j in range(batch_size) if row_logical[i].tolist()[j]] for i in range(batch_size)] # indices that can be sampled from for each row
        # print('iidx_sample_from', iidx_sample_from)

        idx_semi_hard = []
        for i in range(batch_size):
            idx = iidx_sample_from[i]
            if len(idx) > 0: 
                idx_semi_hard.append(random.sample(idx, 1)[0])
            else:
                idx_semi_hard.append(idx_hardneg[i])
        return idx_semi_hard

    def triplet(self, score_mat):

        # print(score_mat)
        batch_size = score_mat.size()[0]       
        true_score = torch.diagonal(score_mat)
        # print("true_score:\n", true_score)
        # print("true_score dimension:\n", true_score.size())

        neg_index_1 = self.idxSemiHardRow(score_mat)
        neg_index_2 = self.idxSemiHardRow(score_mat.T)
        # print("neg_index_1:\n", neg_index_1); print("neg_index_2:\n", neg_index_2)
        neg_1 = score_mat[[range(batch_size), neg_index_1]] # hard negatives for GEX 
        neg_2 = score_mat[[neg_index_2, range(batch_size)]] # hard negatives for ATAC
        # print("neg_1:\n", neg_1); print("neg_2:\n", neg_2)
        
        loss_1 = torch.max(self.margin - true_score + neg_1, torch.zeros(1, batch_size).to(config.DEVICE))
        loss_2 = torch.max(self.margin - true_score + neg_2, torch.zeros(1, batch_size).to(config.DEVICE))

        return torch.mean(loss_1 + loss_2)

    def crossEntropy(self, score_mat):

        batch_size = score_mat.size()[0]
        target = torch.arange(batch_size)

        loss_1 = self.cross_entropy_loss(score_mat, target.to(config.DEVICE))
        loss_2 = self.cross_entropy_loss(score_mat.T, target.to(config.DEVICE))

        return 0.5 * (loss_1 + loss_2)
        
    def cellMatchingProb(self, score_mat):

        score_norm_gex = score_mat.softmax(dim = 0)
        score_norm_atac = score_mat.softmax(dim = 1)

        match_probs = 0.5 * (torch.diagonal(score_norm_gex) + torch.diagonal(score_norm_atac))
        return torch.mean(match_probs)
        
    # def cellTypeMatchingProbRow(self, score_mat, cell_type):

    #     # Collect list of index list for each cell type
    #     idx_in_type = collections.defaultdict(list)
    #     for i, x in enumerate(cell_type):
    #         idx_in_type[x].append(i)

    #     # Compute matching probs for each cell type
    #     score_mat_norm = score_mat.softmax(dim = 0)
    #     probs = []
    #     for idx in idx_in_type.values():
    #         prob_type = 0
    #         for i in idx:
    #             prob_type += score_mat_norm[i, idx].sum()
    #         probs.append(prob_type / len(idx)) 

    #     # Take average of matching prob from cell types
    #     ct_match_prob = torch.tensor(probs).mean()

    #     return ct_match_prob

    # def cellTypeMatchingProb(self, score_mat, cell_type): # NS's method
    #     row = self.cellTypeMatchingProbRow(score_mat, cell_type) # Softmax on rows (normalize GEX)
    #     col = self.cellTypeMatchingProbRow(score_mat.T, cell_type) # Softmax on cols (normalize ATAC) 
        
    #     return 0.5 * (row + col)

    def cellTypeMatchingProb(self, score_mat, cell_type): # XF's method

        # Compute matching probs for each cell type
        idx_in_type = collections.defaultdict(list)
        for i, x in enumerate(cell_type):
            idx_in_type[x].append(i)

        sum_score_mat = torch.zeros(len(idx_in_type.values()),len(idx_in_type.values()))
        for i, dx in enumerate(idx_in_type.values()):
          for j, dx2 in enumerate(idx_in_type.values()):
            tem = score_mat[np.ix_(dx, dx2)].sum()
            sum_score_mat[i,j] = tem

        score_mat_norm = 0.5 * (sum_score_mat.softmax(dim = 0) + sum_score_mat.softmax(dim = 1))
        # print('sum_score_mat:\n', sum_score_mat)
        return torch.mean(torch.diagonal(score_mat_norm))

    def forward(self, gex_out_0, gex_out_1, atac_out_0, atac_out_1, cell_type):
      
        score_mat = self.similarityScore(gex_out_0, gex_out_1, atac_out_0, atac_out_1)#; print("score_mat:\n", score_mat)
        # print('score_mat:\n', score_mat)
        loss_triplet = self.triplet(score_mat)
        loss_cross = self.crossEntropy(score_mat)
        loss = loss_triplet + loss_cross
        # print(score_mat); print(cell_type)
        ct_match_prob = self.cellTypeMatchingProb(score_mat, cell_type)
        cell_match_prob = self.cellMatchingProb(score_mat)

        return loss.to(config.DEVICE), loss_triplet, loss_cross, ct_match_prob, cell_match_prob


























