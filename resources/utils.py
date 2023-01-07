import torch
import gc
import collections 
import torch.nn as nn 


def inference(model, criterion, data_val):

    # Initialize encoder & decoder 
    model.eval()
    model.to(config.DEVICE)
    criterion.to(config.DEVICE)
    
    running_loss_cross = 0.0
    running_loss_triplet = 0.0
    running_ct_prob = running_ct_prob2 = 0.0  
    running_cell_prob = running_cell_prob2 = 0.0  
    for iter, data in enumerate(data_val):
      gex_input = data['gex'].to(config.DEVICE)
      atac_input = data['atac'].to(config.DEVICE)
      cell_type_input = data['cell_type']

      ### Forward
      gex_out_0, gex_out_1, atac_out_0, atac_out_1 = model(gex_input, atac_input)

      ### Compute loss
      loss, loss_triplet, loss_cross, ct_match_prob, cell_match_prob = criterion(gex_out_0, gex_out_1, atac_out_0, atac_out_1, cell_type_input)

      running_loss_cross += loss_cross.item()
      running_loss_triplet += loss_triplet.item()
      running_ct_prob += ct_match_prob.item()
      running_cell_prob += cell_match_prob.item()

      del gex_input
      del atac_input
      del cell_type_input
      gc.collect()
      # torch.cuda.empty_cache()
      # if (iter + 1) % 10 == 0: print("iter", iter)

    return running_loss_cross / len(data_val), running_loss_triplet / len(data_val), running_ct_prob / len(data_val), running_cell_prob / len(data_val)