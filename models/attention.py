import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,query_dim,context_dim,embed_dim):
        super().__init__()

        # projection for same scale
        self.query_proj = nn.Linear(query_dim, embed_dim) 
        self.context_proj = nn.Linear(context_dim, embed_dim) 
        
        self.dropout = nn.Dropout(0.1)
        # scoring layer
        self.output_proj = nn.Linear(embed_dim,8)

    def forward(self,context,query):
        # query -->context
        # query [B,query_dim] , context [B,N,context_dim] N = 49 for image as context, N=16 for text as context
        # text--->image                                    
        # context: [B,49,2048]
        # query: [B,512]

        #image-->text
        # context: [B,16,512]
        # query: [B,2048]

        # [B,query_dim] --> [B,1,query_dim]
        query= query.unsqueeze(1) # [B,1,query_dim]
        

        query_proj = self.query_proj(query) #  [B,N,query_dim]-->[B,1,embed_dim]
        context_proj = self.context_proj(context) # [B,N,context_dim]-->[B,N,embed_dim]

        combined_features = self.dropout(torch.tanh(query_proj + context_proj)) # [B,1,embed_dim]+[B,N,embed_dim]=[B,N,embed_dim]

        attention = self.output_proj(combined_features) # [B,N,8]

        weights = torch.softmax(attention,dim=1)  #[B,N,8]
        glimpses = torch.matmul(weights.transpose(1, 2), context)  #[B,8,N]*[B,N,context_dim] = [B,8,2048]

        # context_att = (context * alpha).sum(dim=1) # [B,context_dim] weighted sum  || [B,N,c_dim]*[B,N,1] --> [B,49,c_dim] --> sum dim1 --> [B,c_dim]
        return glimpses.reshape(context.size(0), -1)  # [B,2048*8]
    
