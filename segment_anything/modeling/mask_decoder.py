import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
import pdb
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # HQ-SAM parameters
        #----------------------------------------------------------------------------------------
        self.hf_token = nn.Embedding(1, transformer_dim) # HQ-Ouptput-Token
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) # corresponding new MLP layer for HQ-Ouptput-Token
        self.num_mask_tokens = self.num_mask_tokens + 1
        # three conv fusion layers for obtaining HQ-Feature
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )
        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

        #============================
        self.pca_fc = nn.Linear(self.num_mask_tokens**2,transformer_dim)
        #============================

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
       
        
        #--------------------------------------------------------
        #v1
        # vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        
        #v2--------------
        #vit_features_sum = torch.zeros_like(interm_embeddings[0].permute(0, 3, 1, 2)) 
        #for embeddings in interm_embeddings:
        #    permuted_embeddings = embeddings.permute(0, 3, 1, 2)
        #    vit_features_sum += permuted_embeddings
        #vit_features = vit_features_sum
        #v2--------------
        
        #v3--------------
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) + interm_embeddings[-1].permute(0, 3, 1, 2)
        #--------------------------------------------------------
        
        
        
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_pred = iou_pred[:, mask_slice]
            iou_pred, max_iou_idx = torch.max(iou_pred,dim=1)
            iou_pred = iou_pred.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:,mask_slice]
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens)]
        if hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq
        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b,1,1,1)

        #=============================================================
        pca_list: List[torch.Tensor] = []
        pca = PCA(n_components=mask_tokens_out.shape[1]) 
        pca_features = mask_tokens_out.clone()
        for i in range(self.num_mask_tokens):
            pca_feature_ = pca_features[:, i, :]
            pca_list.append(pca_feature_)
        pca_feature = torch.cat(pca_list, dim=0)
        #===================================================================================PCA
        

        # features = pca_feature.cpu().numpy()
        # principal_components = pca.fit_transform(features)

        # principal_components = torch.from_numpy(principal_components).cuda().reshape(-1)
                
        # principal_components = self.pca_fc(principal_components).unsqueeze(0).cpu().numpy()
        
        # #principal_components = pca.fit_transform(features)
        # output_folder = 'output_images'
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        # def save_image(data, filename, title):
        #     # Normalize data to [0, 255]
        #     data_normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        #     data_normalized = data_normalized.astype(np.uint8)
            
        #     # Apply binary threshold
        #     _, binary_image = cv2.threshold(data_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        #     # Save image
        #     cv2.imwrite(filename, binary_image)
            
        #     # Optional: Save with matplotlib to include the title
        #     plt.imshow(binary_image, cmap='gray')
        #     plt.title(title)
        #     plt.axis('off')
        #     plt.savefig(filename.replace('.png', '_with_title.png'), bbox_inches='tight')
        #     plt.close()
        # original_filename = os.path.join(output_folder, 'original_features.png')
        # save_image(features, original_filename, 'Original Features')
        # pca_filename = os.path.join(output_folder, 'pca_features.png')
        # save_image(principal_components, pca_filename, 'PCA Features')
        
        # print(f"Images saved in {output_folder} folder.")
        # pdb.set_trace()
        #======================================================================================PCA
        #========================
        #principal_components = pca.fit_transform(features)
        #pdb.set_trace()
        hyper_in_list: List[torch.Tensor] = []

        
        
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                #========================================
                #hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
                pca_feature = pca.fit_transform(pca_feature.detach().cpu().numpy())
                pca_feature = torch.from_numpy(pca_feature).cuda().reshape(-1)
                
                pca_feature = self.pca_fc(pca_feature).unsqueeze(0)
                
                concatenated_tensor = mask_tokens_out[:, i, :] + pca_feature
                hyper_in_list.append(self.hf_mlp(concatenated_tensor))
                
                #========================================
    

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        
        masks_sam = (hyper_in[:,:self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:,self.num_mask_tokens-1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_sam_hq],dim=1)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

#=======================================================================sam=====================


# import torch
# from torch import nn
# from torch.nn import functional as F

# from typing import List, Tuple, Type

# from .common import LayerNorm2d


# class MaskDecoder(nn.Module):
#     def __init__(
#         self,
#         *,
#         transformer_dim: int,
#         transformer: nn.Module,
#         num_multimask_outputs: int = 3,
#         activation: Type[nn.Module] = nn.GELU,
#         iou_head_depth: int = 3,
#         iou_head_hidden_dim: int = 256,
#     ) -> None:
#         """
#         Predicts masks given an image and prompt embeddings, using a
#         transformer architecture.

#         Arguments:
#           transformer_dim (int): the channel dimension of the transformer
#           transformer (nn.Module): the transformer used to predict masks
#           num_multimask_outputs (int): the number of masks to predict
#             when disambiguating masks
#           activation (nn.Module): the type of activation to use when
#             upscaling masks
#           iou_head_depth (int): the depth of the MLP used to predict
#             mask quality
#           iou_head_hidden_dim (int): the hidden dimension of the MLP
#             used to predict mask quality
#         """
#         super().__init__()
#         self.transformer_dim = transformer_dim
#         self.transformer = transformer

#         self.num_multimask_outputs = num_multimask_outputs

#         self.iou_token = nn.Embedding(1, transformer_dim)
#         self.num_mask_tokens = num_multimask_outputs + 1
#         self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
#             LayerNorm2d(transformer_dim // 4),
#             activation(),
#             nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
#             activation(),
#         )
#         self.output_hypernetworks_mlps = nn.ModuleList(
#             [
#                 MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
#                 for i in range(self.num_mask_tokens)
#             ]
#         )

#         self.iou_prediction_head = MLP(
#             transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
#         )  #256 256 4 3

#     def forward(
#         self,
#         image_embeddings: torch.Tensor,   #[B, 256, 64, 64]
#         image_pe: torch.Tensor,           #[1, 256, 64, 64]
#         sparse_prompt_embeddings: torch.Tensor, #[B, 3, 256]
#         dense_prompt_embeddings: torch.Tensor,  #[B, 256, 64, 64]
#         multimask_output: bool,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Predict masks given image and prompt embeddings.

#         Arguments:
#           image_embeddings (torch.Tensor): the embeddings from the image encoder
#           image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
#           sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
#           dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
#           multimask_output (bool): Whether to return multiple masks or a single
#             mask.

#         Returns:
#           torch.Tensor: batched predicted masks
#           torch.Tensor: batched predictions of mask quality
#         """

#         masks, iou_pred = self.predict_masks(
#             image_embeddings=image_embeddings,
#             image_pe=image_pe,
#             sparse_prompt_embeddings=sparse_prompt_embeddings,
#             dense_prompt_embeddings=dense_prompt_embeddings,
#         )

#         # Select the correct mask or masks for output
#         if multimask_output:
#             mask_slice = slice(1, None)
#         else:
#             mask_slice = slice(0, 1)
#         masks = masks[:, mask_slice, :, :]
#         iou_pred = iou_pred[:, mask_slice]

#         # Prepare output
#         return masks, iou_pred

#     def predict_masks(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         sparse_prompt_embeddings: torch.Tensor,
#         dense_prompt_embeddings: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Predicts masks. See 'forward' for more details."""
#         # Concatenate output tokens

#         output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)  #iou_token:[1,256]  mask_tokens:[4,256]
#         output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
#         tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

#         # Expand per-image data in batch direction to be per-mask
#         # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
#         src = image_embeddings
#         src = src + dense_prompt_embeddings
#         pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
#         b, c, h, w = src.shape

#         # Run the transformer
#         hs, src = self.transformer(src, pos_src, tokens)
#         iou_token_out = hs[:, 0, :]
#         mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

#         # Upscale mask embeddings and predict masks using the mask tokens
#         src = src.transpose(1, 2).view(b, c, h, w)
#         upscaled_embedding = self.output_upscaling(src)
#         hyper_in_list: List[torch.Tensor] = []
#         for i in range(self.num_mask_tokens):
#             hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
#         hyper_in = torch.stack(hyper_in_list, dim=1)  #[1,4,32]

#         b, c, h, w = upscaled_embedding.shape  #[1, 32, 256, 256]
#         masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

#         # Generate mask quality predictions
#         iou_pred = self.iou_prediction_head(iou_token_out)

#         return masks, iou_pred


# # Lightly adapted from
# # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_layers: int,
#         sigmoid_output: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#         self.sigmoid_output = sigmoid_output
#         self.relu = nn.ReLU(inplace=False)
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#             # x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x) #源码
#             if i < self.num_layers - 1:
#                 x = F.relu(layer(x))
#             else:
#                 x = layer(x)

#         if self.sigmoid_output:
#             x = F.sigmoid(x)
#         return x