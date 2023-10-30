import torch
import torch.nn as nn


class SimilarityNet(nn.Module):
    def __init__(self, in_features: int, hidden_units: int):
        super().__init__()

        self.sequential = nn.Sequential(
                nn.BatchNorm1d(num_features=in_features),
                nn.Linear(in_features=in_features, out_features=hidden_units, bias=True),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=hidden_units),
                nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=True),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units, out_features=2, bias=True)
        )

    def forward(self, x):
        return self.sequential(x)


class SimilarityTransformer(nn.Module):
    def __init__(self, in_features: int, d_model: int):
        super(SimilarityTransformer, self).__init__()

        self.embedder = nn.Linear(in_features=in_features, out_features=d_model, bias=True)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                                              nhead=8,
                                                              dim_feedforward=256,
                                                              batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder,
                                             num_layers=1)
        self.scorer = nn.Linear(in_features=d_model, out_features=2, bias=True)

        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.scorer(self.dropout_2(self.encoder(self.dropout_1(self.embedder(x)))))
