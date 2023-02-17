import torch
import torch.nn as nn
from model.News_embedding import News_embedding
from model.User_modeling import User_modeling
from utils.metrics import *

class Softmax_BCELoss(nn.Module):
    def __init__(self, config):
        super(Softmax_BCELoss, self).__init__()
        self.config = config
        self.softmax = nn.Softmax(dim=-1)
        self.bceloss = nn.BCELoss()

    def forward(self, predict, truth):
        predict = self.config['trainer']['smooth_lamda'] * predict
        predict = self.softmax(predict)
        loss = self.bceloss(predict, truth)
        return loss

class KREDModel(nn.Module):
    def __init__(self, config, user_history_dict, doc_feature_dict, entity_embedding, relation_embedding, adj_entity,
                 adj_relation, entity_num, position_num, type_num):
        super(KREDModel, self).__init__()
        self.config = config
        self.user_history_dict = user_history_dict
        self.doc_feature_dict = doc_feature_dict
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.entity_num = entity_num
        self.position_num = position_num
        self.type_num = type_num
        self.news_embedding = News_embedding(config, doc_feature_dict, entity_embedding, relation_embedding, adj_entity,
                                             adj_relation, entity_num, position_num, type_num)
        # self.user_modeling = User_modeling(config, user_history_dict, self.config['model']['embedding_dim'], self.config['model']['embedding_dim'], doc_feature_dict, entity_embedding,
        #                                    relation_embedding, adj_entity, adj_relation, entity_num, position_num, type_num)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self.vert_mlp_layer1 = nn.Linear(self.config['model']['embedding_dim'], self.config['model']['layer_dim'])
        self.vert_mlp_layer2 = nn.Linear(self.config['model']['layer_dim'], 17)

    def forward(self, user_features, news_features, task):
        candidate_news_embedding, topk_index = self.news_embedding(news_features)
        predict_vert = self.softmax(self.vert_mlp_layer2(self.relu(self.vert_mlp_layer1(candidate_news_embedding))))
        final_prediction = predict_vert
        
        return final_prediction.squeeze(), topk_index