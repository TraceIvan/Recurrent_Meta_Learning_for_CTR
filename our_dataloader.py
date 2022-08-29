import numpy as np
import pickle
import random
random.seed(1234)

class our_Data_loader():
    def __init__(self,dataset,data,user_dict,item_dict,batch_size):
        self.dataset=dataset
        self.data_seq=data
        self.user_dict=user_dict
        self.item_dict=item_dict
        random.shuffle(self.data_seq)
        self.batch_size=batch_size
        self.epoch_size = len(self.data_seq) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data_seq):
            self.epoch_size += 1
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.epoch_size:
            raise StopIteration

        batch_seq=self.data_seq[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
        self.idx += 1

        cur_batch_len=len(batch_seq)

        users=[]
        users_hists=[]
        user_hists_y=[]
        users_hists_len=[]
        users_hists_targets=[]
        users_hists_targets_y=[]

        for cur_seq in batch_seq:
            user_id=cur_seq[0]
            hists=cur_seq[1]
            target_id=cur_seq[2]
            target_y=cur_seq[3]
            hists_y=cur_seq[4]
            user_hists_y.append(hists_y)
            user_info=[user_id]
            user_info.extend(self.user_dict[user_id])#[u_id,gender,age,occupation]
            users.append(user_info)
            if self.dataset=='ml-1m':
                hists_info=[]
                for cur_hist_id in hists:
                    cur_hist=self.item_dict[cur_hist_id]#[rate,year,genre,director]
                    hists_info.append([cur_hist_id,cur_hist[0],cur_hist[1],cur_hist[2][-1],cur_hist[3][-1]])
                if len(hists)==0:
                    hists_info.append([0,0,0,0,0])
                users_hists.append(hists_info)
                users_hists_len.append(len(hists))
                target_info=self.item_dict[target_id]
                users_hists_targets.append([target_id,target_info[0],target_info[1],target_info[2][-1],target_info[3][-1]])
            elif self.dataset=='bookcrossing':
                hists_info = []
                for cur_hist_id in hists:
                    cur_hist = self.item_dict[cur_hist_id]  # ['author','year','publisher']
                    hists_info.append([cur_hist_id, cur_hist[0], cur_hist[1], cur_hist[2]])
                if len(hists) == 0:
                    hists_info.append([0, 0, 0, 0])
                users_hists.append(hists_info)
                users_hists_len.append(len(hists))
                target_info = self.item_dict[target_id]
                users_hists_targets.append(
                    [target_id, target_info[0], target_info[1], target_info[2]])
            elif self.dataset=='avazu':
                hists_info = []
                for cur_hist_id in hists:
                    cur_hist = self.item_dict[cur_hist_id]  # [c1,c14,c15,c16]
                    hists_info.append([cur_hist_id, cur_hist[0], cur_hist[1], cur_hist[2],cur_hist[3]])
                if len(hists) == 0:
                    hists_info.append([0, 0, 0, 0,0])
                users_hists.append(hists_info)
                users_hists_len.append(len(hists))
                target_info = self.item_dict[target_id]
                users_hists_targets.append(
                    [target_id, target_info[0], target_info[1], target_info[2],target_info[3]])
            users_hists_targets_y.append(target_y)

        max_hist_len=max(users_hists_len)
        if self.dataset == 'ml-1m':
            item_feature_size=5
        elif self.dataset == 'bookcrossing':
            item_feature_size = 4
        elif self.dataset == 'avazu':
            item_feature_size = 5
        new_hist_info=np.zeros([cur_batch_len,max_hist_len,item_feature_size],np.int64)
        new_hist_y=np.zeros([cur_batch_len,max_hist_len],np.int64)
        for cur_idx in range(cur_batch_len):
            new_hist_info[cur_idx,:users_hists_len[cur_idx],:]=users_hists[cur_idx]
            new_hist_y[cur_idx,:users_hists_len[cur_idx]]=user_hists_y[cur_idx]


        return self.idx,(users,users_hists_targets,users_hists_targets_y,new_hist_info,users_hists_len,new_hist_y)




