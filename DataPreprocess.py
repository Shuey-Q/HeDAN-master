import os
import dgl
import random
import torch
import pandas as pd
import pickle as pkl
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import train_test_split



PAD = 0
UNK = 2
BOS = 3
EOS = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

class Options(object):

    def __init__(self, DATA_DIR , OUT_DIR ):
        self.social_network_data = DATA_DIR + 'edges.txt'
        self.u2idx_dict = DATA_DIR + 'u2idx.pickle'
        self.idx2u_dict = DATA_DIR + 'idx2u.pickle'
        self.diffusion_network_data = DATA_DIR + 'cascade.txt'
        self.all_data = OUT_DIR + 'cascade.txt'
        self.train_data = OUT_DIR + 'cascadetrain.txt'
        self.valid_data = OUT_DIR + 'cascadevalid.txt'
        self.test_data = OUT_DIR + 'cascadetest.txt'

        self.out_social_graph = OUT_DIR + 'social_graph.pickle'
        self.out_diffusion_graph_CSV = OUT_DIR + 'diffusion_graph.csv'
        self.out_message_graph_CSV = OUT_DIR + "message_graph.csv"
        self.out_final_graph_CSV = OUT_DIR + "final_graph.csv"
        self.out_label = OUT_DIR + 'label.csv'
        #self.out_new_edges = OUT_DIR + 'new_edges.csv'


def LoadRelationGraph(DATA_DIR , OUT_DIR ):
    edges_list = []
    if os.path.exists(DATA_DIR + "user_network.txt"):
        with open(DATA_DIR + "user_network.txt", 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edges.split('\t') for edges in relation_list]
            for nodes in relation_list:
                src_node = int(nodes[0])
                if src_node < 2000000:
                    edges_list += [[src_node, int(dst_node)] for dst_node in nodes if dst_node != '' if int(dst_node) < 2000000]

    else:
        return []
    edges_array = np.array(edges_list)
    return edges_array
#edges_array = LoadRelationGraph()

def Loadweibo_id_dict(DATA_DIR ):
    weibo_id_dict = {}
    with open(DATA_DIR + "weibo_id_dict.txt", 'r') as handle:
        weibo_list = handle.read().strip().split("\n")
        weibo_list = [weibo.strip().split("\t") for weibo in weibo_list]
        _num_weibo = len(weibo_list)
        for weibo in weibo_list:
            weibo_id_dict[weibo[0]] = int(weibo[1])
    return weibo_id_dict, _num_weibo

def LoadDynamicDiffusionGraph(DATA_DIR , OUT_DIR , data_name , interval ):
    options = Options(DATA_DIR, OUT_DIR)
    if not os.path.exists(OUT_DIR + data_name + "_message_graph_"+str(interval)+"interval.csv") or not os.path.exists(OUT_DIR + data_name+ "_diffusion_graph_"+str(interval)+"interval.csv"):
        print("make diffusion graph and message graph")
        weibo_id_dict,_ = Loadweibo_id_dict()
        with open(DATA_DIR +"trainRepost.txt", 'r') as handle:
            cascade_list = handle.read().strip().split("\n")
            cascade_list = [chunk.strip().split("\x01") for chunk in cascade_list if len(chunk.strip()) >= 4]
            message_list = [[weibo_id_dict[chunk[0]], int(chunk[1]), int(chunk[2]), int(chunk[3])] for chunk in cascade_list if chunk[0] in weibo_id_dict and int(chunk[3]) < interval*3600 and int(chunk[1]) < 2000000 and int(chunk[2]) < 2000000]
        message_array = np.array(message_list)
        message_graph_pd = pd.DataFrame({"messageid":message_array[:,0], "latterid":message_array[:,2], "timestamp":message_array[:,3]})
        diffusion_graph_pd = pd.DataFrame({"formerid":message_array[:,1], "latterid":message_array[:,2], "timestamp":message_array[:,3]})
        #message_graph_pd = message_graph_pd.sort_values(by="timestamp")
        #diffusion_graph_pd = diffusion_graph_pd.sort_values(by="timestamp")
        message_graph_pd.to_csv(OUT_DIR + data_name + "_message_graph_" + str(interval) + "interval.csv", index=False)
        diffusion_graph_pd.to_csv(OUT_DIR + data_name + "_diffusion_graph_" + str(interval) + "interval.csv",
                                  index=False)

    else:
        print("find diffusion graph and message graph")
        diffusion_graph_pd = pd.read_csv(OUT_DIR + data_name+ "_diffusion_graph_"+str(interval)+"interval.csv")
        message_graph_pd = pd.read_csv(OUT_DIR + data_name+ "_message_graph_"+str(interval)+"interval.csv")
    return diffusion_graph_pd, message_graph_pd
#LoadDynamicDiffusionGraph()
def MakeGlobalGraph():
    print("make social graph")
    social_edge_array = LoadRelationGraph()
    print("friendship edges: {}".format(len(social_edge_array)))
    print("make social graph")
    diffusion_graph_pd,  message_graph_pd = LoadDynamicDiffusionGraph()
    print("interaction edges: {}".format(len(diffusion_graph_pd)))
    print("interest edges : {}".format(len(message_graph_pd)))

    graph = dgl.heterograph({
        ('user', 'follow', 'user'): (
            torch.LongTensor(social_edge_array[:,0]), torch.LongTensor(social_edge_array[:,1])),
        #('message','interested','user'): (torch.LongTensor(message_graph_pd.messageid),torch.LongTensor(message_graph_pd.vid)),
        ('user', 'retweet', 'user'): (torch.LongTensor(diffusion_graph_pd["formerid"]),
                                     torch.LongTensor(diffusion_graph_pd["latterid"])),
        ('user', 'interest', 'message'): (
            torch.LongTensor(message_graph_pd.latterid), torch.LongTensor(message_graph_pd.messageid)),
    })


    graph.edges['interested'].data['time'] = torch.LongTensor(diffusion_graph_pd.time)
    graph.edges['reweet'].data['time'] = torch.LongTensor(diffusion_graph_pd.time)
    graph.edges['reweet'].data['order'] = torch.LongTensor(diffusion_graph_pd.order)



    return graph

def LoadCascade(DATA_DIR , OUT_DIR , data_name , interval ):
    options = Options(DATA_DIR, OUT_DIR)
    if not os.path.exists(OUT_DIR + "cascade.txt"):
        print("make cascade")
        weibo_id_dict,_num_weibo = Loadweibo_id_dict()
        x = [None] * _num_weibo
        y = [None] * _num_weibo
        with open(DATA_DIR + 'weibo_profile.txt', 'r') as handle:
            line_list = handle.read().strip().split("\n")
            num_weibo = len(line_list)
            for line in line_list:
                line = line.split("\t")
                if line[0] in weibo_id_dict and int(line[1]) < 2000000:
                    x[weibo_id_dict[line[0]]] = list()
                    x[weibo_id_dict[line[0]]].append([weibo_id_dict[line[0]]])
                    x[weibo_id_dict[line[0]]].append((0, int(line[1])))
                    y[weibo_id_dict[line[0]]] = set()
                    y[weibo_id_dict[line[0]]].add(int(line[1]))
        with open(DATA_DIR + 'trainRepost.txt', "r") as handle:
            cascades_list = handle.read().strip().split("\n")
            for cascades in cascades_list:
                cascades = cascades.split("\x01")
                if cascades[0] in weibo_id_dict:
                    if x[weibo_id_dict[cascades[0]]] is None:
                        x[weibo_id_dict[cascades[0]]] = list()
                        x[weibo_id_dict[cascades[0]]].append([weibo_id_dict[cascades[0]]])

                    if y[weibo_id_dict[cascades[0]]] is None:
                        y[weibo_id_dict[cascades[0]]] = set()
                    if len(cascades) >= 3 and int(cascades[2])< 2000000:
                        y[weibo_id_dict[cascades[0]]].add(int(cascades[2]))
                        x[weibo_id_dict[cascades[0]]].append([int(cascades[2]), int(cascades[3])])

        train_x, val_test_x, train_y, val_test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=42)
        out = open(options.train_data, 'w')
        str1 = ','
        str2 = '\t'
        for line in train_x:
            if line is None:
                continue
            t = str()
            for chunk in line:
                t += str1.join('%s' %a for a in chunk)+str2
            t += '\n'
            out.writelines(t)
            out.flush()
        out.close()
        print("train cascades : {}".format(len(train_x)))
        out = open(options.valid_data, 'w')
        str1 = ','
        str2 = '\t'
        for line in val_x:
            if line is None:
                continue
            t = str()
            for chunk in line:
                t += str1.join('%s' % a for a in chunk) + str2
            t += '\n'
            out.writelines(t)
            out.flush()
        out.close()
        print("valid cascades : {}".format(len(val_x)))
        out = open(options.test_data, 'w')
        str1 = ','
        str2 = '\t'
        for line in test_x:
            if line is None:
                continue
            t = str()
            for chunk in line:
                t += str1.join('%s' % a for a in chunk) + str2
            t += '\n'
            out.writelines(t)
            out.flush()
        out.close()
        print("test cascades : {}".format(len(test_x)))
    return train_x, val_x, test_x
#train_x, val_x, test_x = LoadCascade(DATA_DIR = in_path, OUT_DIR = out_path, data_name = data_name, interval = interval)

class DataConstruct(object):
    ''' For data iteration '''

    def __init__(
            self, data_path, data=0, load_dict=True, cuda=False, batch_size=64, shuffle=True, test=False,
            with_EOS=True, interval = interval, data_name = data_name):  # data = 0 for train, 1 for valid, 2 for test
        self.options = Options(data_path)
        self.options.batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self.data = data
        self.test = test
        self.with_EOS = with_EOS
        self.interval = interval
        self.data_name = data_name
        """
        if not load_dict:
            self._buildIndex()
            with open(self.options.u2idx_dict, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2u_dict, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)
        """

        self._train_cascades_timestamp,self._max_timestamp = self._readFromFileTimestamp(self.options.train_data, self.interval, self.data_name)
        self._valid_cascades_timestamp,self._max_timestamp = self._readFromFileTimestamp(self.options.valid_data, self.interval, self.data_name)
        self._test_cascades_timestamp, self._max_timestamp = self._readFromFileTimestamp(self.options.test_data, self.interval, self.data_name)

        self._train_cascades, self._train_label, self._train_index = self._readFromFile(self.options.train_data, self.interval, self.data_name)
        self._valid_cascades, self._valid_label, self._valid_index = self._readFromFile(self.options.valid_data, self.interval, self.data_name)
        self._test_cascades, self._test_label, self._test_index = self._readFromFile(self.options.test_data, self.interval, self.data_name)



        self.train_size = len(self._train_cascades)
        self.valid_size = len(self._valid_cascades)
        self.test_size = len(self._test_cascades)

        #self._valid_index += self.train_size
        #self._test_index += (self.train_size + self.valid_size)

        self.cuda = cuda

        if self.data == 0:
            self._n_batch = int(np.ceil(len(self._train_cascades) / batch_size))
        elif self.data == 1:
            self._n_batch = int(np.ceil(len(self._valid_cascades) / batch_size))
        else:
            self._n_batch = int(np.ceil(len(self._test_cascades) / batch_size))

        self._batch_size = self.options.batch_size

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            random_seed_int = random.randint(0, 1000)
            random.seed(random_seed_int)
            random.shuffle(self._train_cascades)
            random.seed(random_seed_int)
            random.shuffle(self._train_cascades_timestamp)



    def _buildIndex(self):
        # compute an index of the users that appear at least once in the training and testing cascades.
        opts = self.options

        train_user_set = set()
        valid_user_set = set()
        test_user_set = set()

        lineid = 0
        for line in open(opts.train_data):
            lineid += 1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                train_user_set.add(user)

        for line in open(opts.valid_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                valid_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set | valid_user_set | test_user_set

        pos = 0
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        pos += 1

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set) + 2
        self.user_size = len(user_set) + 2
        print("user_size : %d" % (opts.user_size))

    def _readNet(self, filename):
        adj_list = [[], [], []]
        n_edges = 0
        # add self edges
        for i in range(self.user_size):
            adj_list[0].append(i)
            adj_list[1].append(i)
            adj_list[2].append(1)
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            n_edges += 1
            adj_list[0].append(self._u2idx[nodes[0]])
            adj_list[1].append(self._u2idx[nodes[1]])
            adj_list[2].append(1)  # weight
        # print('edge:', n_edges/2)
        return adj_list

    def _readNet_dict_list(self, filename):
        adj_list = {}
        # add self edges
        for i in range(self.user_size):
            adj_list.setdefault(i, [i])  # [i] or []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            adj_list[self._u2idx[nodes[0]]].append(self._u2idx[nodes[1]])
            adj_list[self._u2idx[nodes[1]]].append(self._u2idx[nodes[0]])
        return adj_list

    def _load_ne(self, filename, dim):
        embed_file = open(filename, 'r')
        line = embed_file.readline().strip()
        dim = int(line.split()[1])
        embeds = np.zeros((self.user_size, dim))
        for line in embed_file.readlines():
            line = line.strip().split()
            embeds[self._u2idx[line[0]], :] = np.array(line[1:])
        return embeds

    def _readFromFile(self, filename, interval, data_name):
        """read all cascade from training or testing files. """
        total_len = 0
        t_cascades = []
        t_len = []
        t_index = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split('\t')
            messageid = int(chunks[0])
            caslist = [chunk.split(',') for chunk in chunks[1:]]


            if data_name == "Twitter":
                caslist = [[x[0], x[1] + '0' * (10 - len(x[1]))] if len(x[1]) < 10 else [x[0], x[1]] for x in caslist]
                max_stamp = int(caslist[0][1]) + 3600 * interval  # hours
            elif data_name == "Memetracker":
                caslist = [[x[0], x[1] + '0' * (13 - len(x[1]))] if len(x[1]) < 13 else [x[0], x[1]] for x in caslist]
                max_stamp = int(caslist[0][1]) + 1000 * 3600 * 24 * 7 * interval  # weeks
            else:
                caslist = [[x[0], x[1]] for x in caslist]
                max_stamp = int(caslist[0][1]) + 3600 * 24 * 7 * interval  # weeks

            userlist = [int(x[0]) for x in caslist if int(x[1]) <= interval * 3600]
            oldlist = [int(x[0]) for x in caslist ]

            for chunk in chunks:
                # try:
                user, timestamp = chunk.split(',')
                # except:
                #     print(chunk)
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])


            if len(userlist) >= 1:
                if self.with_EOS:
                    userlist.append(EOS)
                t_cascades.append(userlist)
                t_len.append(np.log1p(len(oldlist)))
                t_index.append(messageid)
        return t_cascades, t_len, t_index

    def _readFromFileTimestamp(self, filename, interval, data_name):
        """read all cascade from training or testing files. """
        t_cascades = []
        maxstamplist = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue

            chunks = line.strip().split('\t')
            caslist = [chunk.split(',') for chunk in chunks[1:]]

            if data_name == "Twitter":
                caslist = [[x[0], x[1] + '0' * (10 - len(x[1]))] if len(x[1]) < 10 else [x[0], x[1]] for x in caslist]
                max_stamp = int(caslist[0][1]) + 3600 * interval # hours
            elif data_name == "Memetracker":
                caslist = [[x[0], x[1] + '0' * (13 - len(x[1]))] if len(x[1]) < 13 else [x[0], x[1]] for x in caslist]
                max_stamp = int(caslist[0][1]) + 1000 * 3600 * 24 * 7 * interval  # weeks
            else:
                caslist = [[x[0], x[1]] for x in caslist]
                max_stamp = int(caslist[0][1]) + 3600 * 24 * 7 * interval  # weeks

            timestamplist = [int(x[1]) for x in caslist if int(x[1])<= interval*3600]



            for chunk in chunks:
                # try:
                user, timestamp = chunk.split(',')
                if data_name == "Twitter":
                    if len(timestamp)<10:
                        timestamp += '0'*(10-len(timestamp))
                timestamp = int(timestamp)
                # timestamp = timestamp // (60 * 60 * 24)
                # except:
                #     print(chunk)
                if user in self._u2idx:
                    timestamplist.append(timestamp)


            if len(timestamplist) >=1:
                if self.with_EOS:
                    timestamplist.append(EOS)
                t_cascades.append(timestamplist)
                maxstamplist.append(interval*3600)
        return t_cascades, maxstamplist

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [PAD] * (max_len - len(inst))
                for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor


        if self._iter_count < self._n_batch:

            batch_idx = self._iter_count

            self._iter_count += 1


            start_idx = batch_idx * self._batch_size

            end_idx = (batch_idx + 1) * self._batch_size


            if self.data == 0:

                seq_insts = self._train_cascades[start_idx:end_idx]

                seq_timestamp = self._train_cascades_timestamp[start_idx:end_idx]
                seq_label = self._train_label[start_idx:end_idx]
                seq_index = self._train_index[start_idx:end_idx]


            elif self.data == 1:

                seq_insts = self._valid_cascades[start_idx:end_idx]
                seq_timestamp = self._valid_cascades[start_idx:end_idx]
                seq_label = self._valid_label[start_idx:end_idx]
                seq_index = self._valid_index[start_idx:end_idx]
            else:
                seq_insts = self._test_cascades[start_idx:end_idx]
                seq_timestamp = self._test_cascades_timestamp[start_idx:end_idx]
                seq_label = self._test_label[start_idx:end_idx]
                seq_index = self._test_index[start_idx:end_idx]

            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            seq_label = torch.FloatTensor(seq_label)
            seq_index = torch.LongTensor(seq_index)
            if self.cuda:
                seq_label = seq_label.cuda()
                seq_index = seq_index.cuda()

            return seq_data, seq_data_timestamp, seq_label, seq_index
        else:

            if self._need_shuffle:
                random.shuffle(self._train_cascades)
                # random.shuffle(self._test_cascades)

            self._iter_count = 0
            raise StopIteration()




