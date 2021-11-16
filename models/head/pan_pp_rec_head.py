import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#for generate dict
import operator
from dict_trie import Trie
from editdistance import eval

from ..loss import acc
from ..post_processing import BeamSearch


class PAN_PP_RecHead(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 voc,
                 char2id,
                 id2char,
                 lang=None,
                 beam_size=1,
                 feature_size=(8, 32)):
        super(PAN_PP_RecHead, self).__init__()
        self.char2id = char2id
        self.id2char = id2char
        self.beam_size = beam_size

        self.conv = nn.Conv2d(input_dim,
                              hidden_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.feature_size = feature_size
        self.encoder = Encoder(hidden_dim, voc, char2id, id2char)
        self.decoder = Decoder(hidden_dim, hidden_dim, 2, voc, char2id,
                               id2char)

        self.dictionary = open("vn_dictionary.txt").read().replace("\n\n", "\n").split("\n")
        self.trie = Trie(self.dictionary)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, output_size):
        return F.upsample(x, size=output_size, mode='bilinear')

    def extract_feature(self,
                        f,
                        output_size,
                        instance,
                        bboxes,
                        gt_words=None,
                        word_masks=None,
                        unique_labels=None):
        x = self.conv(f)
        x = self.relu(self.bn(x))
        x = self._upsample(x, output_size)

        x_crops = []
        if gt_words is not None:
            words = []

        batch_size, _, H, W = x.size()
        pad_scale = 1
        pad = x.new_tensor([-1, -1, 1, 1], dtype=torch.long) * pad_scale
        if self.training:
            offset = x.new_tensor(np.random.randint(-pad_scale, pad_scale + 1,
                                                    bboxes.size()),
                                  dtype=torch.long)
            pad = pad + offset

        bboxes = bboxes + pad
        bboxes[:, :, (0, 2)] = bboxes[:, :, (0, 2)].clamp(0, H)
        bboxes[:, :, (1, 3)] = bboxes[:, :, (1, 3)].clamp(0, W)

        #x.size(0) is batch_size
        for i in range(x.size(0)):
            instance_ = instance[i:i + 1]
            if unique_labels is None:
                unique_labels_, _ = torch.unique(instance_,
                                                 sorted=True,
                                                 return_inverse=True)
            else:
                unique_labels_ = unique_labels[i]
            x_ = x[i]
            if gt_words is not None:
                gt_words_ = gt_words[i]
            if word_masks is not None:
                word_masks_ = word_masks[i]
            bboxes_ = bboxes[i]

            for label in unique_labels_:
                if label == 0:
                    continue
                if word_masks is not None and word_masks_[label] == 0:
                    continue
                t, l, b, r = bboxes_[label]

                mask = (instance_[:, t:b, l:r] == label).float()
                mask = F.max_pool2d(mask.unsqueeze(0),
                                    kernel_size=(3, 3),
                                    stride=1,
                                    padding=1)[0]

                if torch.sum(mask) == 0:
                    continue
                x_crop = x_[:, t:b, l:r] * mask
                _, h, w = x_crop.size()
                if h > w * 1.5:
                    x_crop = x_crop.transpose(1, 2)
                x_crop = F.interpolate(x_crop.unsqueeze(0),
                                       self.feature_size,
                                       mode='bilinear')
                x_crops.append(x_crop)
                if gt_words is not None:
                    words.append(gt_words_[label])
        if len(x_crops) == 0:
            return None, None
        x_crops = torch.cat(x_crops)
        if gt_words is not None:
            words = torch.stack(words)
        else:
            words = None
        return x_crops, words

    def decode(self,rec):
        s = ""
        for c in rec:
            c = int(c)
            if c < 104:
                s += self.id2char[c]
            # elif c == 104:
            #     s += u'口'

        return s
    
    def encode(self,s):
        word=[]
        for c in s:
            word.append(int(self.char2id[c]))

        while len(word) < 32:
            word.append(104)
        
        word = word[:32]
        return word


    def generate_dict(self,targets,maxlen=1):
        
        if self.training:
            #pass
            target_candidates = []
            distance_candidates = []
            #beziers = [p.beziers for p in targets]
            #targets = torch.cat([x for x in targets], dim=0)
            for target in targets:
                rec = target.cpu().detach().numpy()
                rec = self.decode(rec)

                # candidates = {}
                # candidates[rec] = 0
                # for word in self.dictionary:
                #     candidates[word] = eval(rec, word)
                # candidates = sorted(candidates.items(), key=operator.itemgetter(1))[:10]

                candidates_list = list(self.trie.all_levenshtein(rec, 1))
                candidates_list.append(rec)
                candidates_list = list(set(candidates_list))
                candidates = {}
                for candidate in candidates_list:
                    candidates[candidate] = eval(rec, candidate)
                candidates = sorted(candidates.items(), key=operator.itemgetter(1))
                dist_sharp = eval("###", rec)
                while len(candidates) < 10:
                    candidates.append(("###", dist_sharp))
                candidates = candidates[:10]

                candidates_encoded = []
                distance_can = []
                for can in candidates:
                    word = self.encode(can[0])
                    
                    candidates_encoded.append(word)
                    distance_can.append(1 / (can[1] + 0.1))
                # distance_can = softmax(distance_can)

                distance_candidates.append(distance_can)
                target_candidates.append(candidates_encoded)

            distance_candidates = torch.Tensor(distance_candidates)
            # distance_candidates = torch.sum(distance_candidates, dim=0)
            # distance_candidates = nn.functional.log_softmax(distance_candidates, dim=0)

            target_candidates = torch.LongTensor(target_candidates)
            # distance_candidates = torch.Tensor(distance_candidates).to(device='cuda')
            targets = target_candidates
            targets = targets.permute((1, 0, 2))
            targets = {"targets": targets, "scores": distance_candidates}

            return targets
        else:
            target_candidates = []

            distance_candidates = []
            for target in targets:
                rec = target.cpu().detach().numpy()
                rec = self.decode(rec)
                candidates = {}
                for word in self.dictionary:
                    candidates[word] = eval(rec, word)
                candidates = sorted(candidates.items(), key=operator.itemgetter(1))[: maxlen]
                candidates_encoded = []
                distance_can = []
                for can in candidates:
                    word = self.encode(can[0])
                    candidates_encoded.append(word)

                target_candidates.append(candidates_encoded)

            target_candidates = torch.Tensor(target_candidates).to(device="cuda")
            targets = target_candidates

            return targets
            

    def loss_unit(self, inputs, targets, reduce=True):
        
        loss_total=0 
        scores  = targets["scores"]
        targets = targets["targets"]
        total_acc=0.0
        output = []
        for idx in range(targets.size(0)):
            input= inputs[idx]
            target= targets[idx].to('cuda')

            EPS = 1e-6
            N, L, D = input.size()  #inputs.shape: ( number, 32, max_len_vocab ) || targets.shape: ( number, 32 )
            mask = target != self.char2id['PAD']
            input = input.contiguous().view(-1, D)
            target = target.contiguous().view(-1)
            loss_rec = F.cross_entropy(input, target, reduce=False)
            loss_rec = loss_rec.view(N, L)
            
            loss_rec = torch.sum(loss_rec * mask.float(),
                                dim=1) / (torch.sum(mask.float(), dim=1) + EPS)
            acc_rec = acc(torch.argmax(input, dim=1).view(N, L),
                        target.view(N, L),
                        mask,
                        reduce=False)
            if reduce:
                loss_rec = torch.mean(loss_rec)*32  # [valid]
                acc_rec = torch.mean(acc_rec)
            
            output.append(1/loss_rec)

            if idx==0:
                loss_total+= loss_rec
                total_acc+=  acc_rec
            

        
        output = torch.stack(output)
        # [10, n, ]
        #print("list loss: ", output)
        
        #output = torch.Tensor(output).to(device="cuda")
        output = nn.functional.softmax(output, dim=0)
        #print("dist_BF:", scores)
        #scores = [scores.T]*output.size(1)
        #scores = torch.stack(scores).T
        scores = scores.permute(1,0)
        scores = scores.to("cuda")
        scores = torch.mean(scores, dim=0)
        scores = nn.functional.softmax(scores, dim=0)
        #print("dist:", scores)
        
        output = torch.unsqueeze(output, dim=0).to(device="cuda")
        scores = torch.unsqueeze(scores, dim=0).to(device="cuda")
        temp = nn.KLDivLoss(reduce=False)(output, scores)
        
        #print("KLDLoss: ", temp.shape, temp)

        loss_total += torch.squeeze(torch.mean(temp/20, dim=1))
        #print("total", loss_total)
        #print(K)
        losses = {'loss_rec': loss_total, 'acc_rec': total_acc}
        return losses


    def loss(self, input, target, reduce=True):

        EPS = 1e-6
        N, L, D = input.size()  #inputs.shape: ( number, 32, max_len_vocab ) || targets.shape: ( number, 32 )
        mask = target != self.char2id['PAD']
        input = input.contiguous().view(-1, D)
        target = target.contiguous().view(-1)
        loss_rec = F.cross_entropy(input, target, reduce=False)
        loss_rec = loss_rec.view(N, L)
        loss_rec = torch.sum(loss_rec * mask.float(),
                             dim=1) / (torch.sum(mask.float(), dim=1) + EPS)
        acc_rec = acc(torch.argmax(input, dim=1).view(N, L),
                      target.view(N, L),
                      mask,
                      reduce=False)
        if reduce:
            loss_rec = torch.mean(loss_rec)  # [valid]
            acc_rec = torch.mean(acc_rec)
        losses = {'loss_rec': loss_rec, 'acc_rec': acc_rec}

        return losses





    def forward(self, x, targets=None):
        holistic_feature = self.encoder(x)

        if self.training:
            out=[]
            
            targets_candidates= targets['targets'] 
            #print('FULLdata', targets_candidates.shape)
            for i in range(targets_candidates.size(0)):
                target= targets_candidates[i]
                #print("forwardHEAD", target.shape)
                out.append(
                                self.decoder(x, holistic_feature, target)
                )
            return out
        else:
            if self.beam_size <= 1:
                return self.decoder.forward_test(x, holistic_feature)
            else:
                return self.decoder.beam_search(x,
                                                holistic_feature,
                                                beam_size=self.beam_size)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, voc, char2id, id2char):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = len(voc)
        self.START_TOKEN = char2id['EOS']
        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.att = MultiHeadAttentionLayer(self.hidden_dim, 8)

    def forward(self, x):
        batch_size, feature_dim, H, W = x.size()
        x_flatten = x.view(batch_size, feature_dim, H * W).permute(0, 2, 1)
        st = x.new_full((batch_size, ), self.START_TOKEN, dtype=torch.long)
        emb_st = self.emb(st)
        holistic_feature, _ = self.att(emb_st, x_flatten, x_flatten)
        return holistic_feature


class Decoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, voc, char2id,
                 id2char):
        super(Decoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = len(voc)
        self.START_TOKEN = char2id['EOS']
        self.END_TOKEN = char2id['EOS']
        self.NULL_TOKEN = char2id['PAD']
        self.char2id = char2id
        self.id2char = id2char
        self.lstm_u = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm_u.append(nn.LSTMCell(self.hidden_dim, self.hidden_dim))
        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.att = MultiHeadAttentionLayer(self.hidden_dim, 8)
        self.cls = nn.Linear(self.hidden_dim + self.feature_dim,
                             self.vocab_size)

        self.dictionary = open("vn_dictionary.txt").read().replace("\n\n", "\n").split("\n")
        self.trie = Trie(self.dictionary)

    def forward(self, x, holistic_feature, target):
        # print(x.shape, holistic_feature.shape, target.shape)
        target = target.to('cuda')
        batch_size, feature_dim, H, W = x.size()
        x_flatten = x.view(batch_size, feature_dim, H * W).permute(0, 2, 1)

        max_seq_len = target.size(1)
        
        h = []
        for i in range(self.num_layers):
            h.append((x.new_zeros((x.size(0), self.hidden_dim),
                                  dtype=torch.float32),
                      x.new_zeros((x.size(0), self.hidden_dim),
                                  dtype=torch.float32)))

        out = x.new_zeros((x.size(0), max_seq_len + 1, self.vocab_size),
                          dtype=torch.float32)
        for t in range(max_seq_len + 1):
            #print('char: ',t)
            if t == 0:
                xt = holistic_feature
            elif t == 1:
                it = x.new_full((batch_size, ),
                                self.START_TOKEN,
                                dtype=torch.long)
                xt = self.emb(it)
            else:
                it = target[:, t - 2]
                #print("IT", it)
                xt = self.emb(it)

            for i in range(self.num_layers):
                if i == 0:
                    inp = xt
                else:
                    inp = h[i - 1][0]
                h[i] = self.lstm_u[i](inp, h[i])
            ht = h[-1][0]
            out_t, _ = self.att(ht, x_flatten, x_flatten)
            # print(out_t.shape, _.shape)
            out_t = torch.cat((out_t, ht), dim=1)
            # print(out_t.shape)
            # exit()
            out_t = self.cls(out_t)
            out[:, t, :] = out_t
        return out[:, 1:, :]

    def decode(self,rec):
        s = ""
        for c in rec:
            c = int(c)
            if c < 104:
                s += self.id2char[c]
            # elif c == 104:
            #     s += u'口'

        return s
    
    def encode(self, s):
        word=[]
        for c in s:
            word.append(self.char2id[c])

        while len(word) < 32:
            word.append(104)
        
        word = word[:32]
        return word

    def generate_dict(self,targets,maxlen=1):
        
        if self.training:
            pass
            target_candidates = []
            distance_candidates = []
            #beziers = [p.beziers for p in targets]
            targets = torch.cat([x.text for x in targets], dim=0)
            for target in targets:
                rec = target.cpu().detach().numpy()
                rec = self.decode(rec)

                # candidates = {}
                # candidates[rec] = 0
                # for word in self.dictionary:
                #     candidates[word] = eval(rec, word)
                # candidates = sorted(candidates.items(), key=operator.itemgetter(1))[:10]

                candidates_list = list(self.trie.all_levenshtein(rec, 1))
                candidates_list.append(rec)
                candidates_list = list(set(candidates_list))
                candidates = {}
                for candidate in candidates_list:
                    candidates[candidate] = eval(rec, candidate)
                candidates = sorted(candidates.items(), key=operator.itemgetter(1))
                dist_sharp = eval("###", rec)
                while len(candidates) < 10:
                    candidates.append(("###", dist_sharp))
                candidates = candidates[:10]

                candidates_encoded = []
                distance_can = []
                for can in candidates:
                    word = self.encode(can[0])
                    
                    candidates_encoded.append(word)
                    distance_can.append(1 / (can[1] + 0.1))
                # distance_can = softmax(distance_can)

                distance_candidates.append(distance_can)
                target_candidates.append(candidates_encoded)

            distance_candidates = torch.Tensor(distance_candidates).to(device="cuda")
            # distance_candidates = torch.sum(distance_candidates, dim=0)
            # distance_candidates = nn.functional.log_softmax(distance_candidates, dim=0)

            target_candidates = torch.Tensor(target_candidates).to(device="cuda")
            # distance_candidates = torch.Tensor(distance_candidates).to(device='cuda')
            targets = target_candidates
            targets = targets.permute((1, 0, 2))
            targets = {"targets": targets, "scores": distance_candidates}

            return targets
        else:
            target_candidates = []

            #distance_candidates = []
            for target in targets:
                rec = target.cpu().detach().numpy()
                rec = self.decode(rec)
                candidates = {}
                for word in self.dictionary:
                    candidates[word] = eval(rec, word)                                     #len: maxlen(1)
                candidates = sorted(candidates.items(), key=operator.itemgetter(1))[: maxlen]  #dict: [word: distance] 
                candidates_encoded = []
                #candidates_score = []
                #distance_can = []
                for can in candidates:
                    word = self.encode(can[0])  #list | len= 32
                    candidates_encoded.append(word)

                    #candidates_score.append(can[1])   # [n, maxlen]
                                     #word is always correct *not meaning == ground truth
                #candidates_encoded : list | len = 1 or maxlen

                target_candidates.append(candidates_encoded)

            #target_candidates: list[ list(candidates_encoded: list(word: len=32) | len= maxlen ), list, ...]  len = batch_size

            target_candidates = torch.LongTensor(target_candidates)
            #candidates_score  = torch.Tensor(candidates_score)
            
            #target_candidates: Tensor [batch_size, maxlen(1), 32] 

            #targets = target_candidates.permute(1,0,2)
            
            #targets : Tensor [maxlen(1), batch_size, 32]  
            

            return target_candidates

    def to_words(self, seqs, seq_scores=None,decoder_raw=None,n=None,num_candidates=1):

        #seqs: [batch, 32],   seq_scores[batch, 32]
        EPS = 1e-6
        words = []
        word_scores = None
        #seq_scores = None
        if seq_scores is not None:
            word_scores = []

        
        if decoder_raw is not None:
            #decodes =seqs
            targets = self.generate_dict(seqs,num_candidates)

            decodes = torch.zeros((n, 32))
            prob = 1.0
            for i in range(n):
                losses = []
                #decode_candidates = torch.zeros((1, self.attention.max_len))
                target_i = targets[i]
                for j in range(num_candidates):
                    
                    
                    input= decoder_raw[i]
                    target= target_i[j].to('cuda')

                    EPS = 1e-6
                    L, D = input.size()  #inputs.shape: ( number, 32, max_len_vocab ) || targets.shape: ( number, 32 )
                    mask = target != self.char2id['PAD']
                    input = input.contiguous().view(-1, D)
                    target = target.contiguous().view(-1)
                    loss_rec = F.cross_entropy(input, target, reduce=False)
                    loss_rec = loss_rec.view(L)
                    
                    loss_rec = torch.sum(loss_rec * mask.float()
                                        ) / (torch.sum(mask.float()) + EPS)
                    
                    if True:
                        loss_rec = torch.mean(loss_rec)   # [valid]
                        #acc_rec = torch.mean(acc_rec)
                    
                    losses.append(loss_rec.to(device='cpu'))

                min_id = np.argmin(losses)
                decodes[i, :] = targets[i, min_id, :]
                



            
            seqs = decodes
            #seq_scores = None
        
        
        
            
        
        

        for i in range(len(seqs)):
            word = ''
            word_score = 0
            for j, char_id in enumerate(seqs[i]):
                char_id = int(char_id)
                if char_id == self.END_TOKEN:
                    break
                if self.id2char[char_id] in ['PAD', 'UNK']:
                    continue
                word += self.id2char[char_id]
                if seq_scores is not None:
                    word_score += decoder_raw[i, j,char_id]
            words.append(word)
            if seq_scores is not None:
                word_scores.append(word_score / (len(word) + EPS))
        return words, word_scores

        




    def forward_test(self, x, holistic_feature):
        batch_size, feature_dim, H, W = x.size()
        x_flatten = x.view(batch_size, feature_dim, H * W).permute(0, 2, 1)

        h = x.new_zeros(self.num_layers, 2, batch_size, self.hidden_dim)

        max_seq_len = 32
        seq = x.new_full((batch_size, max_seq_len + 1),
                         self.START_TOKEN,
                         dtype=torch.long)
        seq_score = x.new_zeros((batch_size, max_seq_len + 1),
                                dtype=torch.float32) 

        decoder_raw = torch.zeros((batch_size, max_seq_len, 106)).to(x.device)
        
        end = x.new_ones((batch_size, ), dtype=torch.uint8)
        for t in range(max_seq_len + 1):
            if t == 0:
                xt = holistic_feature
            else:
                it = seq[:, t - 1]
                xt = self.emb(it)

            for i in range(self.num_layers):
                if i == 0:
                    inp = xt
                else:
                    inp = h[i - 1, 0]
                h[i, 0], h[i, 1] = self.lstm_u[i](inp, (h[i, 0], h[i, 1]))
            ht = h[-1, 0]
            if t == 0:
                continue
            out_t, _ = self.att(ht, x_flatten, x_flatten)
            out_t = torch.cat((out_t, ht), dim=1) 
            
            out_t = self.cls(out_t)
            decoder_raw[:,t-1,:] = out_t

            score = torch.softmax(out_t, dim=1)
            score, idx = torch.max(score, dim=1)
            seq[:, t] = idx
            seq_score[:, t] = score
            end = end & (idx != self.START_TOKEN)
            if torch.sum(end) == 0:
                break

        words, word_scores = self.to_words(seq[:, 1:], seq_score[:, 1:], decoder_raw,batch_size)

        return words, word_scores

    def beam_search(self, x, holistic_feature, beam_size=2):
        batch_size, c, h, w = x.size()
        x_beam = x.repeat(1, beam_size, 1, 1).view(-1, c, h, w)

        def decode_step(inputs, h, k):
            if len(inputs.shape) == 1:
                inputs = self.emb(inputs)
            for i in range(self.num_layers):
                if i == 0:
                    xt = inputs
                else:
                    xt = h[i - 1, 0]
                h[i, 0], h[i, 1] = self.lstm_u[i](xt, (h[i, 0], h[i, 1]))
            ht = h[-1, 0]
            if ht.size(0) == batch_size:
                out_t = self.att(x, ht)
            else:
                out_t = self.att(x_beam, ht)
            out_t = torch.cat((out_t, ht), -1)
            out_t = torch.softmax(self.cls(out_t), dim=1)
            scores, words = torch.topk(out_t, k, dim=1, sorted=True)

            return words, scores, h

        bs = BeamSearch(decode_step, self.END_TOKEN, beam_size, 32)

        x0 = holistic_feature
        h = x.new_zeros(self.num_layers, 2, batch_size, self.hidden_dim)
        words, scores, h = decode_step(x0, h, 1)
        init_inputs = x.new_full((batch_size, ),
                                 self.START_TOKEN,
                                 dtype=torch.long)
        seqs, seq_scores = bs.beam_search(init_inputs, h)
        words, _ = self.to_words(seqs)
        # print(words)
        # exit()
        return words, seq_scores


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.layer_norm(q)

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        att = torch.matmul(q / self.scale, k.permute(0, 1, 3, 2))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        att = torch.softmax(att, dim=-1)

        out = torch.matmul(self.dropout(att), v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, self.hidden_dim)

        out = self.dropout(self.fc_o(out))

        return out, att
