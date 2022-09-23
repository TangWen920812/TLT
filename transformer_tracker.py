import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformer.backbone import *
from transformer.transformer import *
import transformer.transformer as tt
from torch import nn, einsum
from einops import rearrange
from transformer.data import *
import cv2
import matplotlib.pyplot as plt


def make_nustedtensor(tensor1, ifnusted=True):
    max_size = tt._max_by_axis([list(img.shape) for img in tensor1])
    batch_shape = [len(tensor1)] + max_size
    b, c, d, h, w = batch_shape
    tensor = torch.zeros(batch_shape)
    mask = torch.ones((b, d, h, w))
    for img, pad_img, m in zip(tensor1, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(torch.from_numpy(img))
        m[: img.shape[1], :img.shape[2], :img.shape[3]] = False
    if ifnusted:
        return NestedTensor(tensor.float(), mask)
    else:
        return tensor

class AutoThresModule(nn.Module):
    def __init__(self, channel):
        super(AutoThresModule, self).__init__()
        self.auto_linear = MLP(channel // 2, channel // 2, 1, 2)
        self.project = nn.Conv3d(channel + 1, channel // 2, kernel_size=1)

    def forward(self, feature, mask):
        feature = feature.tensors
        B, C, D, H, W = feature.shape
        mask = F.interpolate(mask, size=(D, H, W))
        feature = self.project(torch.cat((feature, mask), dim=1))
        mask = mask.view(B, 1, -1)
        f = torch.sum(feature.view(B, C, -1) * F.softmax(mask, dim=-1), dim=-1)
        thres = F.sigmoid(self.auto_linear(f))
        return thres

class TransformerTracker(nn.Module):
    def __init__(self, backbone_feature, pos_emb, channel, thres=0.8, fusion_layer=1, istrain=True):
        super(TransformerTracker, self).__init__()
        self.backbone = build_backbone(channel, backbone_feature, pos_emb)
        self.featurefusion = build_featurefusion_network(hidden_dim=backbone_feature[-1], dropout=0.1, nheads=8,
                                                         dim_feedforward=backbone_feature[-1], featurefusion_layers=fusion_layer)
        self.class_embed = nn.Linear(backbone_feature[-1], 1)
        self.bbox_embed = MLP(backbone_feature[-1], backbone_feature[-1], 6, 2)
        self.bbox_reg = nn.Linear(6+3, 6)
        self.query_embed = nn.Embedding(1, backbone_feature[-1])
        self.input_proj = nn.Conv3d(backbone_feature[-1], backbone_feature[-1], kernel_size=1)
        self.thres = thres
        self.istrain = istrain
        # self.auto_thres_module = AutoThresModule(backbone_feature[2])

    def forward(self, moving_img, moving_lesionmsk, fixed_img, fixed_lesionmsk, ifatten=False):
        moving_img = nested_tensor_from_tensor(moving_img)
        fixed_img = nested_tensor_from_tensor(fixed_img)
        # use for batch size is one
        fixed_pos_input = fixed_lesionmsk.cpu().numpy()
        fixed_pos_input = np.array(np.where(fixed_pos_input==fixed_pos_input.max()))[:, 0] / np.array(fixed_pos_input.shape)
        fixed_pos_input = torch.from_numpy(fixed_pos_input[2:]).float().cuda().unsqueeze(0).unsqueeze(1)

        moving_feature, moving_pos = self.backbone(moving_img)
        fixed_feature, fixed_pos = self.backbone(fixed_img)
        # use auto threshold module
        # self.thres = self.auto_thres_module(moving_feature[1], moving_lesionmsk)
        # if not self.istrain: print(self.thres)
        moving_feature, fixed_feature = moving_feature[-1], fixed_feature[-1]
        moving_pos, fixed_pos = moving_pos[-1], fixed_pos[-1]
        moving_feature, moving_msk = moving_feature.decompose()
        fixed_feature, fixed_msk = fixed_feature.decompose()
        moving_feature = self.input_proj(moving_feature)
        fixed_feature = self.input_proj(fixed_feature)
        B, C, D, H, W = moving_feature.shape
        out_feature = moving_feature
        moving_lesionmsk_src = F.interpolate(moving_lesionmsk, size=(D, H, W)).view(B, 1, -1)
        fixed_lesionmsk = F.interpolate(fixed_lesionmsk, size=(D, H, W)).view(B, 1, -1)

        moving_feature = moving_feature.view(B, C, -1)
        fixed_feature = fixed_feature.view(B, C, -1)
        moving_msk = moving_msk.view(B, 1, -1)
        fixed_msk = fixed_msk.view(B, 1, -1)
        moving_pos = moving_pos.contiguous().view(B, C, -1)
        fixed_pos = fixed_pos.contiguous().view(B, C, -1)

        # moving_feature = moving_feature * self.reverse_relu(moving_lesionmsk_src - self.thres)
        moving_lesionmsk = (torch.masked_select(moving_lesionmsk_src, moving_lesionmsk_src>=self.thres))
        moving_feature_cropped = (torch.masked_select(moving_feature, moving_lesionmsk_src>=self.thres))
        moving_msk = (torch.masked_select(moving_msk, moving_lesionmsk_src>=self.thres))
        moving_pos = (torch.masked_select(moving_pos, moving_lesionmsk_src>=self.thres))
        moving_lesionmsk = moving_lesionmsk.view(B, 1, -1)
        moving_feature_cropped = moving_feature_cropped.view(B, C, -1)
        moving_msk = moving_msk.view(B, 1, -1)
        moving_pos = moving_pos.view(B, C, -1)

        atten_mask_moving2fixed = self.make_atten_mask(moving_lesionmsk, fixed_lesionmsk)[0]
        atten_mask_fixed2moving = self.make_atten_mask(fixed_lesionmsk, moving_lesionmsk)[0]
        atten_mask_fixed2fixed = self.make_atten_mask(fixed_lesionmsk, fixed_lesionmsk, src_fixed=True)[0]
        atten_mask_moving2moving = self.make_atten_mask(moving_lesionmsk, moving_lesionmsk, src_fixed=True)[0]

        hs, memory_temp, memory_search, attent_out = self.featurefusion(moving_feature_cropped, moving_msk, fixed_feature, fixed_msk, moving_pos, fixed_pos,
                                atten_mask_moving2moving, atten_mask_moving2fixed, atten_mask_fixed2moving, atten_mask_fixed2fixed)
        outputs_class = self.class_embed(hs)
        coord_feature = (self.bbox_embed(hs) * outputs_class.softmax(dim=-2)).sum(dim=-2)
        outputs_coord = self.bbox_reg(torch.cat((coord_feature, fixed_pos_input), dim=2)).sigmoid()
        if ifatten:
            return outputs_class, outputs_coord, out_feature, attent_out
        else:
            return outputs_class, outputs_coord, out_feature

    def make_atten_mask(self, src_msk, tar_msk, src_fixed=True):
        if src_fixed:
            src_msk_copy = torch.ones_like(src_msk)
            atten_msk = einsum('b k m, b k n -> b m n', src_msk_copy, tar_msk)
        else:
            atten_msk = einsum('b k m, b k n -> b m n', src_msk, tar_msk)
        return atten_msk

    def reverse_relu(self, input):
        input_pos = (input > 0) * 1.0
        input_neg = -torch.relu(-input)
        output = input_pos + input_neg
        return output

def FocalLoss(pred, label, pred_shape, gamma=2, beta=2):
    N = pred.shape[0]
    pred = pred.view(N, 1, -1)
    label = F.interpolate(label, size=pred_shape).float()
    label = label.view(N, -1)
    loss = 0
    for i in range(N):
        _label = label[i].unsqueeze(0)
        probs = F.sigmoid(pred[i].unsqueeze(0))
        pb = torch.where(_label == _label.max(), _label, (1 - _label) ** beta)
        pt = torch.where(_label == _label.max(), probs[:, 0, :], 1 - probs[:, 0, :])
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred[i, 0, :].unsqueeze(0), (_label).float())
        _loss = torch.pow(1 - pt, gamma) * ce_loss * pb
        loss += torch.mean(_loss, dim=1)
    return loss

def AccOnResult(pred, label, pred_shape):
    N = pred.shape[0]
    pred = pred.view(N, pred_shape[0], pred_shape[1], pred_shape[2])
    label = F.interpolate(label, size=pred_shape).float()
    label = label.view(N, pred_shape[0], pred_shape[1], pred_shape[2])
    pred_pos = pred.detach().cpu().numpy()
    label_pos = label.detach().cpu().numpy()
    pred_pos = np.array(np.where(pred_pos==pred_pos.max()))[1:, 0]
    label_pos = np.array(np.where(label_pos==label_pos.max()))[1:, 0]
    return pred_pos, label_pos

def L1Loss(pred, label, mask, pred_shape):
    N = pred.shape[0]
    pred = pred.view(N, 6, -1)
    loss = 0
    for i in range(N):
        _mask = F.interpolate(mask[i].unsqueeze(0), size=pred_shape)
        _mask = (_mask == _mask.max()).float()
        _mask = _mask.view(1, 1, -1)
        _pred = torch.masked_select(pred[i].unsqueeze(0), _mask==1).view(1, -1, 6)
        _label = label[i].unsqueeze(0).repeat(1, _pred.shape[1], 1)
        loss += F.l1_loss(_pred, _label)

    return loss

def L1Loss_avg(pred, label):
    # print(pred.shape, label.shape)
    loss = F.l1_loss(pred, label)
    return loss

def train():
    max_epoches = 300
    batch_size = 4
    lr = 0.00001
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    model = TransformerTracker(backbone_feature=[32, 64, 64*2, 64*3], pos_emb='sine', channel=1, thres=0.7, fusion_layer=3)
    train_dataset = TransLesionData("./DLT/data/train.json", "../data/voxelmorph_data/train", istrain=True)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    def collate_wrapper(batch):
        length = len(batch[0])
        out_tensor = [[] for i in range(length)]
        for b in batch:
            for i in range(length):
                out_tensor[i].append(b[i])
        out_tensor[0] = make_nustedtensor(out_tensor[0], False)
        out_tensor[2] = make_nustedtensor(out_tensor[2], False)
        out_tensor[5] = torch.from_numpy(np.array(out_tensor[5]))
        out_tensor[6] = torch.from_numpy(np.array(out_tensor[6]))
        out_tensor[1] = make_nustedtensor(out_tensor[1], False)
        out_tensor[3] = make_nustedtensor(out_tensor[3], False)
        out_tensor[4] = make_nustedtensor(out_tensor[4], False)
        return out_tensor

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                   collate_fn=collate_wrapper,
                                   pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    max_step = len(train_dataset) // batch_size * max_epoches
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights_dict = torch.load(os.path.join('./saved_model', 'transtracker_8_6.9188.pth'))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    for ep in range(0, max_epoches):
        model.istrain = True
        for iter, pack in enumerate(train_data_loader):
            moving_img = pack[0].float().cuda()
            moving_msk = pack[1].float().cuda()
            fixed_img = pack[2].float().cuda()
            fixed_msk = pack[3].float().cuda()
            cls_label = pack[4].float().cuda()
            box_label = pack[5].float().cuda()

            pred_cls, pred_coord, out_f = model(moving_img, moving_msk, fixed_img, fixed_msk)
            pred_shape = out_f.shape[-3:]
            focal_loss = FocalLoss(pred_cls, cls_label, pred_shape)
            # l1_loss = L1Loss(pred_coord, box_label, cls_label, pred_shape)
            l1_loss = L1Loss_avg(pred_coord[:,:,:3], box_label[:,:,:3])
            loss = focal_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            # print(model.module.auto_thres_module.auto_linear.layers[0].weight.grad)
            optimizer.step()

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset) // batch_size, '/', max_step,
                      'loss:', loss.item(), 'focal loss:', focal_loss.item(), 'l1 loss:', l1_loss.item())
            torch.cuda.empty_cache()

        print('')
        error_valid = validation(model)
        print('avg error on validation set:', error_valid)
        torch.save(model.module.state_dict(), os.path.join('./saved_model',
                                                           'transtracker_' + str(ep) + '_%.4f.pth'%error_valid))

def validation(model):
    batch_size = 1
    valid_dataset = TransLesionData("./DLT/data/valid.json", "../data/voxelmorph_data/valid", istrain=True)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    model.train()
    model.istrain = False
    error_list = []
    with torch.no_grad():
        for iter, pack in enumerate(valid_data_loader):
            moving_img = pack[0].float().cuda()
            moving_msk = pack[1].float().cuda()
            fixed_img = pack[2].float().cuda()
            fixed_msk = pack[3].float().cuda()
            cls_label = pack[4].float().cuda()
            box_label = pack[5].float().cuda()
            img_shape = pack[6].cpu().numpy()

            pred_cls, pred_coord, out_f = model(moving_img, moving_msk, fixed_img, fixed_msk)
            pred_shape = out_f.shape[-3:]

            N = pred_coord.shape[0]
            # # use label or pred
            # mask = F.interpolate(cls_label, size=pred_shape)
            # mask = (mask == mask.max()).float()
            # mask = mask.view(N, 1, -1)
            # pred = pred_coord.view(N, 6, -1)
            # pred = torch.masked_select(pred, mask == 1).view(N, -1, 6)
            # label = box_label.repeat(1, pred.shape[1], 1)

            input_pos = fixed_msk.detach().cpu().numpy()
            input_pos = np.array(np.where(input_pos == input_pos.max()))[2:,0]
            pred, label = pred_coord, box_label
            pred = pred.detach().cpu().numpy()[0, 0]
            label = label.detach().cpu().numpy()[0, 0]
            pred = pred[:3] * img_shape
            _label = label[:3] * img_shape
            label[:3] = label[:3] * img_shape
            label[3:] = label[3:] * img_shape
            print(iter, '/', len(valid_dataset), input_pos, pred, _label, ((pred - _label) ** 2).sum() ** 0.5)
            print(AccOnResult(pred_cls, cls_label, pred_shape), moving_img.shape[2:])

            error_list.append(((pred - _label) ** 2).sum() ** 0.5)
            torch.cuda.empty_cache()

    return np.mean(np.array(error_list))

def test():
    batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = TransformerTracker(backbone_feature=[32, 64, 64 * 2, 64 * 3], pos_emb='sine',
                               channel=1, thres=0.9, fusion_layer=3, istrain=False)
    valid_dataset = TransLesionData("./DLT/data/test.json", "../data/voxelmorph_data/test", istrain=True)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    weights_dict = torch.load(os.path.join('./saved_model', 'transtracker_129_7.0144.pth'))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    model.istrain = False
    error_list = []
    error_list_xyz = [[], [], []]
    acc_list_10, acc_list_r = [], []
    thres = 10
    with torch.no_grad():
        for iter, pack in enumerate(valid_data_loader):
            moving_img = pack[0].float().cuda()
            moving_msk = pack[1].float().cuda()
            fixed_img = pack[2].float().cuda()
            fixed_msk = pack[3].float().cuda()
            cls_label = pack[4].float().cuda()
            box_label = pack[5].float().cuda()
            img_shape = pack[6].cpu().numpy()

            pred_cls, pred_coord, out_f = model(moving_img, moving_msk, fixed_img, fixed_msk)
            pred_shape = out_f.shape[-3:]

            N = pred_coord.shape[0]
            # # use label or pred
            # mask = F.interpolate(cls_label, size=pred_shape)
            # mask = (mask == mask.max()).float()
            # mask = mask.view(N, 1, -1)
            # pred = pred_coord.view(N, 6, -1)
            # pred = torch.masked_select(pred, mask == 1).view(N, -1, 6)
            # label = box_label.repeat(1, pred.shape[1], 1)

            input_pos = fixed_msk.detach().cpu().numpy()
            input_pos = np.array(np.where(input_pos == input_pos.max()))[2:, 0]
            pred, label = pred_coord, box_label
            pred = pred.detach().cpu().numpy()[0, 0]
            label = label.detach().cpu().numpy()[0, 0]
            pred = pred[:3] * img_shape
            _label = label[:3] * img_shape
            label[:3] = label[:3] * img_shape
            label[3:] = label[3:] * img_shape
            print(iter, '/', len(valid_dataset), input_pos, pred, _label, ((pred - _label) ** 2).sum() ** 0.5)
            print(AccOnResult(pred_cls, cls_label, pred_shape), moving_img.shape[2:])

            error_list.append(((pred - _label) ** 2).sum() ** 0.5)
            error_list_xyz[0].append(abs(pred[0,0,0] - _label[0,0,0]))
            error_list_xyz[1].append(abs(pred[0,0,1] - _label[0,0,1]))
            error_list_xyz[2].append(abs(pred[0,0,2] - _label[0,0,2]))
            lesion_size = (label[3]**2 + label[4]**2 + label[5]**2)**0.5
            # print(label, lesion_size)
            if ((pred - _label) ** 2).sum() ** 0.5 < min(thres, lesion_size):
                acc_list_10.append(1)
            else:
                acc_list_10.append(0)
            if ((pred - _label) ** 2).sum() ** 0.5 < lesion_size:
                acc_list_r.append(1)
            else:
                acc_list_r.append(0)

            torch.cuda.empty_cache()

            # np.save(os.path.join('./saved_model', 'moving_feature'), out_f.detach().cpu().numpy())
            # np.save(os.path.join('./saved_model', 'hs'), hs.detach().cpu().numpy())
            # np.save(os.path.join('./saved_model', 'memory_temp'), memory_temp.detach().cpu().numpy())
            # np.save(os.path.join('./saved_model', 'memory_search'), memory_search.detach().cpu().numpy())
            # break

    print('MED:', np.mean(np.array(error_list)), np.std(np.array(error_list)))
    print('MEDz:', np.mean(np.array(error_list_xyz[0])), np.std(np.array(error_list_xyz[0])))
    print('MEDy:', np.mean(np.array(error_list_xyz[1])), np.std(np.array(error_list_xyz[1])))
    print('MEDx:', np.mean(np.array(error_list_xyz[2])), np.std(np.array(error_list_xyz[2])))
    print('CPM@10mm:', np.mean(np.array(acc_list_10)), np.std(np.array(acc_list_10)))
    print('CPM@R:', np.mean(np.array(acc_list_r)), np.std(np.array(acc_list_r)))

def paint_result():
    batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = TransformerTracker(backbone_feature=[32, 64, 64 * 2, 64 * 3], pos_emb='sine',
                               channel=1, thres=0.7, fusion_layer=3, istrain=False)
    valid_dataset = TransLesionData("./DLT/data/test.json", "../data/voxelmorph_data/test", istrain=True)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    weights_dict = torch.load(os.path.join('./saved_model', 'transtracker_215_6.1928.pth'))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    model.istrain = False
    error_list = []
    error_list_xyz = [[], [], []]
    acc_list_10, acc_list_r = [], []
    thres = 10
    with torch.no_grad():
        for iter, pack in enumerate(valid_data_loader):
            moving_img = pack[0].float().cuda()
            moving_msk = pack[1].float().cuda()
            fixed_img = pack[2].float().cuda()
            fixed_msk = pack[3].float().cuda()
            cls_label = pack[4].float().cuda()
            box_label = pack[5].float().cuda()
            img_shape = pack[6].cpu().numpy()

            pred_cls, pred_coord, out_f, atten_list = model(moving_img, moving_msk, fixed_img, fixed_msk, ifatten=True)
            pred_shape = out_f.shape[-3:]

            N = pred_coord.shape[0]
            input_pos = fixed_msk.detach().cpu().numpy()
            input_pos = np.array(np.where(input_pos == input_pos.max()))[2:, 0]
            pred, label = pred_coord, box_label
            pred = pred.detach().cpu().numpy()[0, 0]
            label = label.detach().cpu().numpy()[0, 0]
            pred = pred[:3] * img_shape
            _label = label[:3] * img_shape
            label[:3] = label[:3] * img_shape
            label[3:] = label[3:] * img_shape
            print(iter, '/', len(valid_dataset), input_pos, pred, _label, ((pred - _label) ** 2).sum() ** 0.5)
            print(AccOnResult(pred_cls, cls_label, pred_shape), moving_img.shape[2:])

            error_list.append(((pred - _label) ** 2).sum() ** 0.5)
            error_list_xyz[0].append(abs(pred[0, 0, 0] - _label[0, 0, 0]))
            error_list_xyz[1].append(abs(pred[0, 0, 1] - _label[0, 0, 1]))
            error_list_xyz[2].append(abs(pred[0, 0, 2] - _label[0, 0, 2]))
            lesion_size = (label[3] ** 2 + label[4] ** 2 + label[5] ** 2) ** 0.5
            # print(label, lesion_size)
            if ((pred - _label) ** 2).sum() ** 0.5 < min(thres, lesion_size):
                acc_list_10.append(1)
            else:
                acc_list_10.append(0)
            if ((pred - _label) ** 2).sum() ** 0.5 < lesion_size:
                acc_list_r.append(1)
            else:
                acc_list_r.append(0)

            atten_out = atten_list[-1]
            try:
                attent_out = atten_out.view(1, 1, pred_shape[0], pred_shape[1], pred_shape[2], -1)
                attent_out = attent_out.max(axis=-1)[0]
                attent_out = F.interpolate(attent_out, size=moving_img.shape[-3:])
                attent_out = attent_out.detach().cpu().numpy()[0,0]
                attent_out = np.transpose(attent_out[int(label[0])])
                attent_out = (1 - (attent_out - attent_out.min())/(attent_out.max() - attent_out.min())) * 255
                attent_out = attent_out.astype('uint8')
                _attent_out = cv2.resize(attent_out, (attent_out.shape[0]*8, attent_out.shape[1]*8))
                attent_out = cv2.resize(_attent_out, (attent_out.shape[0], attent_out.shape[1]))
                attent_out = cv2.applyColorMap(attent_out, cv2.COLORMAP_JET)
                attent_out = cv2.cvtColor(attent_out, cv2.COLOR_BGR2RGB)
                image_out = moving_img.detach().cpu().numpy()[0, 0]
                image_out = (((image_out - image_out.min())/(image_out.max() - image_out.min())) * 255).astype('uint8')
                image_out = np.transpose(image_out[int(label[0])])
                image_out = np.array([image_out, image_out, image_out]).transpose((1,2,0)).copy()
                cv2.circle(image_out, (int(label[2]), int(label[1])), int(min(label[-1], label[-2])), (0,0,255), 0)
                # cv2.imwrite(os.path.join('./result_show', str(iter)+'_img.png'), image_out)
                # cv2.imwrite(os.path.join('./result_show', str(iter)+'_att.png'), attent_out)
                att_temp = torch.zeros((1, 1, pred_shape[0], pred_shape[1], pred_shape[2])).float()
                _moving_msk = moving_msk.cpu().numpy()
                temp_label = input_pos
                moving_msk = F.interpolate(moving_msk, size=(pred_shape[0], pred_shape[1], pred_shape[2])).cpu()
                att_temp[moving_msk > 0.7] = atten_out.view(1, 1, pred_shape[0]*pred_shape[1]*pred_shape[2], -1).max(dim=2)[0].cpu()
                att_temp = F.interpolate(att_temp, size=moving_img.shape[-3:]).cpu().numpy()[0, 0]
                att_temp = np.transpose(att_temp[int(temp_label[-3])])
                att_temp = (att_temp - att_temp.min()) / (att_temp.max() - att_temp.min()) * 255 - 1
                att_temp = att_temp.astype('uint8')
                _att_temp = cv2.resize(att_temp, (att_temp.shape[0] * 8, att_temp.shape[1] * 8))
                att_temp = cv2.resize(_att_temp, (att_temp.shape[0], att_temp.shape[1]))
                att_temp = cv2.applyColorMap(att_temp, cv2.COLORMAP_JET)
                att_temp = cv2.cvtColor(att_temp, cv2.COLOR_BGR2RGB)
                image_temp = fixed_img.detach().cpu().numpy()[0, 0]
                image_temp = (((image_temp - image_temp.min()) / (image_temp.max() - image_temp.min())) * 255).astype('uint8')
                image_temp = np.transpose(image_temp[int(temp_label[0])])
                image_temp = np.array([image_temp, image_temp, image_temp]).transpose((1, 2, 0)).copy()
                cv2.circle(image_temp, (int(temp_label[-1]), int(temp_label[-2])), int(min(label[-1], label[-2])), (0, 0, 255), 0)
                temp_out = np.concatenate((image_temp, att_temp), axis=1)
                cv2.imwrite(os.path.join('./result_show', str(iter) + '_temp.png'), temp_out)
                cv2.imwrite(os.path.join('./result_show', str(iter)+'.png'), np.concatenate((image_out, attent_out), axis=1))
                torch.cuda.empty_cache()
            except:
                torch.cuda.empty_cache()

    print('MED:', np.mean(np.array(error_list)), np.std(np.array(error_list)))
    print('MEDz:', np.mean(np.array(error_list_xyz[0])), np.std(np.array(error_list_xyz[0])))
    print('MEDy:', np.mean(np.array(error_list_xyz[1])), np.std(np.array(error_list_xyz[1])))
    print('MEDx:', np.mean(np.array(error_list_xyz[2])), np.std(np.array(error_list_xyz[2])))
    print('CPM@10mm:', np.mean(np.array(acc_list_10)), np.std(np.array(acc_list_10)))
    print('CPM@R:', np.mean(np.array(acc_list_r)), np.std(np.array(acc_list_r)))
    return

def paint_result_list():
    batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = TransformerTracker(backbone_feature=[32, 64, 64 * 2, 64 * 3], pos_emb='sine',
                               channel=1, thres=0.7, fusion_layer=3, istrain=False)
    valid_dataset = TransLesionData("./DLT/data/test.json", "../data/voxelmorph_data/test", istrain=True)
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    weights_dict = torch.load(os.path.join('./saved_model', 'transtracker_8_6.9188.pth'))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    model.istrain = False
    error_list = []
    error_list_xyz = [[], [], []]
    acc_list_10, acc_list_r = [], []
    thres = 10

    def process_attenmap(attenmap, img, pred_shape, label, mask=None):
        if mask == None:
            attent_out = attenmap.view(1, 1, pred_shape[0], pred_shape[1], pred_shape[2], -1)
            attent_out = attent_out.max(axis=-1)[0]
            attent_out = F.interpolate(attent_out, size=img.shape[-3:])
            attent_out = attent_out.detach().cpu().numpy()[0, 0]
        else:
            att_temp = torch.zeros((1, 1, pred_shape[0], pred_shape[1], pred_shape[2])).float()
            _moving_msk = mask.cpu().numpy()
            moving_msk = F.interpolate(mask, size=(pred_shape[0], pred_shape[1], pred_shape[2])).cpu()
            att_temp[moving_msk > 0.7] = attenmap.view(1, 1, pred_shape[0] * pred_shape[1] * pred_shape[2], -1).max(dim=2)[0].cpu()
            attent_out = F.interpolate(att_temp, size=img.shape[-3:]).cpu().numpy()[0, 0]

        attent_out = np.transpose(attent_out[int(label[0])])
        attent_out = (attent_out - attent_out.min()) / (attent_out.max() - attent_out.min()) * 255 - 1
        attent_out = attent_out.astype('uint8')
        _attent_out = cv2.resize(attent_out, (attent_out.shape[0] * 8, attent_out.shape[1] * 8))
        attent_out = cv2.resize(_attent_out, (attent_out.shape[0], attent_out.shape[1]))
        attent_out = cv2.applyColorMap(attent_out, cv2.COLORMAP_JET)
        attent_out = cv2.cvtColor(attent_out, cv2.COLOR_BGR2RGB)
        return attent_out

    def draw_circle(moving_img, label):
        image_out = moving_img.detach().cpu().numpy()[0, 0]
        image_out = (((image_out - image_out.min()) / (image_out.max() - image_out.min())) * 255).astype('uint8')
        image_out = np.transpose(image_out[int(label[0])])
        image_out = np.array([image_out, image_out, image_out]).transpose((1, 2, 0)).copy()
        cv2.circle(image_out, (int(label[2]), int(label[1])), int(min(label[-1], label[-2])), (0, 0, 255), 0)
        return image_out

    with torch.no_grad():
        for iter, pack in enumerate(valid_data_loader):
            moving_img = pack[0].float().cuda()
            moving_msk = pack[1].float().cuda()
            fixed_img = pack[2].float().cuda()
            fixed_msk = pack[3].float().cuda()
            cls_label = pack[4].float().cuda()
            box_label = pack[5].float().cuda()
            img_shape = pack[6].cpu().numpy()

            pred_cls, pred_coord, out_f, atten_list = model(moving_img, moving_msk, fixed_img, fixed_msk, ifatten=True)
            pred_shape = out_f.shape[-3:]

            N = pred_coord.shape[0]
            input_pos = fixed_msk.detach().cpu().numpy()
            input_pos = np.array(np.where(input_pos == input_pos.max()))[2:, 0]
            pred, label = pred_coord, box_label
            pred = pred.detach().cpu().numpy()[0, 0]
            label = label.detach().cpu().numpy()[0, 0]
            pred = pred[:3] * img_shape
            _label = label[:3] * img_shape
            label[:3] = label[:3] * img_shape
            label[3:] = label[3:] * img_shape
            print(iter, '/', len(valid_dataset), input_pos, pred, _label, ((pred - _label) ** 2).sum() ** 0.5)
            print(AccOnResult(pred_cls, cls_label, pred_shape), moving_img.shape[2:])

            error_list.append(((pred - _label) ** 2).sum() ** 0.5)
            error_list_xyz[0].append(abs(pred[0, 0, 0] - _label[0, 0, 0]))
            error_list_xyz[1].append(abs(pred[0, 0, 1] - _label[0, 0, 1]))
            error_list_xyz[2].append(abs(pred[0, 0, 2] - _label[0, 0, 2]))
            lesion_size = (label[3] ** 2 + label[4] ** 2 + label[5] ** 2) ** 0.5
            # print(label, lesion_size)
            if ((pred - _label) ** 2).sum() ** 0.5 < min(thres, lesion_size):
                acc_list_10.append(1)
            else:
                acc_list_10.append(0)
            if ((pred - _label) ** 2).sum() ** 0.5 < lesion_size:
                acc_list_r.append(1)
            else:
                acc_list_r.append(0)

            try:
                img_search = [draw_circle(moving_img, label)]
                img_template = [draw_circle(fixed_img, [input_pos[0], input_pos[1], input_pos[2], label[-3], label[-2], label[-1]])]
                for i in range(3):
                    img_search.append(process_attenmap(atten_list[i][1], moving_img, pred_shape, label, mask=None))
                    img_template.append(process_attenmap(atten_list[i][0], fixed_img, pred_shape, input_pos, mask=moving_msk))
                img_search.append(process_attenmap(atten_list[-1], moving_img, pred_shape, label, mask=None))
                img_template.append(process_attenmap(atten_list[-1], fixed_img, pred_shape, input_pos, mask=moving_msk))
                cv2.imwrite(os.path.join('./result_show', str(iter) + '_temp.png'),  np.concatenate(img_template, axis=1))
                cv2.imwrite(os.path.join('./result_show', str(iter) + '.png'), np.concatenate(img_search, axis=1))
                torch.cuda.empty_cache()
            except:
                torch.cuda.empty_cache()

    print('MED:', np.mean(np.array(error_list)), np.std(np.array(error_list)))
    print('MEDz:', np.mean(np.array(error_list_xyz[0])), np.std(np.array(error_list_xyz[0])))
    print('MEDy:', np.mean(np.array(error_list_xyz[1])), np.std(np.array(error_list_xyz[1])))
    print('MEDx:', np.mean(np.array(error_list_xyz[2])), np.std(np.array(error_list_xyz[2])))
    print('CPM@10mm:', np.mean(np.array(acc_list_10)), np.std(np.array(acc_list_10)))
    print('CPM@R:', np.mean(np.array(acc_list_r)), np.std(np.array(acc_list_r)))
    return

train()
# test()
# paint_result_list()














