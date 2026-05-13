from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import pdb
import scipy.io

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='llcm', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='sysu_deen_p4_n6_lr_0.1_seed_0_best.t', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/root/autodl-tmp/project/LLCM-main/DEEN/Dataset/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = './Datasets/RegDB/'
    n_class = 206
    test_mode = [2, 1]
elif dataset =='llcm':
    data_path = '/root/autodl-tmp/project/LLCM-main/DEEN/Dataset/LLCM/'
    n_class = 713
    test_mode = [1, 2] #[2, 1]: VIS to IR; [1, 2]: IR to VIS
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
pool_dim = 2048
print('==> Building model..')
net = embed_net(n_class, dataset, arch=args.arch)
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    # 假设 TTA_FACTOR 已经在全局定义为 3
    # TTA_FACTOR = 3

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):

            target_batch_size = input.size(0)
            input = Variable(input.cuda())

            feat_pool_raw, feat_fc_raw = net(input, input, test_mode[0])

            # --------------------- 【核心 TTA 聚合和修复】 ----------------------

            # 1. 计算理论上应该输出的总特征数量 (例如 64 * 3 = 192, 或 51 * 3 = 153)
            TTA_FACTOR = feat_pool_raw.shape[0] // target_batch_size
            expected_raw_size = target_batch_size * TTA_FACTOR

            # 2. 检查模型输出是否大于预期（例如 192 > 153），如果是，则切片到预期大小
            if feat_pool_raw.shape[0] > expected_raw_size:
                print(
                    f"Clipping TTA output from {feat_pool_raw.shape[0]} to expected {expected_raw_size} in batch {batch_idx}")
                feat_pool_raw = feat_pool_raw[:expected_raw_size]
                feat_fc_raw = feat_fc_raw[:expected_raw_size]

            # 3. 重新塑形：将 (N * TTA_FACTOR, D) 变为 (N, TTA_FACTOR, D)
            #    这一步将所有增强特征按图片分组。
            feat_pool_reshaped = feat_pool_raw.view(target_batch_size, TTA_FACTOR, -1)
            feat_fc_reshaped = feat_fc_raw.view(target_batch_size, TTA_FACTOR, -1)

            # 4. 聚合：沿着 TTA 维度（dim=1）取平均值，得到最终特征 (N, D)
            feat_pool = torch.mean(feat_pool_reshaped, dim=1)
            feat_fc = torch.mean(feat_fc_reshaped, dim=1)

            # -------------------------------------------------------------

            # 5. 赋值: batch_num 总是等于 target_batch_size (例如 64 或 51)
            batch_num = feat_pool.shape[0]

            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc
def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):

            target_batch_size = input.size(0)
            input = Variable(input.cuda())

            feat_pool_raw, feat_fc_raw = net(input, input, test_mode[1])

            # --------------------- 【核心 TTA 聚合和修复】 ----------------------

            # 1. 动态计算 TTA 因子，并计算理论应有的输出大小
            #    注意：这里我们假设 TTA 因子是固定的整数 (192 / 64 = 3)
            #    如果 feat_pool.shape[0] 是 0，除法会失败，但通常不会发生。
            TTA_FACTOR = feat_pool_raw.shape[0] // target_batch_size
            expected_raw_size = target_batch_size * TTA_FACTOR

            # 2. 检查冗余：如果模型输出多于理论值，则切片
            if feat_pool_raw.shape[0] > expected_raw_size:
                print(
                    f"Clipping TTA output from {feat_pool_raw.shape[0]} to expected {expected_raw_size} in batch {batch_idx}")
                feat_pool_raw = feat_pool_raw[:expected_raw_size]
                feat_fc_raw = feat_fc_raw[:expected_raw_size]

            # 3. 重新塑形：(N * TTA_FACTOR, D) -> (N, TTA_FACTOR, D)
            feat_pool_reshaped = feat_pool_raw.view(target_batch_size, TTA_FACTOR, -1)
            feat_fc_reshaped = feat_fc_raw.view(target_batch_size, TTA_FACTOR, -1)

            # 4. 聚合：沿着 TTA 维度取平均值 (保留了增强信息)
            feat_pool = torch.mean(feat_pool_reshaped, dim=1)
            feat_fc = torch.mean(feat_fc_reshaped, dim=1)

            # -------------------------------------------------------------

            batch_num = feat_pool.shape[0]  # 现在 batch_num 总是等于 target_batch_size

            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc
# def extract_gall_feat(gall_loader):
#     net.eval()
#     print('Extracting Gallery Feature...')
#     start = time.time()
#     ptr = 0
#     gall_feat_pool = np.zeros((ngall, pool_dim))
#     gall_feat_fc = np.zeros((ngall, pool_dim))
#     with torch.no_grad():
#         for batch_idx, (input, label) in enumerate(gall_loader):
#             # batch_num = input.size(0)  # <-- 问题所在行，不再使用它
#             input = input.cuda()  # Variable 在新版 PyTorch 中已弃用，可以直接用 tensor
#             target_batch =  input.shape[0]
#             feat_pool, feat_fc = net(input, input, test_mode[0])
#             feat_pool = feat_pool[:target_batch]
#             feat_fc = feat_fc[:target_batch]
#             # --- 修改开始 ---
#             # 1. 获取模型输出特征的真实数量
#             actual_batch_num = feat_pool.shape[0]
#
#             # 2. 使用真实的数量进行切片和赋值
#             gall_feat_pool[ptr:ptr + actual_batch_num, :] = feat_pool.detach().cpu().numpy()
#             gall_feat_fc[ptr:ptr + actual_batch_num, :] = feat_fc.detach().cpu().numpy()
#
#             # 3. 使用真实的数量更新指针
#             ptr = ptr + actual_batch_num
#             # --- 修改结束 ---
#
#     print('Extracting Time:\t {:.3f}'.format(time.time() - start))
#     return gall_feat_pool, gall_feat_fc
#
#
# def extract_query_feat(query_loader):
#     net.eval()
#     print('Extracting Query Feature...')
#     start = time.time()
#     ptr = 0
#     query_feat_pool = np.zeros((nquery, pool_dim))
#     query_feat_fc = np.zeros((nquery, pool_dim))
#     with torch.no_grad():
#         for batch_idx, (input, label) in enumerate(query_loader):
#             # batch_num = input.size(0) # <-- 问题所在行，不再使用它
#             input = input.cuda()  # Variable 在新版 PyTorch 中已弃用，可以直接用 tensor
#             target_batch = input.shape[0]
#             feat_pool, feat_fc = net(input, input, test_mode[1])
#             feat_pool = feat_pool[:target_batch]
#             feat_fc = feat_fc[:target_batch]
#             # --- 修改开始 ---
#             # 1. 获取模型输出特征的真实数量 (这会是 192)
#             actual_batch_num = feat_pool.shape[0]
#
#             # 2. 使用真实的数量 (192) 进行切片和赋值
#             #    现在等号左边的形状是 [ptr:ptr+192, :]
#             #    等号右边的形状是 (192, 2048)
#             #    形状匹配，问题解决
#             query_feat_pool[ptr:ptr + actual_batch_num, :] = feat_pool.detach().cpu().numpy()
#             query_feat_fc[ptr:ptr + actual_batch_num, :] = feat_fc.detach().cpu().numpy()
#
#             # 3. 使用真实的数量更新指针
#             ptr = ptr + actual_batch_num
#             # --- 修改结束 ---

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc

def process_llcm(img_dir, mode = 1):
    if mode== 1:
        input_data_path = os.path.join(data_path, 'idx/test_vis.txt')
    elif mode ==2:
        input_data_path = os.path.join(data_path, 'idx/test_nir.txt')
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        file_cam = [int(s.split('c0')[1][0]) for s in data_file_list]
        
    return file_image, np.array(file_label), np.array(file_cam)


def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    ir_cameras = ['cam1','cam2','cam4','cam5']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)
    
if dataset == 'llcm':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'llcm_agw_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_llcm(data_path, mode=test_mode[0])

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

    result = {'gallery_f':gall_feat_fc,'gallery_label':gall_label,'gallery_cam':gall_cam,'query_f':query_feat_fc,'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('tsne.mat',result)

elif dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader)
    
    print(len(query_feat_fc), len(gall_feat_fc))
    print(gall_cam)

    result = {'gallery_f':gall_feat_fc,'gallery_label':gall_label,'gallery_cam':gall_cam,'query_f':query_feat_fc,'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('tsne1.mat',result)
