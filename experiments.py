import os
import argparse
import numpy as np
import pandas as pd

from scipy.linalg import eigh

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pytorchcv.model_provider import get_model
from generator import GeneratorGDFQ, GeneratorQimera

parser = argparse.ArgumentParser(description='PCA features & generate images')
parser.add_argument('--pca_source', action='store_true')
parser.add_argument('--pca_gdfq', action='store_true')
parser.add_argument('--pca_qimera', action='store_true')
parser.add_argument('--pca_mix', action='store_true')
parser.add_argument('--pca_path', action='store_true')
parser.add_argument('--num_dot_per_mix', type=int, default=200)
parser.add_argument('--num_dot_per_path', type=int, default=200)

parser.add_argument('--image_gdfq', action='store_true')
parser.add_argument('--image_qimera', action='store_true')
parser.add_argument('--image_mix', action='store_true')

parser.add_argument('--gdfq_generator_path', type=str)
parser.add_argument('--qimera_generator_path', type=str)


def reduce_df(dataframe, num_per_class):
    df_list = []
    for i in range(10):
        df_list.append(dataframe.iloc[i * 5000: i * 5000 + num_per_class])
    df = pd.concat(df_list)
    return df


class GeneratorGDFQ(nn.Module):
    def __init__(self, options=None, conf_path=None):
        super(GeneratorGDFQ, self).__init__()
        self.label_emb = nn.Embedding(10, 100)
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )

        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(3, affine=False)
        )

    def forward(self, z, labels):
        gen_input = torch.mul(self.label_emb(labels), z)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img
    
    
class GeneratorQimera(nn.Module):
    def __init__(self, options=None, conf_path=None, teacher_weight=None, freeze=True, fc_reduce=False):
        super(GeneratorQimera, self).__init__()
        self.fc_reduce = fc_reduce
        if teacher_weight==None:
            self.label_emb = nn.Embedding(10, 64)
        else:
            self.label_emb = nn.Embedding.from_pretrained(teacher_weight, freeze=freeze)

        self.embed_normalizer = nn.BatchNorm1d(self.label_emb.weight.T.shape,affine=False,track_running_stats=False)
        
        if fc_reduce:
            self.fc_reducer = nn.Linear(in_features=self.label_emb.weight.shape[-1], out_features=64)
            
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(64, 128 * self.init_size ** 2)) 

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )

        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(3, affine=False)
        )

    def forward(self, z, labels, linear=None, z2=None):
        if linear == None:
            gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T #noise before norm
            if self.fc_reduce:
                embed_norm = self.fc_reducer(embed_norm)
        else:
            embed_norm = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T #sep noise before norm
            if self.fc_reduce:
                embed_norm = self.fc_reducer(embed_norm)
            gen_input = (embed_norm * linear.unsqueeze(2)).sum(dim=1)
            
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


if __name__ == '__main__':
    args = parser.parse_args()

    if True in [args.pca_gdfq, args.image_gdfq]:
        generator_GDFQ = GeneratorGDFQ().cuda()
        generator_GDFQ.load_state_dict(torch.load(args.gdfq_generator_path, map_location="cuda:0"))
        generator_GDFQ.eval()

    if True in [args.pca_qimera, args.pca_mix, args.pca_path, args.image_qimera, args.image_mix]:
        generator_qimera = GeneratorQimera().cuda()
        generator_qimera.load_state_dict(torch.load(args.qimera_generator_path, map_location="cuda:0"))
        generator_qimera.eval()

    if True in [args.pca_source, args.pca_gdfq, args.pca_qimera, args.pca_mix, args.pca_path]:
        os.makedirs('./pca_results', exist_ok=True)
        net = get_model("resnet20_cifar10", pretrained=True)
        net = net.cuda()
        net.eval()
        print('teacher network is ready')

        train_set = datasets.CIFAR10('./', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
        ]))
        test_set = datasets.CIFAR10('./', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
        ]))

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                output = net(data)
                _, pred = output.max(1)
                correct += (pred == target).sum()
                total += len(target)

        print(f'teacher net accuracy: {correct / total * 100 : .3f}')

        features = []
        targets = []
        with torch.no_grad():
            for i, (data, target) in enumerate(train_loader):
                data = data.cuda()
                _, feature = net(data, out_feature=True)
                features.append(feature.cpu())
                targets.append(target)

        features_cat = torch.cat(features, dim=0)
        targets_cat = torch.cat(targets, dim=0)

        feature_scaler = StandardScaler().fit(features_cat)
        standardized_data = feature_scaler.transform(features_cat)
        
        covar_matrix = np.matmul(standardized_data.T , standardized_data)
        values, vectors = eigh(covar_matrix, eigvals=(62,63))
        vectors = vectors.T

        if args.pca_source:
            print('pca of source data')
            new_coordinates = np.matmul(vectors, standardized_data.T)
            new_coordinates = np.vstack((new_coordinates, targets_cat)).T
            dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
            dataframe.sort_values(by=['label'], axis=0, inplace=True)
            df = reduce_df(dataframe, 1000)

            pca_result = sns.FacetGrid(df, hue="label", height=10, hue_kws={'marker':['x'] * 10}).map(plt.scatter, '1st_principal', '2nd_principal')

            pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')
            plt.savefig('./pca_results/pca_source.png')

        if args.pca_gdfq:
            print('pca of GDFQ')

            features_GDFQ = []
            targets_GDFQ = []

            with torch.no_grad():
                for i in range(50):
                    z = torch.randn(200, 100).cuda()
                    labels = (torch.ones(200) * (i // 5)).type(torch.LongTensor)
                    targets_GDFQ.append(labels)
                    labels = labels.cuda()
                    z = z.contiguous()
                    labels = labels.contiguous()
                    images = generator_GDFQ(z, labels)
                    _, feature = net(images, out_feature=True)
                    features_GDFQ.append(feature.cpu())
        
            features_cat_GDFQ = torch.cat(features_GDFQ, dim=0)
            targets_cat_GDFQ = torch.cat(targets_GDFQ, dim=0)

            standardized_data_GDFQ = feature_scaler.transform(features_cat_GDFQ)

            new_coordinates_GDFQ = np.matmul(vectors, standardized_data_GDFQ.T)
            new_coordinates_GDFQ = np.vstack((new_coordinates_GDFQ, targets_cat_GDFQ)).T
            dataframe_GDFQ = pd.DataFrame(data=new_coordinates_GDFQ, columns=("1st_principal", "2nd_principal", "label"))

            pca_result = sns.FacetGrid(dataframe_GDFQ, hue="label", height=10, hue_kws={'marker':['x'] * 10}).map(plt.scatter, '1st_principal', '2nd_principal')

            pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')
            plt.savefig('./pca_results/pca_gdfq.png')

        if args.pca_qimera or args.pca_mix or args.pca_path:
            print('pca of Qimera')

            features_qimera = []
            targets_qimera = []

            with torch.no_grad():
                for i in range(50):
                    z = torch.randn(200, 64).cuda()
                    labels = (torch.ones(200) * (i // 5)).type(torch.LongTensor)
                    targets_qimera.append(labels)
                    labels = labels.cuda()
                    z = z.contiguous()
                    labels = labels.contiguous()
                    images = generator_qimera(z, labels)
                    _, feature = net(images, out_feature=True)
                    features_qimera.append(feature.cpu())
        
            features_cat_qimera = torch.cat(features_qimera, dim=0)
            targets_cat_qimera = torch.cat(targets_qimera, dim=0)

            standardized_data_qimera = feature_scaler.transform(features_cat_qimera)

            new_coordinates_qimera = np.matmul(vectors, standardized_data_qimera.T)
            new_coordinates_qimera = np.vstack((new_coordinates_qimera, targets_cat_qimera)).T
            dataframe_qimera = pd.DataFrame(data=new_coordinates_qimera, columns=("1st_principal", "2nd_principal", "label"))

            if args.pca_qimera:
                pca_result = sns.FacetGrid(dataframe_qimera, hue="label", height=10, hue_kws={'marker':['x'] * 10}).map(plt.scatter, '1st_principal', '2nd_principal')

                pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')
                plt.savefig('./pca_results/pca_qimera.png')

            if args.pca_mix or args.pca_path:
                print('pca of embedding superposing')
                linear = []
                for i in range(args.num_dot_per_mix):
                    linear.append(torch.tensor([[i / args.num_dot_per_mix, 1 - (i / args.num_dot_per_mix)]]))
                linear = torch.cat(linear, dim=0)
                features_qimera_mix = []
                targets_qimera_mix = []
                with torch.no_grad():
                    for i in range(10):
                        for j in range(i + 1, 10):
                            z = torch.randn(args.num_dot_per_mix, 2, 64).cuda()
                            z = z.contiguous()
                            labels = torch.tensor([[i, j]])
                            labels = torch.cat([labels] * args.num_dot_per_mix, dim=0).cuda()
                            target = torch.cat([torch.tensor([300])] * args.num_dot_per_mix, dim=0)
                            targets_qimera_mix.append(target)
                            l = linear.cuda()

                            images = generator_qimera(z, labels, l)
                            _, feature = net(images, out_feature=True)
                            features_qimera_mix.append(feature.cpu())

                features_cat_qimera_mix = torch.cat(features_qimera_mix, dim=0)
                targets_cat_qimera_mix = torch.cat(targets_qimera_mix, dim=0)

                standardized_data_qimera_mix = feature_scaler.transform(features_cat_qimera_mix)

                new_coordinates_qimera_mix = np.matmul(vectors, standardized_data_qimera_mix.T)
                new_coordinates_qimera_mix = np.vstack((new_coordinates_qimera_mix, targets_cat_qimera_mix)).T
                dataframe_qimera_mix = pd.DataFrame(data=new_coordinates_qimera_mix, columns=("1st_principal", "2nd_principal", "label"))

                if args.pca_mix:
                    pca_result = sns.FacetGrid(
                                 pd.concat([dataframe_qimera_mix, dataframe_qimera]),
                                 hue="label",
                                 hue_order=[300.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] ,
                                 height=10,
                                 hue_kws={'marker':['x'] * 11, 'color': ['black'] + sns.color_palette("tab10")[:10]}
                                ).map(plt.scatter, '1st_principal', '2nd_principal')

                    pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')
                    plt.savefig('./pca_results/pca_qimera_mix.png')

                if args.pca_path:
                    print('pca of path')
                    linear = []
                    for i in range(args.num_dot_per_path):
                        linear.append(torch.tensor([[i / args.num_dot_per_path, 1 - (i / args.num_dot_per_path)]]))
                    linear = torch.cat(linear, dim=0)

                    features_no_noise = []
                    targets_no_noise = []

                    with torch.no_grad():
                        for i in range(10):
                            for j in range(i + 1, 10):
                                z = torch.zeros(args.num_dot_per_path, 2, 64).cuda()
                                z = z.contiguous()
                                labels = torch.tensor([[i, j]])
                                labels = torch.cat([labels] * args.num_dot_per_path, dim=0).cuda()
                                target = torch.tensor([(i + 1) * 10 + j])
                                target = torch.cat([target] * args.num_dot_per_path, dim=0)
                                targets_no_noise.append(target)
                                l = linear.cuda()
                                images = generator_qimera(z, labels, l)
                                _, feature = net(images, out_feature=True)
                                features_no_noise.append(feature.cpu())
            
                    features_cat_no_noise = torch.cat(features_no_noise, dim=0)
                    targets_cat_no_noise = torch.cat(targets_no_noise, dim=0)

                    standardized_data_no_noise = feature_scaler.transform(features_cat_no_noise)

                    new_coordinates_no_noise = np.matmul(vectors, standardized_data_no_noise.T)
                    new_coordinates_no_noise = np.vstack((new_coordinates_no_noise, targets_cat_no_noise)).T
                    dataframe_no_noise = pd.DataFrame(data=new_coordinates_no_noise, columns=("1st_principal", "2nd_principal", "label"))

                    MID_DOT_NUM = 11

                    linear = []
                    for i in range(MID_DOT_NUM):
                        linear.append(torch.tensor([[i/(MID_DOT_NUM - 1), 1 - (i/(MID_DOT_NUM - 1))]]))

                    linear = torch.cat(linear, dim=0)
    
                    features_no_noise_ten = []
                    targets_no_noise_ten = []
        
                    with torch.no_grad():
                        for i in range(10):
                            for j in range(i + 1, 10):
                                z = torch.zeros(MID_DOT_NUM, 2, 64).cuda()
                                z = z.contiguous()
            
                                labels = torch.tensor([[i, j]])
                                labels = torch.cat([labels] * MID_DOT_NUM, dim=0).cuda()
                                target = torch.tensor([(i + 1) * 10 + j + 500])
                                target = torch.cat([target] * MID_DOT_NUM, dim=0)
                                targets_no_noise_ten.append(target)
                                l = linear.cuda()

                                images = generator_qimera(z, labels, l)
                                _, feature = net(images, out_feature=True)
                                features_no_noise_ten.append(feature.cpu())
            
                    features_cat_no_noise_ten = torch.cat(features_no_noise_ten, dim=0)
                    targets_cat_no_noise_ten = torch.cat(targets_no_noise_ten, dim=0)

                    color = sns.color_palette("tab10")

                    features_total = torch.cat([features_cat_qimera, features_cat_no_noise, features_cat_qimera_mix, features_cat_no_noise_ten], dim=0)
                    targets_total = torch.cat([targets_cat_qimera, targets_cat_no_noise, targets_cat_qimera_mix, targets_cat_no_noise_ten], dim=0)
                    no_noise_start = len(targets_cat_qimera)
                    mix_start = no_noise_start + len(targets_cat_no_noise)
                    no_noise_ten_start = mix_start + len(targets_cat_qimera_mix)

                    standardized_data_total = feature_scaler.transform(features_total)

                    x = 0
                    for i in range(10):
                        for j in range(i + 1, 10):
                            if j - i == 1:
                                selected_features = torch.cat([
                                    features_cat_qimera[i * 1000: (i + 1) * 1000],
                                    features_cat_qimera[j * 1000: (j + 1) * 1000],
                                    features_cat_no_noise[x * args.num_dot_per_path: (x + 1) * args.num_dot_per_path],
                                    features_cat_qimera_mix[x * args.num_dot_per_mix: (x + 1) * args.num_dot_per_mix],
                                    features_cat_no_noise_ten[x * MID_DOT_NUM: (x + 1) * MID_DOT_NUM]
                                ] , dim=0)
            
                                selected_targets = torch.cat([
                                    targets_cat_qimera[i * 1000: (i + 1) * 1000],
                                    targets_cat_qimera[j * 1000: (j + 1) * 1000],
                                    targets_cat_no_noise[x * args.num_dot_per_path: (x + 1) * args.num_dot_per_path],
                                    targets_cat_qimera_mix[x * args.num_dot_per_mix: (x + 1) * args.num_dot_per_mix],
                                    targets_cat_no_noise_ten[x * MID_DOT_NUM: (x + 1) * MID_DOT_NUM]
                                ], dim=0)
                                standardized_data_selected = feature_scaler.transform(selected_features)
                                covar_matrix_selected = np.matmul(standardized_data_selected.T , standardized_data_selected)

                                values_selected, vectors_selected = eigh(covar_matrix_selected, eigvals=(62,63))

                                vectors_selected = vectors_selected.T

                                new_coordinates_selected = np.matmul(vectors_selected, standardized_data_selected.T)
                                new_coordinates_selected = np.vstack((new_coordinates_selected, selected_targets)).T
                                df_selected = pd.DataFrame(data=new_coordinates_selected, columns=("1st_principal", "2nd_principal", "label"))
                                pca_result = sns.FacetGrid(df_selected, hue="label", height=10, hue_kws={'marker':['o', 'o', 'o', 'o', 'o'], 's':[30, 30, 30, 30, 300], 'color':[color[i], color[j], 'black', 'lightgreen', 'black'],}).map(plt.scatter, '1st_principal', '2nd_principal')
                                pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')

                                plt.savefig(f'pca_results/pca_path_{i}_{j}.png')
                            x += 1

    if True in [args.image_gdfq, args.image_qimera, args.image_mix]:
        os.makedirs('./generated_images', exist_ok=True)
        if args.image_gdfq:
            print('generate images with GDFQ')
            images_GDFQ = []
            with torch.no_grad():
                z_GDFQ = torch.randn(10, 100).cuda()
                z_GDFQ = z_GDFQ.contiguous()

                for i in range(10):
                    labels = (torch.ones(10) * i).type(torch.LongTensor).cuda()
                    labels = labels.contiguous()
                    image_GDFQ = generator_GDFQ(z_GDFQ, labels)
                    images_GDFQ.append(image_GDFQ)

            images_GDFQ = torch.cat(images_GDFQ, dim=0)
            torchvision.utils.save_image(images_GDFQ, './generated_images/GDFQ.png', nrow=10)

        if args.image_qimera:
            print('generate images with Qimera')
            images_qimera = []
            with torch.no_grad():
                z_qimera = torch.randn(10, 64).cuda()
                z_qimera = z_qimera.contiguous()

                for i in range(10):
                    labels = (torch.ones(10) * i).type(torch.LongTensor).cuda()
                    labels = labels.contiguous()
                    image_qimera = generator_qimera(z_qimera, labels)
                    images_qimera.append(image_qimera)

            images_qimera = torch.cat(images_qimera, dim=0)
            torchvision.utils.save_image(images_qimera, './generated_images/qimera.png', nrow=10)

        if args.image_mix:
            print('generate images with Qimera & embedding superposing')
            images_qimera_multilabel = []
            with torch.no_grad():
                z = torch.randn(1, 64).cuda()
                z = z.contiguous()
                for i in range(10):
                    for j in range(10):
                        labels = torch.tensor([[i, j]]).cuda()
                        labels = labels.contiguous()
                        linear = torch.nn.functional.softmax(torch.ones(1, 2),dim=1).cuda()
                        images = generator_qimera(z, labels, linear)
                        images_qimera_multilabel.append(images)
                images = torch.cat(images_qimera_multilabel, dim=0)

            torchvision.utils.save_image(images, f'./generated_images/qimera_mix.png', nrow=10) 

