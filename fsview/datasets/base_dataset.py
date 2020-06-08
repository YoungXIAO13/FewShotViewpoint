
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


## Lighting noise transform
class TransLightning(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


## ImageNet statistics
imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203],
                                ])
}


## Define normalization and random gaussian noise for input image
add_noise = TransLightning(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec'])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


## Crop the image using random bounding box with IoU >= 0.7 compared with the ground truth
def random_crop(im, x, y, w, h):
    left = max(0, x + int(np.random.uniform(-0.1, 0.1) * w))
    upper = max(0, y + int(np.random.uniform(-0.1, 0.1) * h))
    right = min(im.size[0], x + int(np.random.uniform(0.9, 1.1) * w))
    lower = min(im.size[1], y + int(np.random.uniform(0.9, 1.1) * h))
    im_crop = im.crop((left, upper, right, lower))
    return im_crop


## Resize image to the desired dimension without changing aspect ratio
def resize_pad(im, dim):
    w, h = im.size
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.size[0]) / 2))
    right = int(np.floor((dim - im.size[0]) / 2))
    top = int(np.ceil((dim - im.size[1]) / 2))
    bottom = int(np.floor((dim - im.size[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))
    return im


## Rotate the default canonical azimuth for NOCS
def nocs_rot(rgb, theta):
    """
    :param rgb: original rgb image representing local field with each channel correponding to x, y, z
    :param theta: rotation angle in azimuth in degrees
    """
    assert(rgb.shape[2] == 3)
    mask_zero = rgb.sum(-1) == 0
    x = (rgb[:, :, 0] - 128) / 256.
    y = (rgb[:, :, 1] - 128) / 256.
    theta = radians(theta)
    scale = abs(sin(theta)) + abs(cos(theta))
    x_rot = (x * cos(theta) - y * sin(theta)) / scale
    y_rot = (x * sin(theta) + y * cos(theta)) / scale
    r_rot = (x_rot * 256 + 128).astype('uint8')
    g_rot = (y_rot * 256 + 128).astype('uint8')
    rgb[:, :, 0] = r_rot
    rgb[:, :, 1] = g_rot
    rgb[mask_zero, :] = 0
    return Image.fromarray(rgb)


## Load images rendered using viewpoints on semi-sphere
def read_semisphere(render_transform, render_path, view_num, tour, rotation=0):
    """
    Read multi view rendered images from the target path
    :param render_transform: image processing applied to the rendered image
    :param render_path: folder containing the rendered images for training example
    :param view_num: number of rendered images used as 3D shape representation
    :param tour: number of elevations of the rendered images
    :param rotation: randomization with respect to the canonical view in term of azimuth
    :return:
    """
    render_names = sorted([name for name in os.listdir(render_path)])
    step = int(72 / (view_num / tour))
    renders_low = np.linspace(0, 71, 72, dtype='int')
    renders_mid = renders_low + 72
    renders_up = renders_mid + 72

    if basename(render_path) == 'nocs':
        rotation_angle = rotation * 5
        rotation = 0
    else:
        rotation_angle = 0

    if tour == 1:
        render_ids = np.concatenate((renders_mid[rotation:], renders_mid[:rotation]))[::step]
    elif tour == 2:
        render_ids = np.concatenate((np.concatenate((renders_low[rotation:], renders_low[:rotation]))[::step],
                                     np.concatenate((renders_mid[rotation:], renders_mid[:rotation]))[::step]))
    else:
        render_ids = np.concatenate((np.concatenate((renders_low[rotation:], renders_low[:rotation]))[::step],
                                     np.concatenate((renders_mid[rotation:], renders_mid[:rotation]))[::step],
                                     np.concatenate((renders_up[rotation:], renders_up[:rotation]))[::step]))
    views = []
    for i in range(0, len(render_ids)):
        render = Image.open(os.path.join(render_path, render_names[render_ids[i]])).convert('RGB')
        if rotation_angle != 0:
            render = nocs_rot(np.array(render), rotation_angle)
        render = render_transform(render)
        views.append(render.unsqueeze(0))
    views = torch.cat(views, 0)
    return views


## Load images rendered using viewpoints on dodecahedron
def read_dodecahedron(render_transform, render_path, view_num, rotation=0):
    """
    Read multi view rendered images from the target path
    :param render_transform: image processing applied to the rendered image
    :param render_path: folder containing the rendered images for training example
    :param view_num: number of rendered images used as 3D shape representation
    :param rotation: randomization with respect to the canonical view in term of azimuth
    :return:
    """
    render_names = sorted([name for name in os.listdir(render_path)])
    rotation_angle = rotation * 5
    views = []
    for i in range(0, view_num):
        render = Image.open(os.path.join(render_path, render_names[i])).convert('RGB')
        if rotation_angle != 0:
            render = nocs_rot(np.array(render), rotation_angle)
        render = render_transform(render)
        views.append(render.unsqueeze(0))
    views = torch.cat(views, 0)
    return views


## Load point clouds
def read_pointcloud(model_path, point_num, rotation=0):
    """
    Read point cloud from the target path
    :param model_path: file path for the point cloud
    :param point_num: input point number of the point cloud
    :param rotation: randomization with respect to the canonical view in term of azimuth
    :return: shape tensor
    """
    # read in original point cloud
    point_cloud_raw = pymesh.load_mesh(model_path).vertices

    # randomly select a fix number of points on the surface
    point_subset = np.random.choice(point_cloud_raw.shape[0], point_num, replace=False)
    point_cloud = point_cloud_raw[point_subset]

    # apply the random rotation on the point cloud
    if rotation != 0:
        alpha = radians(rotation)
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                               [np.sin(alpha), np.cos(alpha), 0.],
                               [0., 0., 1.]])
        point_cloud = np.matmul(point_cloud, rot_matrix.transpose())

    point_cloud = torch.from_numpy(point_cloud.transpose()).float()

    # normalize the point cloud into [0, 1]
    point_cloud = point_cloud - torch.min(point_cloud)
    point_cloud = point_cloud / torch.max(point_cloud)

    return point_cloud



class BaseDataset(data.Dataset):
    def __init__(self,
                 root_dir, annotation_file, 
                 input_dim=224, ):

        self.root_dir = root_dir
        self.input_dim = input_dim
        self.shape = shape
        self.shape_dir = shape_dir
        self.view = view
        self.point_num = point_num
        self.view_num = view_num
        self.train = train
        self.tour = tour
        self.rotated = rotated
        self.random_range = random_range
        self.random_model = random_model
        self.rotational_symmetry_cats = ['ashtray', 'basket', 'bottle', 'bucket', 'can', 'cap', 'cup',
                                         'fire_extinguisher', 'fish_tank', 'flashlight', 'helmet', 'jar', 'paintbrush',
                                         'pen', 'pencil', 'plate', 'pot', 'road_pole', 'screwdriver', 'toothbrush',
                                         'trash_bin', 'trophy']

        # load the data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        frame = frame[frame.elevation != 90]
        frame = frame[frame.difficult == 0]
        if annotation_file == 'ObjectNet3D.txt':
            if keypoint:
                frame = frame[frame.has_keypoints == 1]
                frame = frame[frame.truncated == 0]
                frame = frame[frame.occluded == 0]
            frame.azimuth = (360. + frame.azimuth) % 360
        if train:
            frame = frame[frame.set == 'train']
        else:
            frame = frame[frame.set == 'val']
            frame = frame[frame.truncated == 0]
            frame = frame[frame.occluded == 0]

        # choose cats for Object3D
        if cat_choice is not None:
            if train:
                frame = frame[~frame.cat.isin(cat_choice)] if novel else frame
            else:
                frame = frame[frame.cat.isin(cat_choice)]

        # sample K-shot training data
        if train and shot is not None:
            cats = np.unique(frame.cat)
            fewshot_frame = []
            for cat in cats:
                fewshot_frame.append(frame[frame.cat == cat].sample(n=shot))
            frame = pd.concat(fewshot_frame)

        self.annotation_frame = frame

        # define data augmentation and preprocessing for RGB images in training
        self.im_augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(), normalize, disturb])

        # define data preprocessing for RGB images in validation
        self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])

        # define data preprocessing for rendered multi view images
        self.render_transform = transforms.ToTensor()
        if input_dim != 224:
            self.render_transform = transforms.Compose([transforms.Resize(input_dim), transforms.ToTensor()])

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotation_frame.iloc[idx, -1])
        cat = self.annotation_frame.iloc[idx]['cat']
        cad_index = self.annotation_frame.iloc[idx]['cad_index']

        # select a random shape from the same category in testing
        if self.random_model:
            df_cat = self.annotation_frame[self.annotation_frame.cat == cat]
            cad_index = np.random.choice(np.unique(df_cat.cad_index))

        left = self.annotation_frame.iloc[idx]['left']
        upper = self.annotation_frame.iloc[idx]['upper']
        right = self.annotation_frame.iloc[idx]['right']
        lower = self.annotation_frame.iloc[idx]['lower']

        # use continue viewpoint annotation
        label = self.annotation_frame.iloc[idx, 9:12].values

        # load real images in a Tensor of size C*H*W
        im = Image.open(img_name).convert('RGB')

        if self.train:
            # Gaussian blur
            if min(right - left, lower - upper) > 224 and np.random.random() < 0.3:
                im = im.filter(ImageFilter.GaussianBlur(3))

            # crop the original image with 2D bounding box jittering
            im = random_crop(im, left, upper, right - left, lower - upper)

            # Horizontal flip
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                label[0] = 360 - label[0]
                label[2] = -label[2]

            # Random rotation
            if np.random.random() > 0.5:
                r = max(-60, min(60, np.random.randn() * 30))
                im = im.rotate(r)
                label[2] = label[2] + r
                label[2] += 360 if label[2] < -180 else (-360 if label[2] > 180 else 0)

            # pad it to the desired size
            im = resize_pad(im, self.input_dim)
            im = self.im_augmentation(im)
        else:
            # crop the ground truth bounding box and pad it to the desired size
            im = im.crop((left, upper, right, lower))
            im = resize_pad(im, self.input_dim)
            im = self.im_transform(im)

        label[0] = (360. - label[0]) % 360.
        label[1] = label[1] + 90.
        label[2] = (label[2] + 180.) % 360.
        label = label.astype('int')

        if self.shape is None:
            label = torch.from_numpy(label).long()
            return im, label

        # randomize the canonical view with respect to the azimuth
        # range_0: [-45, 45]; range_1: [-90, 90]; range_2: [-180, 180]
        if self.rotated and cat not in self.rotational_symmetry_cats:
            rotation = np.random.randint(-8, 9) % 72 if self.random_range == 0 else \
                (np.random.randint(-17, 18) % 72 if self.random_range == 1 else np.random.randint(0, 72))
            label[0] = (label[0] - rotation * 5) % 360
        else:
            rotation = 0

        if self.shape == 'nontextured' or self.shape == 'nocs':

            # load render images in a Tensor of size K*C*H*W
            render_path = os.path.join(self.root_dir, self.shape_dir, self.view, cat, '{:02d}'.format(cad_index), self.shape)

            # read multiview rendered images
            if self.view == 'semisphere':
                renders = read_semisphere(self.render_transform, render_path, self.view_num, self.tour, rotation)
            else:
                renders = read_dodecahedron(self.render_transform, render_path, self.view_num, rotation)

            label = torch.from_numpy(label).long()
            if self.train:
                return im, renders, label, cat
            else:
                return im, renders, label

        if self.shape == 'pointcloud':

            # load point_clouds
            pointcloud_path = os.path.join(self.root_dir, self.shape_dir, cat, '{:02d}'.format(cad_index), 'compressed.ply')
            point_cloud = read_pointcloud(pointcloud_path, self.point_num, rotation)

            if self.train:
                return im, point_cloud, label, cat
            else:
                return im, point_cloud, label
            