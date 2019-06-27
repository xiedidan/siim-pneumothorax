def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir):
        self.df = pd.read_csv(df_path)
        self.height = 256
        self.width = 256
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)

        self.df = self.df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)

        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, image_id)
            if os.path.exists(image_path + '.png') and row[" EncodedPixels"].strip() != '-1':
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["annotations"] = row[" EncodedPixels"].strip()
                counter += 1

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path + '.png').convert("RGB")
        width, height = img.size
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        info = self.image_info[idx]

        mask = rle2mask(info['annotations'], width, height)
        mask = Image.fromarray(mask)
        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = np.expand_dims(mask, axis=0)

        pos = np.where(np.array(mask)[0, :, :])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        masks = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return transforms.ToTensor()(img), target

    def __len__(self):
        return len(self.image_info)
