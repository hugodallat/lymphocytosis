import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, sampler
from datetime import date
from PIL import Image
import torch
from torchvision import transforms

class Lympho_Dataset(Dataset):

    def __init__(self, path_images, df, transform=None):
        """
        Args:
            path_images: (str) path to the images origin directory.
            data_df: (DataFrame) list of subjects used.
            transform: Optional, transformations applied to the tensor
        """
        self.path_images = path_images
        self.df = df
        self.transform = transform
        self.list_patients = df['ID'].tolist()
        self.lymph_count = df['LYMPH_COUNT'].tolist()
        self.age = df['AGE'].tolist()
        self.labels = df['LABEL'].tolist()
        self.img_dict = {idx: {'label': self.labels[idx],
                               'age': self.age[idx],
                               'lymph_count': self.lymph_count[idx],
                               'patient': self.list_patients[idx],
                               'images_path': [path_images + '/' + patient + '/' + img_path for img_path in
                                               os.listdir(path_images + '/' + patient)]} for idx, patient in
                         enumerate(self.list_patients)}

    def __len__(self):
        return len(self.df)

    def load_image(self, image_path):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly.
        """
        #         image = imageio.imread(image_path).astype(np.uint8)[...,None]
        image = Image.open(image_path)
        # Stack image on itself 3 times to simulate RGB image (3 channels required for model's input)
        return image

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) the index of the subject whom data is loaded.
        Returns:
            sample: (dict) corresponding data described by the following keys:
                image: (Tensor) Images of the patient's blood cells image in a tensor
                label: (int) the diagnosis code (0 for reactive or 1 for cancerous)
                participant_id: (str) ID of the participant
                lymph_count : (int) Lymphocyte concentration in patient's blood
                age : (int) Patient's age
        """
        if self.transform == None:
            images = [transforms.ToTensor()(self.load_image(image)).unsqueeze(0) for image in
                      self.img_dict[idx]['images_path']]
        else:
            images = [self.transform(self.load_image(image)).unsqueeze(0) for image in
                      self.img_dict[idx]['images_path']]
        # images = [transforms.ToTensor()(image).unsqueeze_(0) for image in images]
        images = torch.cat(images, axis=0)

        age = torch.Tensor([self.img_dict[idx]['age']])
        lymph_count = torch.Tensor([self.img_dict[idx]['lymph_count']])
        patient = self.img_dict[idx]['patient']
        label = torch.Tensor([self.img_dict[idx]['label']])

        sample = {'images': images,
                  'lymph_count': lymph_count,
                  'patient': patient,
                  'label': label,
                  'age': age}
        return sample


def characteristics_table(df):
    """Creates a DataFrame that summarizes the characteristics of the DataFrame df"""
    diagnoses = np.unique(df.LABEL.values)
    population_df = pd.DataFrame(index=diagnoses,
                                 columns=['N', 'age', '%sexF', 'LYMPH_COUNT'])

    for label in population_df.index.values:
        diagnosis_df = df[df.LABEL == label]
        population_df.loc[label, 'N'] = len(diagnosis_df)
        # Age
        mean_age = np.mean(diagnosis_df.AGE)
        std_age = np.std(diagnosis_df.AGE)
        population_df.loc[label, 'age'] = '%.1f ± %.1f' % (mean_age, std_age)
        # Sex
        population_df.loc[label, '%sexF'] = round(
            (len(diagnosis_df[diagnosis_df.GENDER == 'F']) / len(diagnosis_df)) * 100, 1)
        # Lymph count
        mean_MMS = np.mean(diagnosis_df.LYMPH_COUNT)
        std_MMS = np.std(diagnosis_df.LYMPH_COUNT)
        population_df.loc[label, 'LYMPH_COUNT'] = '%.1f ± %.1f' % (mean_MMS, std_MMS)

    return population_df

def split_train_val_data(df, seed=6):
    train_df = df[df['LABEL'] != -1]
    split_df = train_df
    np.random.seed(seed)
    msk = np.random.rand(len(train_df)) < 0.75
    train_df = split_df[msk]
    val_df = split_df[~msk]
    return train_df, val_df

