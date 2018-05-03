import pandas as pd
from os import listdir
from os.path import isfile, join, abspath
from .parsing import *

class Parser():
    def __init__(self, root_dir, dicom_dir, contour_dir, linkfile):
        """
        Initalize the Parser class with paths to the data
        :param root_dir: The absolute path to root dir which contains dicom_dir, contour_dir and the linkfile
        example structure of the directory:
            --root-dir
                --dicom-dir
                --contour-dir
                    --i-contours
                    --o-contours
                --linkfile
        :param dicom_dir: The relative path inside root_dir that contains all dicoms
        :param contour_dir: The relative path inside root_dir that contains all contours
        :param linkfile: The csv file that links sub dirs in dicom_dir to sub dirs in coutour_dir
        """
        self.root_dir = root_dir
        self.img_root_dir = join(root_dir, dicom_dir)
        self.target_root_dir = join(root_dir, contour_dir)
        self.linkfile = pd.read_csv(join(root_dir, linkfile))

    def parse_all_patients(self, check_mask=lambda mask, image:True):
        """
        Generate images and target masks for all patients
        Note that patient info is not retained, all slices across all patients
        are saved to the same array.
        :param check_mask: function to sanity check the mask, with numpy arrays of mask and image as input
        :return: a numpy array of images and an array of masks
        """
        images, masks = [], []
        masks_to_check = []
        for _, patient in self.linkfile.iterrows():
            dicom_id = patient['patient_id']
            contour_id = patient['original_id']
            imgs, msks, to_check = self.parse_patient(dicom_id, contour_id, check_mask)
            print('For patient id: {}, there are {} images and {} masks'.format(dicom_id, len(imgs), len(msks)))
            images.extend(imgs)
            masks.extend(msks)
            masks_to_check.extend(to_check)
        image_array = np.stack(images)
        mask_array = np.stack(masks)
        return image_array, mask_array, masks_to_check

    def parse_patient(self, dicom_id, contour_id, check_mask=lambda mask, image:True):
        """
        given the patient id to identify the dicom sub-dir and the contour sub-dir,
        generate a list of images, a list of corresponding masks, and a list
        of mask paths that need to be checked.
        :param dicom_id: dicom sub dir that contains all image slices for this patient
        :param contour_id: contour sub dir that contains all labeled masks, each corresponds
        to one slice of the patient.
        :param check_mask: function to check if the mask is correct, takes mask and image as arguments,
        default to no check and always returns true
        :return: a list of the 2D slice images of the patient and a list of the masks
        """
        # read in all slice numbers that have contour, then read in corresponding image slices
        i_contour_fnames = listdir(join(self.target_root_dir, contour_id, 'i-contours'))
        images, masks = [], []
        to_check_masks = []
        for contour_name in i_contour_fnames:
            dicom_name = str(int(contour_name.split('-')[2])) + '.dcm'
            try:
                image = parse_dicom_file(join(self.img_root_dir, dicom_id, dicom_name))['pixel_data']
            except FileNotFoundError:
                continue
            contour = parse_contour_file(join(self.target_root_dir, contour_id, 'i-contours', contour_name))
            mask = poly_to_mask(contour, image.shape[1], image.shape[0])
            if check_mask(mask, image):
                images.append(image)
                masks.append(mask)
            else:
                to_check_masks.append(join(self.target_root_dir, contour_id, 'i-contours', contour_name))
        return images, masks, to_check_masks

    @staticmethod
    def check_by_intensity(mask, image, intensity_thresh=0.1):
        """
        Given the mask and the image, check if the masked area has
        median intensity (normalized by the max intensity of the image)
        above a threshold
        :param mask: 2d numpy boolean array
        :param image: 2d numpy float array
        :param intensity_thresh: float
        :return: boolean
        """
        intensity_normalized = np.median(image[mask]) / image.max()
        return True if intensity_normalized > intensity_thresh else False
