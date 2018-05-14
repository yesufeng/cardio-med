import pandas as pd
from os import listdir, makedirs
from os.path import join, exists
from .parsing import *

class Parser():
    def __init__(self, root_dir, dicom_dir, contour_dir, linkfile, processed_dir):
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
        :param processed_dir: The absolute path to save the processed masks and images, the '0' dir
        is used for keras method flow_from_directory.
            --processed_dir
                --images
                    --0
                --masks
                    --0
        """
        self.root_dir = root_dir
        self.img_root_dir = join(root_dir, dicom_dir)
        self.target_root_dir = join(root_dir, contour_dir)
        self.linkfile = pd.read_csv(join(root_dir, linkfile))
        self.img_dir = join(processed_dir, 'images', '0')
        self.i_msk_dir = join(processed_dir, 'i_masks', '0')
        self.o_msk_dir = join(processed_dir, 'o_masks', '0')

    def parse_all_patients(self, check_mask=lambda mask, image:True):
        """
        Generate images and target masks for all patients, saved to sub directories
        in processed_dir
        :param check_mask: function to sanity check the mask, with numpy arrays of mask and image as input
        :return: a list of masks that do not pass the sanity check.
        """
        masks_to_check = []
        if not exists(self.img_dir):
            makedirs(self.img_dir)
        if not exists(self.i_msk_dir):
            makedirs(self.i_msk_dir)
        if not exists(self.o_msk_dir):
            makedirs(self.o_msk_dir)
        for _, patient in self.linkfile.iterrows():
            dicom_id = patient['patient_id']
            contour_id = patient['original_id']
            to_check = self.parse_patient(dicom_id, contour_id, check_mask)
            masks_to_check.extend(to_check)
        return masks_to_check

    def parse_patient(self, dicom_id, contour_id, check_mask=lambda mask, image:True):
        """
        given the patient id to identify the dicom sub-dir and the contour sub-dir,
        generate and save parsed images and masks into the processed_dir. image
        and mask(s) (could have both inner and outer masks, or just inner mask) from the
        same patient and the same slice are saved with the same name:
        dicom_id-sclice_num.jpeg, but under different
        directories. Saved image pixel depth option is detailed here:
        https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
        :param dicom_id: dicom sub dir that contains all image slices for this patient
        :param contour_id: contour sub dir that contains all labeled masks, each corresponds
        to one slice of the patient. Each could contain either just inner or both inner and outer contours.
        :param check_mask: function to check if the mask is correct, takes mask and image as arguments,
        default to no check and always returns true
        :return: a list of inner mask file names that do not pass the sanity check
        """
        # read in all slice numbers that have contour, then read in corresponding image slices
        i_contour_fnames = listdir(join(self.target_root_dir, contour_id, 'i-contours'))
        to_check_masks = []
        for icontour_name in i_contour_fnames:
            slice_num = int(icontour_name.split('-')[2])
            dicom_name = str(slice_num) + '.dcm'
            ocontour_name = icontour_name.replace('icontour', 'ocontour')
            try:
                image = parse_dicom_file(join(self.img_root_dir, dicom_id, dicom_name))['pixel_data']
            except FileNotFoundError:
                continue
            img_dst = join(self.img_dir, dicom_id + '-' + str(slice_num) + '.jpeg')
            i_mask_dst = join(self.i_msk_dir, dicom_id + '-' + str(slice_num) + '.jpeg')
            i_contour = parse_contour_file(join(self.target_root_dir, contour_id, 'i-contours', icontour_name))
            o_contour = parse_contour_file(join(self.target_root_dir, contour_id, 'o-contours', ocontour_name))
            i_mask = poly_to_mask(i_contour, image.shape[1], image.shape[0])
            if check_mask(i_mask, image):
                Image.fromarray(image).convert('L').save(img_dst)
                Image.fromarray(np.uint8(255*i_mask)).convert('L').save(i_mask_dst)
            else:
                to_check_masks.append(join(self.target_root_dir, contour_id, 'i-contours', icontour_name))
            if o_contour:
                o_mask = poly_to_mask(o_contour, image.shape[1], image.shape[0])
                o_mask_dst = join(self.o_msk_dir, dicom_id + '-' + str(slice_num) + '.jpeg')
                Image.fromarray(np.uint8(255*o_mask)).convert('L').save(o_mask_dst)
        return to_check_masks

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
