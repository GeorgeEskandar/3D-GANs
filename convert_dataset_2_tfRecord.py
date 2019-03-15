import os
import random
from rep_medio.medio import convert_tf
from rep_medio.medio import read_dcm
import json
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
#=================================================#
#      Convert dicom files to .nii files
#=================================================#

dir_path = "Brain_DICOM_Data"
# for i in range(1, 16):
#     if i!=3:
#         path1 = dir_path + "\\" + str(i) + "\\t1_tse_tra_Kopf_0002_extended"
#         path2 = dir_path + "\\" + str(i) + "\\t1_tse_tra_Kopf_Motion_0003_extended"
#         read_dcm.convert_dir2nifti(path1, "Brain_DICOM_Data\\Nii\\Nii"+str(i))
#         read_dcm.convert_dir2nifti(path2, "Brain_DICOM_Data\\Nii\\Nii"+str(i))
# read_dcm.convert_dir2nifti("Brain_DICOM_Data\\16\\t1_tse_tra_Kopf_0002_extended", "Brain_DICOM_Data\\Nii\\Nii16")
# read_dcm.convert_dir2nifti("Brain_DICOM_Data\\16\\t1_tse_tra_Kopf_Motion_0003_extended", "Brain_DICOM_Data\\Nii\\Nii16")

#=================================================#
#      Convert .nii files to tfrecords
#=================================================#

nii_dataset_dir = dir_path + "\\nii_dataset"
# for filename in os.listdir(nii_dataset_dir):
#     file_path = os.path.join(nii_dataset_dir, filename)
#     print(file_path)
#     img = nib.load(file_path).get_fdata()
#     filename_base = os.path.splitext(filename)[0]
#     convert_tf.im2tfrecord(img, dir_path + "\\tfrecord_dataset\\" + filename_base + ".tfrecord")
#     print(dir_path + "\\tfrecord_dataset\\" + filename_base + ".tfrecord")

#=================================================#
#       Read tfRecords
#=================================================#
tfrecord_dataset_dir = dir_path + "\\tfrecord_dataset"
reader = tf.TFRecordReader()
# print(os.listdir(tfrecord_dataset_dir))
filename_queue = tf.train.string_input_producer([tfrecord_dataset_dir + "\\id1motion.tfrecord"])
__, serialized_example = reader.read(filename_queue)
img = convert_tf.parse_function(serialized_example)
sess = tf.InteractiveSession()
tf.train.start_queue_runners(sess)
print("inside session")
print(img.eval().shape)

plt.imshow(img.eval()[:,:,5])
plt.show()


