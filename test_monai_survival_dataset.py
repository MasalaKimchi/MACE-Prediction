import os
import numpy as np
import pandas as pd
import tempfile
import shutil
import torch
import pydicom
from pydicom.dataset import Dataset as DicomDataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from datetime import datetime
from dataset import MonaiSurvivalDataset

# Create a temporary directory for dummy DICOM files
tmp_dir = tempfile.mkdtemp()

def create_dummy_dicom(path, shape=(16, 16, 8)):
    arr = np.random.randint(-1000, 1000, size=shape, dtype=np.int16)
    series_dir = os.path.join(path)
    os.makedirs(series_dir, exist_ok=True)
    for i in range(shape[2]):
        slice_path = os.path.join(series_dir, f"slice_{i:03d}.dcm")
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(slice_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = "CT"
        ds.ContentDate = str(datetime.now().date()).replace('-', '')
        ds.ContentTime = str(datetime.now().time()).replace(':', '').replace('.', '')
        ds.Rows, ds.Columns = shape[0], shape[1]
        ds.InstanceNumber = i + 1
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = arr[:, :, i].tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(slice_path)
    return series_dir

# Create dummy data
n_samples = 3
dicom_dirs = []
for i in range(n_samples):
    dicom_dir = os.path.join(tmp_dir, f"dicom_{i}")
    create_dummy_dicom(dicom_dir)
    dicom_dirs.append(dicom_dir)

# Create a dummy CSV file
df = pd.DataFrame({
    "Dicom path": dicom_dirs,
    "time": np.random.randint(1, 1000, size=n_samples),
    "event": np.random.randint(0, 2, size=n_samples),
    "feature1": np.random.randn(n_samples),
    "feature2": np.random.randn(n_samples),
})
csv_path = os.path.join(tmp_dir, "dummy.csv")
df.to_csv(csv_path, index=False)

# Test MonaiSurvivalDataset
ds = MonaiSurvivalDataset(
    csv_path=csv_path,
    feature_cols=["feature1", "feature2"],
    image_size=(16, 16, 8),
    use_cache=False,  # For testing, avoid caching
    augment=False
)

print(f"Dataset length: {len(ds)}")
for i in range(len(ds)):
    x, t, e, image, dicom_path = ds[i]
    print(f"Sample {i}:")
    print(f"  Features: {x}")
    print(f"  Time: {t}")
    print(f"  Event: {e}")
    print(f"  Image shape: {image.shape}")
    print(f"  DICOM path: {dicom_path}")

# Cleanup
def cleanup():
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    cleanup() 