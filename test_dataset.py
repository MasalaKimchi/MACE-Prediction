import os
import numpy as np
import pandas as pd
import tempfile
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from datetime import datetime
from dataset import SurvivalDataset, MonaiSurvivalDataset

def create_dummy_dicom(path, shape=(16, 16, 2)):
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

def create_dummy_csv(tmp_dir, n_samples=2):
    dicom_dirs = [os.path.join(tmp_dir, f"dicom_{i}") for i in range(n_samples)]
    for d in dicom_dirs:
        create_dummy_dicom(d, shape=(16, 16, 2))
    df = pd.DataFrame({
        "Dicom path": dicom_dirs,
        "time": np.random.randint(1, 100, size=n_samples),
        "event": np.random.randint(0, 2, size=n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
    })
    csv_path = os.path.join(tmp_dir, "dummy.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def test_survival_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = create_dummy_csv(tmp_dir)
        ds = SurvivalDataset(csv_path=csv_path, feature_cols=["feature1", "feature2"], image_size=(16, 16, 2), augment=False)
        x, t, e, image, dicom_path = ds[0]
        print(f"SurvivalDataset: x={x.shape}, t={t}, e={e}, image={image.shape}, dicom_path={dicom_path}")
        ds2 = MonaiSurvivalDataset(csv_path=csv_path, feature_cols=["feature1", "feature2"], image_size=(16, 16, 2), use_cache=False, augment=False)
        x2, t2, e2, image2, dicom_path2 = ds2[0]
        print(f"MonaiSurvivalDataset: x={x2.shape}, t={t2}, e={e2}, image={image2.shape}, dicom_path={dicom_path2}")

def main():
    test_survival_dataset()

if __name__ == "__main__":
    main() 