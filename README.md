# coral_paper

Code for reproducibility of 2D track analysis on several coral species, run the following code on a linux machine with python installed:

```bash
# Install depencencies
pip install -r requirements.txt
cd helpers && git clone https://github.com/nivan/vfkm
cd vfkm/src && make && cd ../../../

# Obtain and extract data
wget https://ndownloader.figshare.com/files/28714086 --output-document Track_Data.tar.gz
tar xvf Track_Data.tar.gz

# Run analysis
for f in Track_Data/*; do
    echo $f
	python preprocess_imaris.py $f
	python -W ignore analysis_tracks.py $f
done
```

To visualise a particular replicate from a species, run the following python code:

```python
import matplotlib.pyplot as plt
im=plt.imread("Track_Data/Montipora/2020-02-18_montipora_1_stats/REPORT.png")
fig = plt.figure(figsize=(im.shape[1]/200, im.shape[0]/200), dpi=50)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(im)
```
