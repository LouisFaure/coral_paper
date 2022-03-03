# coral_paper

Code for reproducibility of 2D track analysis on several coral species, run the following code on a linux machine with python installed:

```bash
# Install depencencies
pip install -r requirements.txt
cd helpers && git clone https://github.com/nivan/vfkm
cd vfkm/src && make && cd ../../../
```

## Track analysis

### Dynamics

```bash
wget https://figshare.com/ndownloader/files/34277030 --output-document tracks.tar.gz
tar xvf tracks.tar.gz

# Run analysis
for f in Track_Data_processed/*; do
    echo $f
    python -W ignore analysis_tracks.py "$f"
done
```

To visualise a particular replicate from a species, run the following python code:

```python
import matplotlib.pyplot as plt
im=plt.imread("Track_Data_processed/M. foliosa/2020-02-18_foliosa_1_stats/REPORT.png")
fig = plt.figure(figsize=(im.shape[1]/200, im.shape[0]/200), dpi=50)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(im)
```

### UMAP and clustering analyis

Run the following notebook:

## Neurotransmitter analysis

```bash
wget https://figshare.com/ndownloader/files/34277312--output-document neuro_tracks.tar.gz
tar xvf neuro_tracks.tar.gz

# Run analysis
for f in Neuro_Data_processed/*; do
    echo $f
    python -W ignore neuro_analysis.py "$f"
done
```

## Revision analysis

```bash
wget https://figshare.com/ndownloader/files/34277333--output-document revision_stat_processed.tar.gz
tar xvf revision_stat_processed.tar.gz
```

Then run the following notebook:
