# SCMNet_SR


</details> 


---

⚙️ Requirements
---

- [PyTorch >= 1.8](https://pytorch.org/)
- [BasicSR >= 1.3.5](https://github.com/xinntao/BasicSR-examples/blob/master/README.md) 


🎈 Datasets
---

*Training*: [DF2K](https://openmmlab.medium.com/awesome-datasets-for-super-resolution-introduction-and-pre-processing-55f8501f8b18).

*Testing*: Set5, Set14, BSD100, Urban100, Manga109 ([Google Drive](https://drive.google.com/file/d/1SbdbpUZwWYDIEhvxQQaRsokySkcYJ8dq/view?usp=sharing)/[Baidu Netdisk](https://pan.baidu.com/s/1zfmkFK3liwNpW4NtPnWbrw?pwd=nbjl)).

*Preparing*: Please refer to the [Dataset Preparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) of BasicSR.



▶️ Train and Test
---

The [BasicSR](https://github.com/XPixelGroup/BasicSR) framework is utilized to train our SCMNet, also testing. Please refer to the usage of basicsr.

#### Training with the example option

```
# train SCMNet for x4 effieicnt SR
python basicsr/train.py -opt options/SCCM_DF2K_100w_x4SR.yml
```

#### Testing with the example option

```
python basicsr/test.py -opt options/SCCM_DF2K_x4SR.yml
```

