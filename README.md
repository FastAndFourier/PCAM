# PCAM
Contribution to the PatchCamelyon Challenge (semester project in image processing)

The goal of this challenge is to detect metastatic cells in sentinel lymph  node for more accurate diagnosis of breast cancer.
Most contributors have developed effective models but few have produced explainable data. It is indeed crucial to provide histopathologists with readable data, namely heat maps, on which they can rely for their diagnosis.

Along with two colleagues, we proposed two methods: a CNN + LIME-based approach and a texture-based approach using Gabor filters.
We've reached a AUC of 0.94 using the first method and a AUC of 0.83 using the second one. 

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

`utilitiesGabor`: helper functions to generate filterbanks, extract texture features and classify.

`model16_AUC85.pkl`: pre-trained Random Forest with 0.85 AUC, taking texture feature vectors as input 
