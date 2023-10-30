# Virtual-Try-On-App

Our project will be focusing on designing a 2D image-based virtual try on system. Virtual try on consists in generating an image of a reference person wearing a given try-on garment. This kind of problem is usually solved with a two-stage approach,  incorporating at least both a geometric transformation module to warp the selected garment and a generative try-on module to reconstruct the realistic try-on image given the person representation and the warped cloth.
We propose a complete pipeline built on this system performing in-the-wild virtual-try-on, consisting in image enhancing, background removal, a content-based retrieval system, cloth warping & try-on and a final super-resolution upscaling based on StableDiffusion.

The Warping and Try-On networks are trained by us through the DressCode dataset provided by Unimore. Substantial effort was put into dataset preprocessing and network adaptation. Moreover, our generative network was provided with a transformer-based block for establishing global mutual dependencies between the cloth and the person representations.
We trained both our transformer-based generative module and another similar module and compared the outputs on a common test set, with results demonstrating better performances for the former.

A complete demo of the pipeline can found in this [notebook](https://github.com/felicia-puzone/virtual-try-on-app/blob/main/Inference_pipeline_presentation.ipynb).
The requested checkpoints are stored in our Google Drive space:
- [CIT.pth](https://drive.google.com/file/d/1OrKhHYulzqpHO4qoCtxfqMKkaRnx_mwV/view?usp=drive_link)
- [SCHP](https://drive.google.com/file/d/1-3zM2BQ64kdjtnYu8fy7gQlo41NrowIp/view?usp=drive_link)
- [Geometric module - Upper Body](https://drive.google.com/file/d/1rU1wowreyZB2Wcq27zpopuN332cf6w9z/view?usp=drive_link)
- [Geometric module - Dresses](https://drive.google.com/file/d/1xgKCJtUeOU8-EC0gsMvIUKgnBTYdZADG/view?usp=drive_link)
- [Retrieval Net - Upper Body])(https://drive.google.com/file/d/1xgKCJtUeOU8-EC0gsMvIUKgnBTYdZADG/view?usp=drive_link)
- [Generative Module - CPVTON+](https://drive.google.com/file/d/1-H5Ht5aIx2dkXZ5pr7sSNH-4sP4XQ52p/view?usp=drive_link)
- [Generative Module - CIT - Dresses](https://drive.google.com/file/d/1xaXEdQCcv9nbQvp8SzoABgQOOvQFV_vv/view?usp=drive_link)

Other files:
- [Retrieval Repository](https://drive.google.com/file/d/1xaXEdQCcv9nbQvp8SzoABgQOOvQFV_vv/view?usp=drive_link)
- [DressCode5.0](https://drive.google.com/file/d/1xgj8co1LbH4vgpiPcWtRx7YXo2TatN2q/view?usp=drive_link)
